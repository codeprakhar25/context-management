#!/usr/bin/env python3
"""Freeze a user-dir snapshot into HierStore hard tree.

Reads data/user_dir_snap/sources.json (logical root → filesystem paths).
Writes:
  data/user_dir_snap/hierstore.sqlite
  data/user_dir_snap/META.json
  data/user_dir_snap/place_tasks_from_snap.jsonl   # gold path = file folder (SFT later)
  runs/tree_viz_user_dir.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.store import HierStore, Op, MAX_DEPTH, normalize_path  # noqa: E402

DEFAULT_CFG = ROOT / "data" / "user_dir_snap" / "sources.json"

# local import of viz helpers (scripts/ not a package)
import importlib.util

_viz_spec = importlib.util.spec_from_file_location(
    "viz_tree", ROOT / "scripts" / "viz_tree.py"
)
_viz = importlib.util.module_from_spec(_viz_spec)
assert _viz_spec and _viz_spec.loader
_viz_spec.loader.exec_module(_viz)
tree_from_store = _viz.tree_from_store
render_ascii = _viz.render_ascii


def slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "x"


def iter_files(
    src: Path,
    *,
    extensions: set[str],
    skip_dirs: set[str],
) -> list[Path]:
    if not src.exists():
        return []
    if src.is_file():
        return [src] if src.suffix.lower() in extensions else []
    out: list[Path] = []
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        if any(part in skip_dirs for part in p.parts):
            continue
        if p.suffix.lower() not in extensions:
            continue
        out.append(p)
    return sorted(out)


def path_for_file(
    logical_root: str,
    src_root: Path,
    file_path: Path,
    max_depth: int,
) -> list[str] | None:
    """Map file → hard-tree path (folder containing file), capped at max_depth."""
    try:
        rel = file_path.relative_to(src_root)
    except ValueError:
        rel = Path(file_path.name)
    parts = [slug(logical_root)]
    # area = source folder name, unless it duplicates the logical root
    area = slug(src_root.name)
    if area and area != slug(logical_root):
        parts.append(area)
    parent_parts = [slug(x) for x in rel.parent.parts if x not in (".", "")]
    parts.extend(parent_parts)
    # cap depth: keep root + as much tail as fits
    if len(parts) > max_depth:
        # keep logical root + last (max_depth-1) segments
        parts = [parts[0]] + parts[-(max_depth - 1) :]
    try:
        return normalize_path(parts)
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "user_dir_snap",
    )
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    roots = cfg["roots"]
    max_depth = int(cfg.get("max_depth") or MAX_DEPTH)
    max_bytes = int(cfg.get("max_file_bytes") or 80_000)
    max_per_root = int(cfg.get("max_files_per_root") or 80)
    extensions = {e.lower() for e in cfg.get("extensions") or [".md", ".txt"]}
    skip_dirs = set(cfg.get("skip_dir_names") or [])

    args.out.mkdir(parents=True, exist_ok=True)
    db = args.out / "hierstore.sqlite"
    if db.exists():
        db.unlink()

    store = HierStore(db, roots=roots, max_depth=max_depth)
    ops: list[Op] = []
    place_tasks: list[dict] = []
    skipped: list[dict] = []
    per_root_count: dict[str, int] = {r: 0 for r in roots}

    for logical_root, src_list in (cfg.get("sources") or {}).items():
        if logical_root not in roots:
            skipped.append({"reason": "unknown_root", "root": logical_root})
            continue
        for src_s in src_list:
            src = Path(src_s).expanduser()
            files = iter_files(src, extensions=extensions, skip_dirs=skip_dirs)
            for fp in files:
                if per_root_count[logical_root] >= max_per_root:
                    skipped.append(
                        {
                            "reason": "max_files_per_root",
                            "root": logical_root,
                            "file": str(fp),
                        }
                    )
                    continue
                try:
                    raw = fp.read_bytes()
                except OSError as e:
                    skipped.append({"reason": str(e), "file": str(fp)})
                    continue
                if len(raw) > max_bytes:
                    skipped.append(
                        {
                            "reason": "too_large",
                            "file": str(fp),
                            "bytes": len(raw),
                        }
                    )
                    continue
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    skipped.append({"reason": "not_utf8", "file": str(fp)})
                    continue
                text = text.strip()
                if not text:
                    skipped.append({"reason": "empty", "file": str(fp)})
                    continue
                # leaf fact text = first ~2k chars + path hint
                body = text if len(text) <= 2000 else text[:2000] + "…"
                path = path_for_file(logical_root, src, fp, max_depth)
                if path is None or path[0] not in roots:
                    skipped.append({"reason": "bad_path", "file": str(fp)})
                    continue
                digest = hashlib.sha1(str(fp).encode()).hexdigest()[:10]
                fid = f"{logical_root}_{digest}"
                ops.append(Op(op="MKDIR", path=path))
                ops.append(
                    Op(
                        op="ADD",
                        fact_id=fid,
                        text=body,
                        path=path,
                        kind="file_note",
                        t=datetime.fromtimestamp(
                            fp.stat().st_mtime, tz=timezone.utc
                        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    )
                )
                place_tasks.append(
                    {
                        "id": f"snap_{fid}",
                        "kind": "place",
                        "text": body[:500],
                        "gold_path": path,
                        "roots": roots,
                        "source_file": str(fp),
                        "frozen": True,
                    }
                )
                per_root_count[logical_root] += 1

    if ops:
        # dedupe consecutive MKDIRs by applying in one batch (store mkdir -p idempotent)
        store.apply_ops(ops, manager="user_dir_snapshot")

    meta = {
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": str(args.config),
        "db": str(db),
        "roots": roots,
        "max_depth": max_depth,
        "n_facts": len(store.read_all()),
        "n_dirs": len(store.list_dirs()),
        "per_root_files": per_root_count,
        "n_place_tasks": len(place_tasks),
        "n_skipped": len(skipped),
        "skip_reasons": {},
        "note": "Frozen snapshot for path fidelity / subtree probes / later SFT labels. Re-run to refresh.",
        "sft_hint": "place_tasks_from_snap.jsonl gold_path = file folder; train placer later",
    }
    from collections import Counter

    meta["skip_reasons"] = dict(Counter(s["reason"] for s in skipped))

    (args.out / "META.json").write_text(json.dumps(meta, indent=2))
    (args.out / "skipped.jsonl").write_text(
        "\n".join(json.dumps(s) for s in skipped[:500])
        + ("\n" if skipped else "")
    )
    (args.out / "place_tasks_from_snap.jsonl").write_text(
        "\n".join(json.dumps(t) for t in place_tasks) + ("\n" if place_tasks else "")
    )

    viz_meta = tree_from_store(store, max_text=50)
    ascii_tree = render_ascii(viz_meta)
    viz_path = ROOT / "runs" / "tree_viz_user_dir.txt"
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"=== USER DIR SNAPSHOT (frozen {meta['frozen_at']}) ===\n"
        f"facts={meta['n_facts']} dirs={meta['n_dirs']} "
        f"per_root={per_root_count}\n\n"
    )
    viz_path.write_text(header + ascii_tree + "\n")
    store.close()

    print(json.dumps({k: meta[k] for k in meta if k != "config"}, indent=2))
    print("wrote", db)
    print("wrote", args.out / "place_tasks_from_snap.jsonl")
    print("wrote", viz_path)


if __name__ == "__main__":
    main()
