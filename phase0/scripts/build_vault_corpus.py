#!/usr/bin/env python3
"""Build corpus B: one placement benchmark per vault, both splits.

Each vault is a separate store rather than one pooled tree. A folksonomy belongs
to one person -- folder names collide across vaults ("projects", "daily"), and
the folder list a placer is shown should be that person's own. Pooling would
invent a hierarchy nobody actually uses.

Per vault this drives the existing pipeline unchanged:
  snapshot_user_dir.py    -> hierstore.sqlite + place tasks
  build_user_dir_split.py -> item-stratified split (folders seen)
  build_user_dir_split.py -> folder-disjoint split (folders unseen)

Results pool at the item level afterwards, which is what the coverage curve
needs -- accuracy against how many notes already sit in the target folder.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = [
    ".git", ".github", ".obsidian", ".trash", "node_modules", "__pycache__",
    ".venv", "venv", "site-packages", "attachments", "assets", "images",
    "img", "files", "excalidraw", ".smart-env", "templates", "_templates",
    "Templates", ".space", "z_assets", "99-Meta",
]


def run(args: list[str]) -> tuple[int, str]:
    p = subprocess.run(args, capture_output=True, text=True, timeout=1800, cwd=ROOT)
    return p.returncode, (p.stdout + p.stderr)[-2000:]


def build_one(v: dict, raw: Path, out_root: Path, args) -> dict:
    name = v["full_name"]
    slug = name.replace("/", "__")
    src = raw / slug
    if not src.exists():
        return {"repo": name, "status": "missing_clone"}

    snap = out_root / slug
    snap.mkdir(parents=True, exist_ok=True)
    # Each vault gets a single logical root named for the repo, so gold paths
    # stay inside one namespace and cannot collide with another vault's.
    (snap / "sources.json").write_text(json.dumps({
        "note": f"corpus B vault {name} @ {v['commit']}",
        "roots": ["vault"],
        "max_depth": args.max_depth,
        "max_file_bytes": 120000,
        "max_files_per_root": args.max_notes,
        "extensions": [".md", ".markdown"],
        "skip_dir_names": SKIP_DIRS,
        "sources": {"vault": [str(src.resolve())]},
    }, indent=2) + "\n")

    code, log = run([sys.executable, "scripts/snapshot_user_dir.py",
                     "--config", str(snap / "sources.json"), "--out", str(snap)])
    if code != 0:
        return {"repo": name, "status": "snapshot_failed", "log": log[-600:]}

    for extra, prefix in (([], ""), (["--split-by", "folder"], "fold_")):
        cmd = [sys.executable, "scripts/build_user_dir_split.py",
               "--snap", str(snap), "--val-frac", str(args.val_frac)] + extra
        if prefix:
            cmd += ["--out-prefix", prefix]
        code, log = run(cmd)
        if code != 0:
            return {"repo": name, "status": f"split_failed{prefix and '_folder'}",
                    "log": log[-600:]}

    try:
        sp = json.loads((snap / "split.json").read_text())
    except Exception:  # noqa: BLE001
        sp = {}
    n = lambda f: sum(1 for _ in (snap / f).open()) if (snap / f).exists() else 0  # noqa: E731
    return {
        "repo": name, "status": "ok", "snap": str(snap),
        "n_clean": n("place_tasks_clean.jsonl"),
        "n_train": n("train.jsonl"), "n_val": n("val.jsonl"),
        "n_fold_train": n("fold_train.jsonl"), "n_fold_val": n("fold_val.jsonl"),
        "split_meta": {k: sp.get(k) for k in ("n_folders", "n_raw", "n_clean")},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.4)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--max-notes", type=int, default=250,
                    help="per-vault note cap; stops one large vault dominating")
    ap.add_argument("--only", default="", help="substring filter on repo name")
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text())
    vaults = [v for v in man["vaults"] if args.only in v["full_name"]]
    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    for i, v in enumerate(vaults, 1):
        r = build_one(v, args.raw, args.out, args)
        results.append(r)
        if r["status"] == "ok":
            print(f"[{i}/{len(vaults)}] {r['repo']:<44} clean {r['n_clean']:>4}  "
                  f"item {r['n_train']}/{r['n_val']}  "
                  f"fold {r['n_fold_train']}/{r['n_fold_val']}", flush=True)
        else:
            print(f"[{i}/{len(vaults)}] {r['repo']:<44} {r['status']}", flush=True)

    ok = [r for r in results if r["status"] == "ok"]
    (args.out / "build_report.json").write_text(json.dumps({
        "n_ok": len(ok), "n_failed": len(results) - len(ok),
        "total_clean": sum(r["n_clean"] for r in ok),
        "total_val_item": sum(r["n_val"] for r in ok),
        "total_val_fold": sum(r["n_fold_val"] for r in ok),
        "results": results,
    }, indent=2) + "\n")
    print(f"\n{len(ok)}/{len(results)} vaults built  "
          f"clean {sum(r['n_clean'] for r in ok)}  "
          f"val item {sum(r['n_val'] for r in ok)}  "
          f"val fold {sum(r['n_fold_val'] for r in ok)}")


if __name__ == "__main__":
    main()
