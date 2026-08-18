#!/usr/bin/env python3
"""ASCII / JSON tree view of a HierStore (or materialize place_tasks gold)."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.store import HierStore, Op, path_key  # noqa: E402


def tree_from_store(store: HierStore, *, max_text: int = 60) -> dict:
    dirs = store.list_dirs()
    facts = store.read_all(valid_only=False)
    by_path: dict[str, list[dict]] = defaultdict(list)
    for f in facts:
        by_path[path_key(f["path"])].append(f)
    return {
        "roots": store.roots(),
        "max_depth": store.max_depth,
        "n_dirs": len(dirs),
        "n_facts_valid": sum(1 for f in facts if f["valid"]),
        "n_facts_total": len(facts),
        "dirs": dirs,
        "by_path": {
            k: [
                {
                    "id": f["id"],
                    "valid": f["valid"],
                    "text": (f["text"][:max_text] + ("…" if len(f["text"]) > max_text else "")),
                }
                for f in vs
            ]
            for k, vs in sorted(by_path.items())
        },
    }


def render_ascii(meta: dict, *, show_invalid: bool = False) -> str:
    """Render folder tree with fact leaves."""
    # build nested dict
    Node = dict  # name -> {"_kids": {}, "_facts": []}
    root: dict = {"_kids": {}, "_facts": []}

    def ensure(path: list[str]) -> dict:
        cur = root
        for seg in path:
            kids = cur.setdefault("_kids", {})
            cur = kids.setdefault(seg, {"_kids": {}, "_facts": []})
        return cur

    for d in meta["dirs"]:
        ensure(d)
    for pk, facts in meta["by_path"].items():
        segs = [s for s in pk.split("/") if s]
        node = ensure(segs)
        for f in facts:
            if f["valid"] or show_invalid:
                node.setdefault("_facts", []).append(f)

    lines = [
        f"roots={meta['roots']}  dirs={meta['n_dirs']}  "
        f"facts_valid={meta['n_facts_valid']}/{meta['n_facts_total']}  "
        f"max_depth={meta['max_depth']}",
        "",
    ]

    def walk(node: dict, prefix: str, name: str, is_last: bool) -> None:
        branch = "└── " if is_last else "├── "
        if name:
            lines.append(prefix + branch + name + "/")
            child_prefix = prefix + ("    " if is_last else "│   ")
        else:
            child_prefix = prefix
        kids = sorted(node.get("_kids", {}).items())
        facts = node.get("_facts") or []
        # facts as leaves under this folder
        items = [("dir", k, v) for k, v in kids] + [("fact", f["id"], f) for f in facts]
        for i, item in enumerate(items):
            last = i == len(items) - 1
            kind, key, val = item
            if kind == "dir":
                walk(val, child_prefix, key, last)
            else:
                mark = " " if val.get("valid", True) else "×"
                b = "└── " if last else "├── "
                lines.append(
                    f"{child_prefix}{b}[{mark}] {val['id']}: {val['text']}"
                )

    # top-level kids only (roots)
    kids = sorted(root.get("_kids", {}).items())
    if not kids:
        lines.append("(empty tree)")
    for i, (name, node) in enumerate(kids):
        walk(node, "", name, i == len(kids) - 1)
    return "\n".join(lines)


def materialize_place_tasks(tasks_path: Path) -> HierStore:
    tasks = [
        json.loads(l)
        for l in tasks_path.read_text().splitlines()
        if l.strip()
    ]
    roots = tasks[0].get("roots") if tasks else ["work", "personal", "inbox"]
    td = Path(tempfile.mkdtemp(prefix="viz_place_"))
    store = HierStore(td / "t.sqlite", roots=roots)
    for t in tasks:
        if t.get("kind") == "move":
            # final gold path only
            path = t["gold_path"]
        else:
            path = t["gold_path"]
        fid = t.get("fact_id") or f"f_{t['id']}"
        store.apply_ops(
            [
                Op(op="MKDIR", path=path),
                Op(op="ADD", fact_id=fid, text=t["text"], path=path),
            ],
            manager="viz",
        )
    return store


def locomo_view(store_path: Path) -> str:
    """LoCoMo is project-scoped shallow paths — summarize, don't dump 2.5k leaves."""
    store = HierStore(store_path)
    try:
        facts = store.read_all(valid_only=True)
        by_proj: dict[str, int] = defaultdict(int)
        for f in facts:
            by_proj[str(f.get("project"))] += 1
        lines = [
            "=== LoCoMo hierstore (current AlwaysADD bank) ===",
            f"n_facts_valid={len(facts)}  note: paths are mostly ['project', conv_id]",
            "This is NOT the hard work/personal/inbox tree — old Flat vs Hier arm.",
            "",
            "conversation / project counts:",
        ]
        for p, n in sorted(by_proj.items(), key=lambda x: -x[1]):
            lines.append(f"  project/{p}/  ({n} facts)")
        lines.append("")
        lines.append("sample paths:")
        for f in facts[:5]:
            lines.append(f"  {path_key(f['path'])}  ::  {f['text'][:50]}…")
        return "\n".join(lines)
    finally:
        store.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument(
        "--place-tasks",
        type=Path,
        default=ROOT / "data" / "storage_oracle" / "place_tasks.jsonl",
        help="materialize gold place_tasks tree if no --store",
    )
    ap.add_argument("--locomo", action="store_true", help="summarize LoCoMo store")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--ascii-out", type=Path, default=None)
    args = ap.parse_args()

    chunks: list[str] = []

    if args.locomo or not args.store:
        locomo_path = ROOT / "data" / "locomo" / "hierstore.sqlite"
        if locomo_path.exists():
            chunks.append(locomo_view(locomo_path))
            chunks.append("")

    store = None
    tmp_close = False
    if args.store:
        store = HierStore(args.store)
    else:
        # default: show intended hard-tree shape from place_tasks gold
        store = materialize_place_tasks(args.place_tasks)
        tmp_close = True
        chunks.append(
            "=== Hard-tree shape (place_tasks GOLD materialized) ===\n"
            "This is the intended product tree vocabulary — not LoCoMo.\n"
        )

    try:
        meta = tree_from_store(store)
        ascii_tree = render_ascii(meta)
        chunks.append(ascii_tree)
        text = "\n".join(chunks)
        print(text)
        if args.ascii_out:
            args.ascii_out.parent.mkdir(parents=True, exist_ok=True)
            args.ascii_out.write_text(text + "\n")
            print("wrote", args.ascii_out, file=sys.stderr)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(meta, indent=2))
            print("wrote", args.json_out, file=sys.stderr)
    finally:
        store.close()


if __name__ == "__main__":
    main()
