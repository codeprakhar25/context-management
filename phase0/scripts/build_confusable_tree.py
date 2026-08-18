#!/usr/bin/env python3
"""Synthetic hard-tree corpus where Flat ANN is tempted by wrong folder.

Design: many near-duplicate / shared-vocabulary facts across two work/*
projects. Distinctive tokens are thin; twins share templates. Some facts
are *identical text* in both folders (path is the only disambiguator).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.store import HierStore, Op  # noqa: E402

ROOTS = ["work", "personal", "inbox"]
PROJECTS = ["slm-lab", "clothing-fit"]

# Shared templates — only {tag} differs (short, easy to bury in embed space)
SHARED = [
    ("deploy", "Deploy failed on Friday night after the config change"),
    ("quota", "Hit GPU quota mid-run; jobs paused until Monday"),
    ("metric", "Primary metric dropped after the last merge"),
    ("meeting", "Ship the ablation before the freeze next week"),
    ("bug", "NaNs appeared when batching long inputs"),
    ("bill", "Cloud bill spiked after leaving the pod warm overnight"),
    ("deadline", "Writeup blocked on the remaining eval plots"),
    ("rollback", "Rolled back the hot fix; staging is green again"),
]

# Identical text twins (same string, two folders) — path-only disambiguation
IDENTICAL = [
    ("alert", "PagerDuty fired twice during the overnight sweep"),
    ("note", "Action item: ping infra about disk pressure"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "confusable_tree")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    db = args.out / "hierstore.sqlite"
    if db.exists():
        db.unlink()

    store = HierStore(db, roots=ROOTS)
    ops: list[Op] = [
        Op(op="MKDIR", path=["personal", "pets"]),
        Op(
            op="ADD",
            fact_id="pers_buddy",
            text="Adopted a dog named Buddy last weekend",
            path=["personal", "pets"],
        ),
    ]
    queries = []

    for proj in PROJECTS:
        base = ["work", proj]
        ops.append(Op(op="MKDIR", path=base))
        ops.append(Op(op="MKDIR", path=base + ["results"]))

        for tname, text in SHARED:
            # thin tag suffix so not 100% identical, still confusable
            full = f"{text} [{proj}]"
            path = base + ["results"] if tname in ("metric", "bug", "rollback") else base
            fid = f"{proj}__{tname}"
            ops.append(Op(op="ADD", fact_id=fid, text=full, path=path))
            other = PROJECTS[1] if proj == PROJECTS[0] else PROJECTS[0]
            queries.append(
                {
                    "id": f"q_{fid}",
                    "question": text + "?",  # no project name in question
                    "active_path": base,
                    "gold_ids": [fid],
                    "twin_id": f"{other}__{tname}",
                    "project": proj,
                    "kind": "shared",
                }
            )

        for tname, text in IDENTICAL:
            fid = f"{proj}__{tname}"
            ops.append(
                Op(op="ADD", fact_id=fid, text=text, path=base + ["results"])
            )
            other = PROJECTS[1] if proj == PROJECTS[0] else PROJECTS[0]
            queries.append(
                {
                    "id": f"q_{fid}",
                    "question": text + " Which project's note is this?",
                    "active_path": base,
                    "gold_ids": [fid],
                    "twin_id": f"{other}__{tname}",
                    "project": proj,
                    "kind": "identical",
                }
            )

    store.apply_ops(ops, manager="build_confusable")
    (args.out / "queries.jsonl").write_text(
        "\n".join(json.dumps(q) for q in queries) + "\n"
    )
    meta = {
        "db": str(db),
        "n_facts": len(store.read_all()),
        "n_queries": len(queries),
        "roots": ROOTS,
        "projects": PROJECTS,
        "note": "Near-duplicate + identical-text twins; questions omit project name",
    }
    (args.out / "META.json").write_text(json.dumps(meta, indent=2))
    store.close()
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
