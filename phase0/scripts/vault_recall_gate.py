#!/usr/bin/env python3
"""Candidate-recall gate over every corpus-B vault, both splits.

A cascade can never beat the recall of its candidate generator, so this is the
ceiling on every cascade number corpus B will produce. Embeddings only -- run it
before spending anything on the picking half.

The two sources fail in opposite places, and the folder-disjoint column is the
whole reason both exist:
  note  folders of the nearest training notes. Cannot reach a folder that holds
        no training note, so it is 0.000 on folder-disjoint by construction.
  path  folders whose path string embeds nearest to the note. Reaches any folder
        in the tree.

Pools per-item across vaults (weighted by val size), because a 10-item vault and
a 100-item vault should not count equally.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_one(snap: Path, split: str, ns: list[int], out: Path) -> dict | None:
    pre = "fold_" if split == "folder" else ""
    train, val = snap / f"{pre}train.jsonl", snap / f"{pre}val.jsonl"
    store = snap / "hierstore.sqlite"
    if not (train.exists() and val.exists() and store.exists()):
        return None
    p = subprocess.run(
        [sys.executable, "scripts/candidate_recall.py",
         "--train", str(train), "--val", str(val), "--store", str(store),
         "--out", str(out), "--n", *[str(n) for n in ns]],
        capture_output=True, text=True, cwd=ROOT, timeout=1800,
    )
    if p.returncode != 0:
        print(f"    FAIL {snap.name} {split}: {p.stderr[-300:]}", file=sys.stderr)
        return None
    return json.loads(out.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[10, 20, 50])
    args = ap.parse_args()

    snaps = sorted(d for d in args.build.iterdir()
                   if d.is_dir() and (d / "hierstore.sqlite").exists())
    args.out.mkdir(parents=True, exist_ok=True)
    per_vault: dict[str, dict] = {}

    for split in ("item", "folder"):
        for i, snap in enumerate(snaps, 1):
            r = run_one(snap, split, args.n,
                        args.out / f"{snap.name}__{split}.json")
            if r:
                per_vault.setdefault(snap.name, {})[split] = r
            print(f"  [{split} {i}/{len(snaps)}] {snap.name}", flush=True)

    # pool per item, not per vault
    pooled: dict[str, dict] = {}
    for split in ("item", "folder"):
        pooled[split] = {}
        rows = [v[split] for v in per_vault.values() if split in v]
        tot = sum(r["n_val"] for r in rows) or 1
        for mode in ("note", "path", "union"):
            pooled[split][mode] = {
                str(n): round(
                    sum(r["recall"][mode][str(n)] * r["n_val"] for r in rows) / tot, 4
                )
                for n in args.n
            }
        pooled[split]["n_val"] = tot
        pooled[split]["n_vaults"] = len(rows)
        pooled[split]["mean_folders_per_vault"] = round(
            sum(r["n_folders_in_tree"] for r in rows) / max(len(rows), 1), 1
        )

    (args.out / "pooled.json").write_text(json.dumps(
        {"pooled": pooled, "per_vault": per_vault}, indent=2) + "\n")

    for split in ("item", "folder"):
        p = pooled[split]
        print(f"\n=== {split} split — {p['n_vaults']} vaults, {p['n_val']} val items, "
              f"{p['mean_folders_per_vault']} folders/vault ===")
        print(f"{'source':<8}" + "".join(f"{'@'+str(n):>9}" for n in args.n))
        for mode in ("note", "path", "union"):
            print(f"{mode:<8}" + "".join(f"{p[mode][str(n)]:>9.3f}" for n in args.n))


if __name__ == "__main__":
    main()
