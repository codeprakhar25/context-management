#!/usr/bin/env python3
"""kNN baseline over every corpus-B vault, both splits.

Embeddings only. On the folder-disjoint split this is 0 by construction — the
gold folder holds no training note — and that is the result. Pools per item.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_one(snap: Path, split: str, ks: list[int], cache: Path, out: Path) -> dict | None:
    pre = "fold_" if split == "folder" else ""
    train, val = snap / f"{pre}train.jsonl", snap / f"{pre}val.jsonl"
    if not (train.exists() and val.exists()):
        return None
    if (out / "summary.json").exists():
        return json.loads((out / "summary.json").read_text())
    p = subprocess.run(
        [sys.executable, "scripts/knn_placer_baseline.py",
         "--train", str(train), "--val", str(val),
         "--out", str(out), "--cache", str(cache),
         "--k", *[str(k) for k in ks]],
        capture_output=True, text=True, cwd=ROOT, timeout=1800,
    )
    if p.returncode != 0:
        print(f"    FAIL {snap.name} {split}: {p.stderr[-300:]}", file=sys.stderr)
        return None
    return json.loads((out / "summary.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 8, 12])
    ap.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "runs" / "_embed_cache" / "candidate_recall.json",
    )
    args = ap.parse_args()

    snaps = sorted(d for d in args.build.iterdir()
                   if d.is_dir() and (d / "hierstore.sqlite").exists())
    args.out.mkdir(parents=True, exist_ok=True)
    per_vault: dict[str, dict] = {}

    for split in ("item", "folder"):
        for i, snap in enumerate(snaps, 1):
            r = run_one(snap, split, args.k, args.cache,
                        args.out / f"{snap.name}__{split}")
            if r:
                per_vault.setdefault(snap.name, {})[split] = r
            print(f"  [{split} {i}/{len(snaps)}] {snap.name}", flush=True)

    pooled: dict[str, dict] = {}
    for split in ("item", "folder"):
        rows = [v[split] for v in per_vault.values() if split in v]
        tot = sum(r["n"] for r in rows) or 1
        by_k = {}
        for k in args.k:
            by_k[str(k)] = {
                m: round(sum(r["by_k"][str(k)][m] * r["n"] for r in rows) / tot, 4)
                for m in ("path_exact", "path_soft", "branch_ok")
            }
        pooled[split] = {"n_val": tot, "n_vaults": len(rows), "by_k": by_k}

    (args.out / "pooled.json").write_text(json.dumps(
        {"pooled": pooled, "per_vault": {
            name: {sp: {"n": v[sp]["n"], "by_k": v[sp]["by_k"]} for sp in v}
            for name, v in per_vault.items()
        }}, indent=2) + "\n")

    for split in ("item", "folder"):
        p = pooled[split]
        print(f"\n=== {split} split — {p['n_vaults']} vaults, {p['n_val']} val items ===")
        print(f"{'k':<6}" + "".join(f"{m:>12}" for m in ("exact", "soft", "branch")))
        for k in args.k:
            b = p["by_k"][str(k)]
            print(f"{k:<6}{b['path_exact']:12.3f}{b['path_soft']:12.3f}{b['branch_ok']:12.3f}")


if __name__ == "__main__":
    main()
