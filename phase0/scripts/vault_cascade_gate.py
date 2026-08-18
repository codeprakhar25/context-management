#!/usr/bin/env python3
"""Cascade placer over every corpus-B vault.

Default: note@20 on the item split, path@20 on the folder-disjoint split.
path@50 is nearly the full tree on these vaults (mean 39 folders, only 5
vaults have >50) so it is not a default — pass --folder-n 50 to spend it.

Check parse_fail in each summary before reading path_exact. Pools per item.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_one(
    snap: Path, split: str, mode: str, n: int, model: str, provider: str,
    workers: int, out: Path,
) -> dict | None:
    pre = "fold_" if split == "folder" else ""
    train, val, store = snap / f"{pre}train.jsonl", snap / f"{pre}val.jsonl", snap / "hierstore.sqlite"
    if not (train.exists() and val.exists() and store.exists()):
        return None
    if (out / "summary.json").exists():
        return json.loads((out / "summary.json").read_text())
    p = subprocess.run(
        [sys.executable, "scripts/cascade_placer.py",
         "--train", str(train), "--val", str(val), "--store", str(store),
         "--mode", mode, "--n", str(n),
         "--model", model, "--provider", provider,
         "--workers", str(workers), "--out", str(out)],
        capture_output=True, text=True, cwd=ROOT, timeout=3600,
    )
    if p.returncode != 0:
        print(f"    FAIL {snap.name} {split}: {p.stderr[-400:]}", file=sys.stderr)
        return None
    return json.loads((out / "summary.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--item-mode", default="note")
    ap.add_argument("--item-n", type=int, default=20)
    ap.add_argument("--folder-mode", default="path")
    ap.add_argument("--folder-n", type=int, default=20)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only-split", choices=["item", "folder", "both"], default="both")
    args = ap.parse_args()

    snaps = sorted(d for d in args.build.iterdir()
                   if d.is_dir() and (d / "hierstore.sqlite").exists())
    args.out.mkdir(parents=True, exist_ok=True)
    per_vault: dict[str, dict] = {}

    jobs = []
    if args.only_split in ("item", "both"):
        jobs.append(("item", args.item_mode, args.item_n))
    if args.only_split in ("folder", "both"):
        jobs.append(("folder", args.folder_mode, args.folder_n))

    for split, mode, n in jobs:
        for i, snap in enumerate(snaps, 1):
            tag = f"{snap.name}__{split}_{mode}{n}"
            r = run_one(snap, split, mode, n, args.model, args.provider,
                        args.workers, args.out / tag)
            if r:
                per_vault.setdefault(snap.name, {})[split] = r
                pf = r.get("n_parse_fail", 0)
                flag = f"  PARSE_FAIL={pf}" if pf else ""
                print(f"  [{split} {i}/{len(snaps)}] {snap.name}  "
                      f"exact={r.get('path_exact')}  recall={r.get('candidate_recall')}"
                      f"{flag}", flush=True)
            else:
                print(f"  [{split} {i}/{len(snaps)}] {snap.name}  FAIL", flush=True)

    pooled: dict[str, dict] = {}
    for split, _, _ in jobs:
        rows = [v[split] for v in per_vault.values() if split in v]
        tot = sum(r["n"] for r in rows) or 1
        pooled[split] = {
            "n_val": tot,
            "n_vaults": len(rows),
            "n_parse_fail": sum(r.get("n_parse_fail", 0) for r in rows),
            "path_exact": round(sum(r["path_exact"] * r["n"] for r in rows) / tot, 4),
            "path_soft": round(sum(r["path_soft"] * r["n"] for r in rows) / tot, 4),
            "candidate_recall": round(
                sum(r["candidate_recall"] * r["n"] for r in rows) / tot, 4
            ),
            "exact_when_gold_in_shortlist": round(
                sum(r["exact_when_gold_in_shortlist"] * r["n"] for r in rows) / tot, 4
            ),
            "mean_prompt_tokens": round(
                sum(r.get("mean_prompt_tokens", 0) * r["n"] for r in rows) / tot, 1
            ),
        }

    (args.out / "pooled.json").write_text(json.dumps(
        {"pooled": pooled, "per_vault": {
            name: {sp: {k: v[sp][k] for k in (
                "n", "path_exact", "path_soft", "candidate_recall",
                "exact_when_gold_in_shortlist", "n_parse_fail", "mean_prompt_tokens",
            ) if k in v[sp]} for sp in v}
            for name, v in per_vault.items()
        }}, indent=2) + "\n")

    for split, p in pooled.items():
        print(f"\n=== {split}  {p['n_vaults']} vaults  {p['n_val']} items  "
              f"parse_fail={p['n_parse_fail']} ===")
        print(f"  exact {p['path_exact']:.3f}  soft {p['path_soft']:.3f}  "
              f"recall {p['candidate_recall']:.3f}  "
              f"exact|in-list {p['exact_when_gold_in_shortlist']:.3f}  "
              f"tok/call {p['mean_prompt_tokens']}")


if __name__ == "__main__":
    main()
