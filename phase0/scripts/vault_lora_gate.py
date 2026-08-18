#!/usr/bin/env python3
"""LoRA placer eval over every corpus-B vault, one split, one deployed model.

Each vault keeps its own store, so existing_dirs must be read per vault, not
shared. Calls eval_fireworks_placer.py once per vault with --store pointed at
that vault's hierstore.sqlite. Pools per item, same as the other vault gates.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_one(snap: Path, split: str, model: str, provider: str, workers: int, out: Path) -> dict | None:
    pre = "fold_" if split == "folder" else ""
    val, store = snap / f"{pre}val.jsonl", snap / "hierstore.sqlite"
    if not (val.exists() and store.exists()):
        return None
    if (out / "summary.json").exists():
        return json.loads((out / "summary.json").read_text())
    p = subprocess.run(
        [sys.executable, "scripts/eval_fireworks_placer.py",
         "--tasks", str(val), "--store", str(store),
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
    ap.add_argument("--split", choices=["item", "folder"], required=True)
    ap.add_argument("--model", required=True, help="model#deployment route")
    ap.add_argument("--provider", default="fireworks")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    snaps = sorted(d for d in args.build.iterdir()
                   if d.is_dir() and (d / "hierstore.sqlite").exists())
    args.out.mkdir(parents=True, exist_ok=True)
    per_vault: dict[str, dict] = {}

    for i, snap in enumerate(snaps, 1):
        r = run_one(snap, args.split, args.model, args.provider, args.workers,
                     args.out / snap.name)
        if r:
            per_vault[snap.name] = r
            pf = r.get("parse_fail", 0)
            flag = f"  PARSE_FAIL={pf}" if pf else ""
            print(f"  [{i}/{len(snaps)}] {snap.name}  exact={r.get('path_exact')}"
                  f"{flag}", flush=True)
        else:
            print(f"  [{i}/{len(snaps)}] {snap.name}  FAIL", flush=True)

    rows = list(per_vault.values())
    tot = sum(r["n"] for r in rows) or 1
    pooled = {
        "split": args.split,
        "n_val": tot,
        "n_vaults": len(rows),
        "n_parse_fail": sum(r.get("parse_fail", 0) for r in rows),
        "path_exact": round(sum(r["path_exact"] * r["n"] for r in rows) / tot, 4),
        "path_soft": round(sum(r["path_soft"] * r["n"] for r in rows) / tot, 4),
        "mean_prompt_tokens": round(
            sum(r.get("mean_prompt_tokens", 0) * r["n"] for r in rows) / tot, 1
        ),
    }
    (args.out / "pooled.json").write_text(json.dumps(
        {"pooled": pooled, "per_vault": per_vault}, indent=2) + "\n")

    print(f"\n=== {args.split}  {pooled['n_vaults']} vaults  {pooled['n_val']} items  "
          f"parse_fail={pooled['n_parse_fail']} ===")
    print(f"  exact {pooled['path_exact']:.3f}  soft {pooled['path_soft']:.3f}  "
          f"tok/call {pooled['mean_prompt_tokens']}")


if __name__ == "__main__":
    main()
