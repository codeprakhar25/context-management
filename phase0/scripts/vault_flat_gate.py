#!/usr/bin/env python3
"""gpt-4o flat placer over every corpus-B vault.

Lists the whole folder tree — no retrieval. The comparison cell for cascade.
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
    snap: Path, split: str, model: str, provider: str, workers: int, out: Path,
) -> dict | None:
    pre = "fold_" if split == "folder" else ""
    tasks, store = snap / f"{pre}val.jsonl", snap / "hierstore.sqlite"
    if not (tasks.exists() and store.exists()):
        return None
    if (out / "summary.json").exists():
        return json.loads((out / "summary.json").read_text())
    p = subprocess.run(
        [sys.executable, "scripts/eval_fireworks_placer.py",
         "--tasks", str(tasks), "--store", str(store),
         "--model", model, "--provider", provider,
         "--workers", str(workers), "--out", str(out)],
        capture_output=True, text=True, cwd=ROOT, timeout=3600,
    )
    if p.returncode != 0:
        print(f"    FAIL {snap.name} {split}: {p.stderr[-400:] or p.stdout[-400:]}",
              file=sys.stderr)
        return None
    return json.loads((out / "summary.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only-split", choices=["item", "folder", "both"], default="both")
    args = ap.parse_args()

    snaps = sorted(d for d in args.build.iterdir()
                   if d.is_dir() and (d / "hierstore.sqlite").exists())
    args.out.mkdir(parents=True, exist_ok=True)
    per_vault: dict[str, dict] = {}

    splits = ["item", "folder"] if args.only_split == "both" else [args.only_split]
    for split in splits:
        for i, snap in enumerate(snaps, 1):
            r = run_one(snap, split, args.model, args.provider, args.workers,
                        args.out / f"{snap.name}__{split}")
            if r:
                per_vault.setdefault(snap.name, {})[split] = r
                pf = r.get("parse_fail", 0)
                flag = f"  PARSE_FAIL={pf}" if pf else ""
                print(f"  [{split} {i}/{len(snaps)}] {snap.name}  "
                      f"exact={r.get('path_exact')}{flag}", flush=True)
            else:
                print(f"  [{split} {i}/{len(snaps)}] {snap.name}  FAIL", flush=True)

    pooled: dict[str, dict] = {}
    for split in splits:
        rows = [v[split] for v in per_vault.values() if split in v]
        tot = sum(r["n"] for r in rows) or 1
        pooled[split] = {
            "n_val": tot,
            "n_vaults": len(rows),
            "parse_fail": sum(r.get("parse_fail", 0) for r in rows),
            "path_exact": round(sum(r["path_exact"] * r["n"] for r in rows) / tot, 4),
            "path_soft": round(sum(r["path_soft"] * r["n"] for r in rows) / tot, 4),
            "mean_prompt_tokens": round(
                sum(r.get("mean_prompt_tokens", 0) * r["n"] for r in rows) / tot, 1
            ),
        }

    (args.out / "pooled.json").write_text(json.dumps(
        {"pooled": pooled, "per_vault": {
            name: {sp: {k: v[sp][k] for k in (
                "n", "path_exact", "path_soft", "parse_fail", "mean_prompt_tokens",
                "max_depth",
            ) if k in v[sp]} for sp in v}
            for name, v in per_vault.items()
        }}, indent=2) + "\n")

    for split, p in pooled.items():
        print(f"\n=== {split}  {p['n_vaults']} vaults  {p['n_val']} items  "
              f"parse_fail={p['parse_fail']} ===")
        print(f"  exact {p['path_exact']:.3f}  soft {p['path_soft']:.3f}  "
              f"tok/call {p['mean_prompt_tokens']}")


if __name__ == "__main__":
    main()
