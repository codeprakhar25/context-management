#!/usr/bin/env python3
"""Tune RuleV0 thresholds on conflict_v1 split=dev ONLY.

Default: hash64 (cheap).
  --real: text-embedding-3-small (OpenAI) — use before trusting RuleV0 numbers.
Writes data/conflict_v1/rulev0_thresh.json (or --out).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from harness.embed import Embedder  # noqa: E402
from harness.manager import ManagerInput, RuleV0  # noqa: E402
from harness.store import HierStore  # noqa: E402
from harness.write_metrics import ops_match_lenient, ops_match_strict  # noqa: E402


def hash_embed(texts: list[str], dim: int = 64) -> np.ndarray:
    import re

    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for t in re.findall(r"[a-z0-9]+", text.lower()):
            out[i, hash(t) % dim] += 1.0
        n = float(np.linalg.norm(out[i]))
        if n > 0:
            out[i] /= n
    return out


def eval_thresh(
    cases: list[dict],
    update_thresh: float,
    noop_thresh: float,
    *,
    embed_fn,
    model: str,
) -> tuple[float, float]:
    ok_l = ok_s = 0
    for case in cases:
        with tempfile.TemporaryDirectory() as td:
            store = HierStore(Path(td) / "t.sqlite")
            for s in case["seeds"]:
                store.create(s)
            mgr = RuleV0(
                embed_fn=embed_fn,
                update_thresh=update_thresh,
                noop_thresh=noop_thresh,
                model=model,
            )
            logs = mgr.apply(
                ManagerInput(
                    text=case["incoming"]["text"],
                    project=case["incoming"]["project"],
                    fact_id=f"in_{case['id']}",
                ),
                store,
            )
            if ops_match_lenient(case["gold_ops"], logs):
                ok_l += 1
            if ops_match_strict(case["gold_ops"], logs):
                ok_s += 1
            store.close()
    n = len(cases) or 1
    return ok_l / n, ok_s / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "conflict_v1")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--real", action="store_true", help="tune with OpenAI embeds")
    ap.add_argument(
        "--max-dev",
        type=int,
        default=50,
        help="cap stratified dev sample (full 160 with --max-dev 0)",
    )
    args = ap.parse_args()
    out_path = args.out or (args.data / "rulev0_thresh.json")

    cases = [
        json.loads(l)
        for l in (args.data / "cases.jsonl").read_text().splitlines()
        if l.strip()
    ]
    dev = [c for c in cases if c.get("split") == "dev"]
    by: dict[str, list] = {}
    for c in dev:
        by.setdefault(c["type"], []).append(c)
    if args.max_dev and args.max_dev > 0:
        capped: list[dict] = []
        per = max(1, args.max_dev // max(1, len(by)))
        for rows in by.values():
            capped.extend(rows[:per])
        capped = capped[: args.max_dev]
    else:
        capped = list(dev)
    print(f"tuning on n_dev_sample={len(capped)} / {len(dev)} real={args.real}", flush=True)

    if args.real:
        embedder = Embedder(
            cache_path=ROOT / "runs" / "_embed_cache" / "conflict_v1_tune.json"
        )
        embed_fn = lambda texts: embedder.embed_texts(list(texts))
        model = "text-embedding-3-small"
        # real cosine space
        grid_u = [0.78, 0.82, 0.85, 0.88, 0.90]
        grid_n = [0.92, 0.95, 0.97]
    else:
        embed_fn = hash_embed
        model = "hash64"
        grid_u = [0.40, 0.45, 0.50, 0.55]
        grid_n = [0.95, 0.99]

    best = (-1.0, -1.0, grid_u[0], grid_n[-1])  # lenient, strict, u, n
    for u in grid_u:
        for n in grid_n:
            if n <= u:
                continue
            acc_l, acc_s = eval_thresh(
                capped, u, n, embed_fn=embed_fn, model=model
            )
            print(
                f"  u={u:.2f} n={n:.2f} lenient={acc_l:.3f} strict={acc_s:.3f}",
                flush=True,
            )
            if acc_l > best[0] or (acc_l == best[0] and acc_s > best[1]):
                best = (acc_l, acc_s, u, n)

    out = {
        "update_thresh": best[2],
        "noop_thresh": best[3],
        "tuned_on": f"conflict_v1:dev_sample_{len(capped)}",
        "dev_op_accuracy_lenient": best[0],
        "dev_op_accuracy_strict": best[1],
        "embed": model,
        "n_dev_full": len(dev),
        "n_dev_sample": len(capped),
        "note": "Thresholds from DEV only. Score TEST separately; do not re-tune on test.",
    }
    if args.real:
        out["embed_stats"] = embedder.stats()
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    # also keep a stamped copy for real runs
    if args.real:
        stamp = args.data / "rulev0_thresh_real.json"
        stamp.write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", stamp, flush=True)
    print("best", out, flush=True)
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
