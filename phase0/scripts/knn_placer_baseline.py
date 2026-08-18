#!/usr/bin/env python3
"""Nearest-neighbour placer: put the note where the most similar known note lives.

The question this answers: does the LoRA earn its training cost? It beat gpt-4o
by 25 points on folders it had examples for, which is exactly the regime where
plain retrieval over those same examples should also work. If embedding kNN
matches it, the fine-tune bought nothing that a cached embedding index doesn't.

No training, no GPU. On a folder-disjoint split this scores 0 by construction —
the gold folder holds no training file to retrieve — which is itself the result.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from harness.embed import Embedder  # noqa: E402


def soft_score(gold: list[str], pred: list[str]) -> dict:
    exact = gold == pred
    same_root = bool(gold) and bool(pred) and gold[0] == pred[0]
    if exact:
        rel = "exact"
    elif len(gold) <= len(pred) and pred[: len(gold)] == gold:
        rel = "gold_prefix"
    elif len(pred) <= len(gold) and gold[: len(pred)] == pred:
        rel = "pred_prefix"
    elif same_root:
        rel = "same_root"
    else:
        rel = "diff_root"
    return {
        "exact": exact,
        "soft_hit": rel in ("exact", "gold_prefix", "pred_prefix"),
        "branch_ok": same_root,
        "soft_relation": rel,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "runs" / "_embed_cache" / "knn_placer.json",
        help="embed cache; point at candidate_recall.json to reuse the gate",
    )
    args = ap.parse_args()

    load = lambda p: [json.loads(l) for l in p.read_text().splitlines() if l.strip()]  # noqa: E731
    train, val = load(args.train), load(args.val)

    emb = Embedder(cache_path=args.cache)
    tv = np.array(emb.embed_texts([t["text"] for t in train]), dtype=np.float32)
    vv = np.array(emb.embed_texts([v["text"] for v in val]), dtype=np.float32)
    tv /= np.linalg.norm(tv, axis=1, keepdims=True) + 1e-9
    vv /= np.linalg.norm(vv, axis=1, keepdims=True) + 1e-9
    sim = vv @ tv.T
    order = np.argsort(-sim, axis=1)

    key = lambda p: "/" + "/".join(p)  # noqa: E731
    summary = {"train": str(args.train), "val": str(args.val), "n": len(val), "by_k": {}}
    args.out.mkdir(parents=True, exist_ok=True)

    for k in args.k:
        rows = []
        for i, v in enumerate(val):
            nb = order[i][:k]
            # weight each neighbour's folder by similarity; ties break on the closest
            votes: dict[str, float] = collections.defaultdict(float)
            for j in nb:
                votes[key(train[j]["gold_path"])] += float(sim[i][j])
            best = max(votes.items(), key=lambda x: x[1])[0]
            pred = [p for p in best.split("/") if p]
            rows.append(
                {
                    "id": v["id"],
                    "gold_path": v["gold_path"],
                    "pred_path": pred,
                    "top_sim": float(sim[i][nb[0]]),
                    **soft_score(v["gold_path"], pred),
                }
            )
        n = len(rows)
        summary["by_k"][str(k)] = {
            "path_exact": sum(r["exact"] for r in rows) / n,
            "path_soft": sum(r["soft_hit"] for r in rows) / n,
            "branch_ok": sum(r["branch_ok"] for r in rows) / n,
        }
        if k == args.k[0]:
            (args.out / "holdout_results.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n"
            )
    summary["embed"] = emb.stats()
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
