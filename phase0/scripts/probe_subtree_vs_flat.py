#!/usr/bin/env python3
"""Subtree vs Flat ANN retrieve on confusable hard-tree corpus.

Same embed model + cosine. Only candidate pool differs.
Lead: recall@k, twin_intrusion@k (flat pulled other project's twin).
No reader/QA — storage/retrieve signal only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from harness.embed import Embedder  # noqa: E402
from harness.index import MemoryIndex  # noqa: E402
from harness.store import HierStore  # noqa: E402


def load_queries(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def recall_at_k(gold_ids: list[str], hits) -> float:
    if not gold_ids:
        return 0.0
    got = {h.id for h in hits}
    return len(got & set(gold_ids)) / len(set(gold_ids))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "confusable_tree",
    )
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "subtree_vs_flat")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    db = args.data / "hierstore.sqlite"
    qpath = args.data / "queries.jsonl"
    store = HierStore(db)
    facts = store.read_all(valid_only=True)
    queries = load_queries(qpath)
    if args.smoke:
        queries = queries[: args.smoke]

    embedder = Embedder(
        model=args.embed_model,
        cache_path=ROOT / "runs" / "_embed_cache" / "confusable_tree.json",
    )
    ftexts = [f["text"] for f in facts]
    fvecs = embedder.embed_texts(ftexts)
    qvecs = embedder.embed_texts([q["question"] for q in queries])
    index = MemoryIndex(facts, np.asarray(fvecs, dtype=np.float32))

    rows = []
    for i, q in enumerate(queries):
        qv = np.asarray(qvecs[i], dtype=np.float32)
        flat = index.retrieve_flat(qv, k=args.k)
        sub = index.retrieve_subtree(qv, k=args.k, active_path=q["active_path"])
        gold = q["gold_ids"]
        twin = q.get("twin_id")
        flat_ids = [h.id for h in flat]
        sub_ids = [h.id for h in sub]
        row = {
            "id": q["id"],
            "active_path": q["active_path"],
            "gold_ids": gold,
            "twin_id": twin,
            "flat_ids": flat_ids,
            "subtree_ids": sub_ids,
            "recall_flat": recall_at_k(gold, flat),
            "recall_subtree": recall_at_k(gold, sub),
            "twin_in_flat": bool(twin and twin in flat_ids),
            "twin_in_subtree": bool(twin and twin in sub_ids),
        }
        rows.append(row)

    n = len(rows)
    summary = {
        "n": n,
        "k": args.k,
        "embed_model": args.embed_model,
        "n_facts": len(facts),
        "recall_flat": sum(r["recall_flat"] for r in rows) / n,
        "recall_subtree": sum(r["recall_subtree"] for r in rows) / n,
        "delta_recall_subtree_minus_flat": (
            sum(r["recall_subtree"] - r["recall_flat"] for r in rows) / n
        ),
        "twin_intrusion_flat": sum(r["twin_in_flat"] for r in rows) / n,
        "twin_intrusion_subtree": sum(r["twin_in_subtree"] for r in rows) / n,
        "embed_stats": embedder.stats(),
        "by_kind": {},
    }
    for kind in sorted({q.get("kind") or "?" for q in queries}):
        sub = [r for r, q in zip(rows, queries) if (q.get("kind") or "?") == kind]
        if not sub:
            continue
        m = len(sub)
        summary["by_kind"][kind] = {
            "n": m,
            "recall_flat": sum(r["recall_flat"] for r in sub) / m,
            "recall_subtree": sum(r["recall_subtree"] for r in sub) / m,
            "twin_intrusion_flat": sum(r["twin_in_flat"] for r in sub) / m,
            "twin_intrusion_subtree": sum(r["twin_in_subtree"] for r in sub) / m,
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    store.close()
    print(json.dumps(summary, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
