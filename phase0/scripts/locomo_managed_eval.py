#!/usr/bin/env python3
"""Score a manager-built LoCoMo store: QA accuracy AND false invalidation.

The point of measuring both on the same store: Memory-R1 trains its Memory
Manager on R = EM(y_pred, y_gold) (eq. 5) with no per-operation term. If a
manager posts strong QA numbers while `fi_sound` is high, then that reward
cannot see the destruction it is implicitly rewarding.

Retrieval is scoped to the query's own conversation and defaults to k=60 to
match Memory-R1's Answer Agent candidate set. Reader and judge are frozen
across arms so the only thing that varies is the memory bank.

Inputs:  runs/locomo_managed/<arm>/{store.sqlite,ops.jsonl}
Outputs: runs/locomo_managed/<arm>/eval.json  + per-query rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
from harness.index import MemoryIndex  # noqa: E402
from harness.judge import AnswerJudge  # noqa: E402
from harness.metrics import aggregate, exact_match, retrieval_scores, token_f1  # noqa: E402
from harness.reader import Reader  # noqa: E402
from harness.store import HierStore  # noqa: E402
from harness.write_metrics import fi_metrics  # noqa: E402

DATA = ROOT / "data" / "locomo"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", type=Path, required=True)
    ap.add_argument("--queries", type=Path, default=DATA / "queries.jsonl")
    ap.add_argument("--k", type=int, default=60, help="Memory-R1 Answer Agent uses 60")
    ap.add_argument("--reader-model", default="gpt-4o-mini")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--limit-queries", type=int, default=None, help="smoke test")
    ap.add_argument("--workers", type=int, default=8, help="parallel query scoring; results are order-stable")
    ap.add_argument("--no-qa", action="store_true", help="FI only, zero API spend")
    args = ap.parse_args()

    ops_rows = [
        json.loads(l) for l in (args.arm_dir / "ops.jsonl").read_text().splitlines() if l.strip()
    ]
    queries = [
        json.loads(l) for l in args.queries.read_text().splitlines() if l.strip()
    ]
    store = HierStore(args.arm_dir / "store.sqlite")
    facts = store.read_all(valid_only=True)
    final_valid_ids = {f["id"] for f in facts}

    # Restrict to conversations this arm actually ingested (smoke runs).
    convs = {r["conv"] for r in ops_rows}
    queries = [q for q in queries if q["project"] in convs]

    # ---- false invalidation (no API, no new labels) -----------------------
    required_ids = {g for q in queries for g in (q.get("gold_ids") or [])}
    fi = fi_metrics(required_ids, final_valid_ids, ops_rows)
    report: dict = {
        "arm_dir": str(args.arm_dir),
        "n_conversations": len(convs),
        "n_queries_scored": None,
        "final_facts_valid": len(facts),
        "fi": fi,
    }
    print(json.dumps(fi, indent=2), flush=True)

    if args.no_qa:
        (args.arm_dir / "eval.json").write_text(json.dumps(report, indent=2) + "\n")
        print("wrote", args.arm_dir / "eval.json")
        return

    # ---- QA on the same store --------------------------------------------
    qs = [q for q in queries if q.get("gold_answer")]
    if args.limit_queries:
        qs = qs[: args.limit_queries]

    embedder = Embedder(cache_path=ROOT / "runs" / "_embed_cache" / "locomo_managed_eval.json")
    fvecs = embedder.embed_texts([f["text"] for f in facts])
    index = MemoryIndex(facts, fvecs)
    qvecs = embedder.embed_texts([q["text"] for q in qs])

    reader = Reader(model=args.reader_model)
    judge = AnswerJudge(model=args.judge_model)
    t0 = time.perf_counter()

    # Per-query work is independent and almost entirely network wait, so it is
    # thread-pooled. Results are written back by index, never appended, so row
    # order stays deterministic regardless of completion order — the run is
    # reproducible at any --workers setting. temperature=0 throughout.
    rows: list[dict] = [None] * len(qs)  # type: ignore[list-item]
    done = 0
    lock = threading.Lock()

    def score_one(i: int) -> None:
        nonlocal done
        q = qs[i]
        hits = index.retrieve_hier(qvecs[i], k=args.k, project=q["project"], k_global=0)
        pred = reader.answer(q["text"], hits)
        grade = judge.grade(q["text"], q["gold_answer"], pred)
        rs = retrieval_scores(
            [h.id for h in hits],
            q.get("gold_ids") or [],
            [h.project for h in hits],
            q["project"],
            q.get("type", "?"),
        )
        rows[i] = {
            "qid": q["id"],
            "type": q.get("type", "?"),
            "conv": q["project"],
            "n_gold": len(q.get("gold_ids") or []),
            "gold_present": sum(
                1 for g in (q.get("gold_ids") or []) if g in final_valid_ids
            ),
            "em": exact_match(pred, q["gold_answer"]),
            "f1": token_f1(pred, q["gold_answer"]),
            "judge": grade,
            "pred": pred,
            **rs,
        }
        with lock:
            done += 1
            if done % 200 == 0:
                print(f"  q {done}/{len(qs)}  {time.perf_counter()-t0:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(score_one, range(len(qs))))
    rows = [r for r in rows if r is not None]

    agg = aggregate(rows)
    # Does QA actually notice the destruction? Split on whether this question's
    # evidence survived the manager.
    intact = [r for r in rows if r["n_gold"] and r["gold_present"] == r["n_gold"]]
    damaged = [r for r in rows if r["n_gold"] and r["gold_present"] < r["n_gold"]]

    def _m(rs: list[dict], k: str) -> float | None:
        v = [r[k] for r in rs if r.get(k) is not None]
        return round(sum(v) / len(v), 4) if v else None

    report["n_queries_scored"] = len(rows)
    report["qa"] = agg
    report["qa_by_evidence_survival"] = {
        "evidence_intact": {"n": len(intact), "em": _m(intact, "em"), "judge": _m(intact, "judge")},
        "evidence_damaged": {"n": len(damaged), "em": _m(damaged, "em"), "judge": _m(damaged, "judge")},
    }
    report["cost"] = {**reader.stats(), **judge.stats(), **embedder.stats()}
    (args.arm_dir / "eval_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (args.arm_dir / "eval.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "fi": fi,
                "mean_em": agg.get("mean_em"),
                "mean_judge": agg.get("mean_judge"),
                "mean_recall_at_k": agg.get("mean_recall_at_k"),
                "by_evidence_survival": report["qa_by_evidence_survival"],
                "cost": report["cost"],
            },
            indent=2,
        )
    )
    store.close()


if __name__ == "__main__":
    main()
