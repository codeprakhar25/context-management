#!/usr/bin/env python3
"""Phase-0 Flat vs Hier eval runner.

Usage:
  python3 -m harness.run --data data/hard_v1 --ranker embed --methods flat,hier
  python3 -m harness.run --store data/locomo/hierstore.sqlite \\
      --queries data/locomo/queries.jsonl --ranker embed --methods flat,hier
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    import os

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if not k:
            continue
        if not os.environ.get(k):
            os.environ[k] = v


_load_dotenv(ROOT / ".env")

from harness.bm25 import BM25Index
from harness.corpus import load_corpus
from harness.embed import Embedder
from harness.index import MemoryIndex
from harness.metrics import (
    aggregate,
    answer_in_hits,
    exact_match,
    retrieval_scores,
    token_f1,
)
from harness.judge import AnswerJudge
from harness.reader import Reader
from harness.store import HierStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-0 Flat vs Hier memory eval")
    p.add_argument("--data", type=Path, default=None, help="jsonl corpus dir (facts+queries)")
    p.add_argument("--store", type=Path, default=None, help="HierStore sqlite path")
    p.add_argument("--queries", type=Path, default=None, help="queries.jsonl (required with --store)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--methods", type=str, default="flat,hier")
    p.add_argument("--ranker", type=str, default="embed", choices=["bm25", "embed"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--k-global", type=int, default=1)
    p.add_argument("--embed-model", type=str, default="text-embedding-3-small")
    p.add_argument("--reader-model", type=str, default="gpt-4o-mini")
    p.add_argument("--smoke", type=int, default=0)
    p.add_argument("--project", type=str, default=None, help="filter queries to one project/scope")
    p.add_argument("--skip-reader", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "runs" / "_embed_cache")
    p.add_argument("--persist-embeddings", action="store_true", help="write embeds into sqlite store")
    p.add_argument("--reader-workers", type=int, default=8, help="parallel reader threads")
    p.add_argument(
        "--judge",
        action="store_true",
        help="LLM-judge as lead QA metric (EM/F1 stay diagnostic)",
    )
    p.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    return p.parse_args()


def retrieve(ranker, method: str, qtext: str, project, k: int, k_global: int, qvec=None):
    if method == "flat":
        if isinstance(ranker, BM25Index):
            return ranker.retrieve_flat(qtext, k=k)
        return ranker.retrieve_flat(qvec, k=k)
    if method == "hier":
        if isinstance(ranker, BM25Index):
            return ranker.retrieve_hier(qtext, k=k, project=project, k_global=k_global)
        return ranker.retrieve_hier(qvec, k=k, project=project, k_global=k_global)
    raise ValueError(method)


def load_facts_and_queries(args: argparse.Namespace):
    store = None
    if args.store:
        if not args.queries:
            raise SystemExit("--store requires --queries")
        store = HierStore(args.store)
        facts = store.read_all(valid_only=True)
        queries = []
        with args.queries.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    queries.append(json.loads(line))
        corpus_id = f"store:{args.store.name}:{len(facts)}:{len(queries)}"
        return facts, queries, corpus_id, store

    data = args.data or (ROOT / "data" / "toy")
    facts, queries, corpus_id = load_corpus(data)
    return facts, queries, corpus_id, None


def ensure_embed_matrix(
    facts: list[dict],
    embedder: Embedder,
    store: HierStore | None,
    model: str,
    persist: bool,
) -> np.ndarray:
    cached = store.load_embeddings_matrix(facts, model) if store else [None] * len(facts)
    missing_idx = [i for i, v in enumerate(cached) if v is None]
    if missing_idx:
        texts = [facts[i]["text"] for i in missing_idx]
        print(f"embedding {len(texts)} / {len(facts)} facts…")
        new_vecs = embedder.embed_texts(texts)
        for j, i in enumerate(missing_idx):
            cached[i] = new_vecs[j]
            if store is not None and persist:
                store.put_embedding(facts[i]["id"], model, new_vecs[j])
    # all filled
    return np.stack([np.asarray(v, dtype=np.float32) for v in cached], axis=0)


def run_method(
    method: str,
    ranker,
    embedder: Embedder | None,
    reader: Reader | None,
    queries: list[dict],
    k: int,
    k_global: int,
    ranker_name: str,
    qvecs: np.ndarray | None = None,
    reader_workers: int = 1,
    judge: AnswerJudge | None = None,
) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Phase 1: retrieve all (CPU / local)
    packs: list[tuple[dict, list]] = []
    for qi, q in enumerate(queries):
        qtext = q["text"]
        t0 = time.perf_counter()
        qvec = qvecs[qi] if qvecs is not None else None
        hits = retrieve(ranker, method, qtext, q.get("project"), k, k_global, qvec=qvec)
        retrieve_ms = (time.perf_counter() - t0) * 1000
        packs.append((q, hits, retrieve_ms))
        if (qi + 1) % 200 == 0:
            print(f"  {method} retrieve: {qi+1}/{len(queries)}", flush=True)

    # Phase 2: reader (API) — optional parallelism
    preds: list[str] = [""] * len(packs)
    if reader is not None:
        def _ans(i: int) -> tuple[int, str]:
            q, hits, _ = packs[i]
            return i, reader.answer(q["text"], hits)

        workers = max(1, reader_workers)
        if workers == 1:
            for i in range(len(packs)):
                _, pred = _ans(i)
                preds[i] = pred
                if (i + 1) % 50 == 0:
                    print(f"  {method} reader: {i+1}/{len(packs)}", flush=True)
        else:
            print(f"  {method} reader workers={workers}", flush=True)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_ans, i) for i in range(len(packs))]
                done = 0
                for fut in as_completed(futs):
                    i, pred = fut.result()
                    preds[i] = pred
                    done += 1
                    if done % 50 == 0:
                        print(f"  {method} reader: {done}/{len(packs)}", flush=True)

    rows: list[dict] = []
    for i, (q, hits, retrieve_ms) in enumerate(packs):
        gold_ids = q.get("gold_ids") or []
        rs = retrieval_scores(
            hit_ids=[h.id for h in hits],
            gold_ids=gold_ids,
            hit_projects=[h.project for h in hits],
            query_project=q.get("project"),
            query_type=q.get("type", "local"),
        )
        aih = answer_in_hits(q["gold_answer"], [h.text for h in hits])
        pred = preds[i]
        em = exact_match(pred, q["gold_answer"]) if pred else 0.0
        f1 = token_f1(pred, q["gold_answer"]) if pred else 0.0
        judge_score = None
        if judge is not None:
            judge_score = judge.grade(q["text"], q["gold_answer"], pred)
            if (i + 1) % 50 == 0:
                print(f"  {method} judge: {i+1}/{len(packs)}", flush=True)
        rows.append(
            {
                "query_id": q["id"],
                "type": q.get("type"),
                "project": q.get("project"),
                "method": method,
                "ranker": ranker_name,
                "hit_ids": [h.id for h in hits],
                "hit_scores": [round(h.score, 4) for h in hits],
                "gold_ids": gold_ids,
                "gold_answer": q["gold_answer"],
                "pred": pred,
                "em": em,
                "f1": f1,
                "judge": judge_score,
                "answer_in_hits": aih,
                "retrieve_ms": retrieve_ms,
                **rs,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ("flat", "hier"):
            raise SystemExit(f"unsupported method: {m}")

    facts, queries, corpus_id, store = load_facts_and_queries(args)
    if args.project:
        queries = [q for q in queries if q.get("project") == args.project]
    if args.smoke > 0:
        queries = queries[: args.smoke]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or (
        ROOT / "runs" / f"{stamp}_{args.ranker}_{'-'.join(methods)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"corpus_id={corpus_id} facts={len(facts)} queries={len(queries)} "
        f"ranker={args.ranker} store={args.store}"
    )

    embedder = None
    embed_after_ingest = None
    t_ing = time.perf_counter()
    if args.ranker == "bm25":
        ranker = BM25Index(facts)
        ingest_ms = (time.perf_counter() - t_ing) * 1000
        print(f"BM25 index built ingest_ms={ingest_ms:.1f}")
    else:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        embedder = Embedder(
            model=args.embed_model,
            cache_path=args.cache_dir / f"{Path(str(corpus_id)).name}_{args.embed_model}.json".replace(":", "_"),
        )
        persist = bool(args.persist_embeddings and store is not None)
        fact_vecs = ensure_embed_matrix(
            facts, embedder, store, args.embed_model, persist=persist
        )
        ingest_ms = (time.perf_counter() - t_ing) * 1000
        ranker = MemoryIndex(facts, fact_vecs)
        embed_after_ingest = dict(embedder.stats())
        print(f"ingest_ms={ingest_ms:.1f} embed_stats={embed_after_ingest}")

    reader = None if args.skip_reader else Reader(model=args.reader_model)
    judge = AnswerJudge(model=args.judge_model) if args.judge else None

    qvecs = None
    if args.ranker == "embed":
        assert embedder is not None
        print(f"embedding {len(queries)} queries…", flush=True)
        qvecs = embedder.embed_texts([q["text"] for q in queries])

    all_preds: dict[str, list] = {}
    summaries: dict[str, dict] = {}

    for method in methods:
        label = f"{args.ranker}:{method}"
        print(f"=== {label} ===", flush=True)
        rows = run_method(
            method,
            ranker,
            embedder,
            reader,
            queries,
            args.k,
            args.k_global,
            args.ranker,
            qvecs=qvecs,
            reader_workers=args.reader_workers,
            judge=judge,
        )
        all_preds[method] = rows
        summaries[method] = aggregate(rows)
        print(json.dumps(summaries[method], indent=2), flush=True)

    primary = ["judge", "recall_at_k", "answer_in_hits", "cost"] if args.judge else [
        "em",
        "f1",
        "answer_in_hits",
        "recall_at_k",
        "cost",
    ]
    config = {
        "corpus_id": corpus_id,
        "store": str(args.store) if args.store else None,
        "data": str(args.data) if args.data else None,
        "queries": str(args.queries) if args.queries else None,
        "n_facts": len(facts),
        "n_queries": len(queries),
        "methods": methods,
        "ranker": args.ranker,
        "k": args.k,
        "k_global": args.k_global,
        "embed_model": args.embed_model if args.ranker == "embed" else None,
        "reader_model": None if args.skip_reader else args.reader_model,
        "judge": bool(args.judge),
        "judge_model": args.judge_model if args.judge else None,
        "temperature": 0.0,
        "smoke": args.smoke,
        "project_filter": args.project,
        "conflict_policy": store.conflict_policy() if store else None,
        "ingest_ms": ingest_ms,
        "created_utc": stamp,
        "primary_metrics": primary,
        "em_f1": "diagnostic_only",
        "precision_in_scope": "diagnostic_only",
    }

    cost = {
        "embed_ingest": embed_after_ingest,
        "embed_total": embedder.stats() if embedder else None,
        "reader": reader.stats() if reader else None,
        "judge": judge.stats() if judge else None,
        "total_cost_usd_est": round(
            (embedder.cost_usd() if embedder else 0.0)
            + (reader.cost_usd() if reader else 0.0)
            + (
                judge.in_tokens / 1e6 * 0.15 + judge.out_tokens / 1e6 * 0.60
                if judge
                else 0.0
            ),
            6,
        ),
    }

    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "metrics.json").write_text(
        json.dumps({"summary": summaries, "cost": cost}, indent=2)
    )
    with (out_dir / "predictions.jsonl").open("w") as f:
        for method, rows in all_preds.items():
            for r in rows:
                f.write(json.dumps(r) + "\n")

    if "flat" in summaries and "hier" in summaries:
        delta = {
            "hier_minus_flat_judge": summaries["hier"].get("mean_judge", 0.0)
            - summaries["flat"].get("mean_judge", 0.0),
            "hier_minus_flat_em": summaries["hier"]["mean_em"] - summaries["flat"]["mean_em"],
            "hier_minus_flat_f1": summaries["hier"]["mean_f1"] - summaries["flat"]["mean_f1"],
            "hier_minus_flat_answer_in_hits": summaries["hier"]["mean_answer_in_hits"]
            - summaries["flat"]["mean_answer_in_hits"],
            "hier_minus_flat_recall_at_k": summaries["hier"]["mean_recall_at_k"]
            - summaries["flat"]["mean_recall_at_k"],
            "hier_minus_flat_precision_in_scope_diagnostic": summaries["hier"][
                "mean_precision_in_scope"
            ]
            - summaries["flat"]["mean_precision_in_scope"],
        }
        (out_dir / "delta.json").write_text(json.dumps(delta, indent=2))
        print("delta hier-flat:", json.dumps(delta, indent=2))

    if store is not None:
        store.close()

    print(f"wrote {out_dir}")
    print("cost:", json.dumps(cost, indent=2))


if __name__ == "__main__":
    main()
