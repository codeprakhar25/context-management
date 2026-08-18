#!/usr/bin/env python3
"""Retrieve bakeoff: flat vs scoped subtree vs path-graph hybrid.

Thesis-facing retrieve-only compare on multitree twin queries.
Reports recall@k, twin_intrusion@k, and retrieve-only latency (embeds excluded).

  python3 scripts/eval_retrieve_modes.py \\
    --data data/multitree_synth_mid_hard --k 5 10 --out runs/retrieve_modes_mid_hard
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from harness.embed import Embedder  # noqa: E402
from harness.index import Hit, MemoryIndex, _l2_normalize  # noqa: E402
from harness.store import HierStore, path_key  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def recall_at_k(gold_ids: list[str], hits: list[Hit]) -> float:
    if not gold_ids:
        return 0.0
    got = {h.id for h in hits}
    return len(got & set(gold_ids)) / len(set(gold_ids))


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    i = (len(s) - 1) * p / 100.0
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    w = i - lo
    return s[lo] * (1 - w) + s[hi] * w


def build_path_graph(facts: list[dict]) -> dict[str, set[str]]:
    """Undirected edges: sibling (same parent) + parent/child (one level)."""
    by_path: dict[str, list[str]] = defaultdict(list)
    id_to_path: dict[str, list[str]] = {}
    for f in facts:
        fid = f["id"]
        p = list(f.get("path") or [])
        id_to_path[fid] = p
        by_path[path_key(p)].append(fid)

    graph: dict[str, set[str]] = {f["id"]: set() for f in facts}

    # siblings: same exact path
    for ids in by_path.values():
        for a in ids:
            for b in ids:
                if a != b:
                    graph[a].add(b)

    # parent/child: path adjacency one level
    path_to_ids = by_path
    for fid, p in id_to_path.items():
        if len(p) >= 1:
            parent = path_key(p[:-1]) if len(p) > 1 else ""
            # child → facts at parent path
            if parent in path_to_ids:
                for pid in path_to_ids[parent]:
                    if pid != fid:
                        graph[fid].add(pid)
                        graph[pid].add(fid)
            # parent → facts at child paths (one level deeper with this prefix)
            # covered when we process children linking up

    return graph


def retrieve_path_graph_hybrid(
    index: MemoryIndex,
    graph: dict[str, set[str]],
    qvec: np.ndarray,
    k: int,
    k_seed: int,
) -> tuple[list[Hit], int]:
    """Flat seed → 1-hop path-graph expand → cosine rescore → top-k."""
    seeds = index.retrieve_flat(qvec, k=k_seed)
    cand: set[str] = set()
    for h in seeds:
        cand.add(h.id)
        cand |= graph.get(h.id, set())

    q = _l2_normalize(np.asarray(qvec, dtype=np.float32).reshape(1, -1))[0]
    scored: list[Hit] = []
    for fid in cand:
        i = index.id_to_i.get(fid)
        if i is None:
            continue
        f = index.facts[i]
        score = float(index.vectors[i] @ q)
        scored.append(
            Hit(
                id=f["id"],
                text=f["text"],
                score=score,
                project=f.get("project"),
                path=list(f.get("path") or []),
            )
        )
    scored.sort(key=lambda h: -h.score)
    return scored[:k], len(cand)


def latency_stats(ms: list[float]) -> dict:
    return {
        "n": len(ms),
        "mean_ms": round(statistics.fmean(ms), 4) if ms else 0.0,
        "p50_ms": round(pct(ms, 50), 4),
        "p95_ms": round(pct(ms, 95), 4),
        "sum_ms": round(sum(ms), 4),
    }


def eval_modes(
    data_root: Path,
    ks: list[int],
    k_seed: int,
    embed_model: str,
) -> tuple[dict, list[dict]]:
    meta = json.loads((data_root / "META.json").read_text())
    embedder = Embedder(
        model=embed_model,
        cache_path=ROOT / "runs" / "_embed_cache" / "retrieve_modes_twins.json",
    )

    # Preload all trees: facts, queries, index, graph, qvecs
    trees: list[dict] = []
    for tmeta in meta["trees"]:
        tid = tmeta["tree_id"]
        tdir = data_root / tid
        store = HierStore(tdir / "hierstore.sqlite")
        facts = store.read_all(valid_only=True)
        store.close()
        queries = load_jsonl(tdir / "twin_queries.jsonl")
        if not facts or not queries:
            continue
        fvecs = embedder.embed_texts([f["text"] for f in facts])
        qvecs = embedder.embed_texts([q["question"] for q in queries])
        index = MemoryIndex(facts, np.asarray(fvecs, dtype=np.float32))
        graph = build_path_graph(facts)
        trees.append(
            {
                "tree_id": tid,
                "index": index,
                "graph": graph,
                "queries": queries,
                "qvecs": np.asarray(qvecs, dtype=np.float32),
            }
        )

    modes = ("flat", "scoped", "path_graph_hybrid")
    results: list[dict] = []
    # per (k, mode) latency lists after warmup
    lat: dict[tuple[int, str], list[float]] = defaultdict(list)
    cand_sizes: dict[tuple[int, str], list[int]] = defaultdict(list)

    # Warmup: one query per tree per mode per k (discarded)
    for tree in trees:
        if not tree["queries"]:
            continue
        q0 = tree["queries"][0]
        qv0 = tree["qvecs"][0]
        idx: MemoryIndex = tree["index"]
        g = tree["graph"]
        for k in ks:
            _ = idx.retrieve_flat(qv0, k=k)
            _ = idx.retrieve_subtree(qv0, k=k, active_path=q0["active_path"])
            _ = retrieve_path_graph_hybrid(idx, g, qv0, k=k, k_seed=k_seed)

    for tree in trees:
        tid = tree["tree_id"]
        idx = tree["index"]
        g = tree["graph"]
        for i, q in enumerate(tree["queries"]):
            qv = tree["qvecs"][i]
            twin = q.get("twin_id")
            gold = q["gold_ids"]
            row: dict = {
                "id": q["id"],
                "tree_id": tid,
                "kind": q.get("kind"),
                "active_path": q.get("active_path"),
            }
            for k in ks:
                # flat
                t0 = time.perf_counter()
                hits_f = idx.retrieve_flat(qv, k=k)
                ms_f = (time.perf_counter() - t0) * 1000.0
                lat[(k, "flat")].append(ms_f)

                # scoped
                t0 = time.perf_counter()
                hits_s = idx.retrieve_subtree(qv, k=k, active_path=q["active_path"])
                ms_s = (time.perf_counter() - t0) * 1000.0
                lat[(k, "scoped")].append(ms_s)

                # hybrid
                t0 = time.perf_counter()
                hits_h, n_cand = retrieve_path_graph_hybrid(
                    idx, g, qv, k=k, k_seed=k_seed
                )
                ms_h = (time.perf_counter() - t0) * 1000.0
                lat[(k, "path_graph_hybrid")].append(ms_h)
                cand_sizes[(k, "path_graph_hybrid")].append(n_cand)

                for mode, hits, ms in (
                    ("flat", hits_f, ms_f),
                    ("scoped", hits_s, ms_s),
                    ("path_graph_hybrid", hits_h, ms_h),
                ):
                    hit_ids = [h.id for h in hits]
                    row[f"recall@{k}_{mode}"] = recall_at_k(gold, hits)
                    row[f"twin_in@{k}_{mode}"] = bool(twin and twin in hit_ids)
                    row[f"retrieve_ms@{k}_{mode}"] = round(ms, 4)
                row[f"hybrid_cand_n@{k}"] = n_cand
            results.append(row)

    by_k: dict[str, dict] = {}
    for k in ks:
        n = len(results) or 1
        mode_block = {}
        for mode in modes:
            rec = sum(r[f"recall@{k}_{mode}"] for r in results) / n
            intr = sum(1.0 if r[f"twin_in@{k}_{mode}"] else 0.0 for r in results) / n
            block = {
                "recall": round(rec, 4),
                "twin_intrusion": round(intr, 4),
                "latency": latency_stats(lat[(k, mode)]),
            }
            if mode == "path_graph_hybrid":
                cs = cand_sizes[(k, mode)]
                block["hybrid_cand"] = {
                    "mean": round(statistics.fmean(cs), 2) if cs else 0.0,
                    "p50": round(pct([float(x) for x in cs], 50), 2),
                    "k_seed": k_seed,
                }
            mode_block[mode] = block
        by_k[str(k)] = mode_block

    summary = {
        "data": str(data_root),
        "n_queries": len(results),
        "n_trees": len(trees),
        "ks": ks,
        "k_seed": k_seed,
        "modes": list(modes),
        "active_path": "oracle from twin query (router not evaluated)",
        "by_k": by_k,
        "embed_stats": embedder.stats(),
        "note": (
            "Retrieve-only wall time; embeds precomputed/cached. "
            "path_graph_hybrid = flat seed + 1-hop sibling/parent-child + rescore."
        ),
    }
    return summary, results


def print_table(summary: dict) -> None:
    print()
    print(f"n_queries={summary['n_queries']}  trees={summary['n_trees']}  "
          f"k_seed={summary['k_seed']}")
    print(f"{'k':>4}  {'mode':<20}  {'recall':>7}  {'intrusion':>10}  "
          f"{'p50_ms':>8}  {'p95_ms':>8}  {'mean_ms':>8}")
    print("-" * 78)
    for k in summary["ks"]:
        block = summary["by_k"][str(k)]
        for mode in summary["modes"]:
            m = block[mode]
            lat = m["latency"]
            print(
                f"{k:>4}  {mode:<20}  {m['recall']:7.3f}  {m['twin_intrusion']:10.3f}  "
                f"{lat['p50_ms']:8.3f}  {lat['p95_ms']:8.3f}  {lat['mean_ms']:8.3f}"
            )
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "multitree_synth_mid_hard",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "retrieve_modes_mid_hard")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--k-seed", type=int, default=20)
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"data={args.data} ks={args.k} k_seed={args.k_seed}", flush=True)
    summary, rows = eval_modes(args.data, args.k, args.k_seed, args.embed_model)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    print_table(summary)
    print("wrote", args.out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
