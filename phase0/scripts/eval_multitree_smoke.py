#!/usr/bin/env python3
"""Eval multitree synth smoke: twin intrusion + path baselines (+ optional LLM).

Lead:
  - twin_intrusion flat vs subtree (ANN)
  - path baselines on holdout trees: random-dir / exact-oracle sanity
  - optional --llm gpt-4o on holdout place tasks with existing_dirs
"""
from __future__ import annotations

import argparse
import json
import random
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
from harness.store import HierStore, path_key  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def recall_at_k(gold_ids: list[str], hits) -> float:
    if not gold_ids:
        return 0.0
    got = {h.id for h in hits}
    return len(got & set(gold_ids)) / len(set(gold_ids))


def soft_score(gold: list[str], pred: list[str]) -> dict:
    exact = gold == pred
    same_root = bool(gold) and bool(pred) and gold[0] == pred[0]
    if exact:
        rel = "exact"
    elif pred and len(gold) <= len(pred) and pred[: len(gold)] == gold:
        rel = "gold_prefix"
    elif pred and len(pred) <= len(gold) and gold[: len(pred)] == pred:
        rel = "pred_prefix"
    elif same_root:
        rel = "same_root"
    else:
        rel = "diff_root" if pred else "empty"
    return {
        "exact": exact,
        "soft_hit": rel in ("exact", "gold_prefix", "pred_prefix"),
        "branch_ok": same_root,
        "soft_relation": rel,
    }


def eval_twins(data_root: Path, k: int, embed_model: str) -> dict:
    meta = json.loads((data_root / "META.json").read_text())
    embedder = Embedder(
        model=embed_model,
        cache_path=ROOT / "runs" / "_embed_cache" / "multitree_twins.json",
    )
    rows = []
    for tid in [t["tree_id"] for t in meta["trees"]]:
        tdir = data_root / tid
        store = HierStore(tdir / "hierstore.sqlite")
        facts = store.read_all(valid_only=True)
        queries = load_jsonl(tdir / "twin_queries.jsonl")
        fvecs = embedder.embed_texts([f["text"] for f in facts])
        qvecs = embedder.embed_texts([q["question"] for q in queries])
        index = MemoryIndex(facts, np.asarray(fvecs, dtype=np.float32))
        for i, q in enumerate(queries):
            qv = np.asarray(qvecs[i], dtype=np.float32)
            flat = index.retrieve_flat(qv, k=k)
            sub = index.retrieve_subtree(qv, k=k, active_path=q["active_path"])
            twin = q.get("twin_id")
            flat_ids = [h.id for h in flat]
            sub_ids = [h.id for h in sub]
            rows.append(
                {
                    "id": q["id"],
                    "tree_id": tid,
                    "kind": q.get("kind"),
                    "recall_flat": recall_at_k(q["gold_ids"], flat),
                    "recall_subtree": recall_at_k(q["gold_ids"], sub),
                    "twin_in_flat": bool(twin and twin in flat_ids),
                    "twin_in_subtree": bool(twin and twin in sub_ids),
                }
            )
        store.close()

    n = len(rows) or 1
    summary = {
        "n": len(rows),
        "k": k,
        "recall_flat": sum(r["recall_flat"] for r in rows) / n,
        "recall_subtree": sum(r["recall_subtree"] for r in rows) / n,
        "twin_intrusion_flat": sum(r["twin_in_flat"] for r in rows) / n,
        "twin_intrusion_subtree": sum(r["twin_in_subtree"] for r in rows) / n,
        "embed_stats": embedder.stats(),
    }
    return summary, rows


def eval_path_baselines(holdout_tasks: list[dict], seed: int = 7) -> dict:
    """Random among existing_dirs vs always-dump-root vs gold sanity."""
    rng = random.Random(seed)
    rows = []
    for t in holdout_tasks:
        gold = t["gold_path"]
        dirs = t.get("existing_dirs") or []
        # random existing dir
        pred_rand = list(rng.choice(dirs)) if dirs else []
        # always deepest dump-like = last root only if present
        roots = t.get("roots") or []
        pred_dump = [roots[-1]] if roots else []
        sc_r = soft_score(gold, pred_rand)
        sc_d = soft_score(gold, pred_dump)
        sc_g = soft_score(gold, gold)
        rows.append(
            {
                "id": t["id"],
                "tree_id": t.get("tree_id"),
                "gold_path": gold,
                "random": sc_r,
                "dump_root": sc_d,
                "oracle": sc_g,
            }
        )

    def agg(key: str, field: str) -> float:
        return sum(r[key][field] for r in rows) / (len(rows) or 1)

    return {
        "n": len(rows),
        "random_exact": agg("random", "exact"),
        "random_soft": agg("random", "soft_hit"),
        "random_branch": agg("random", "branch_ok"),
        "dump_root_exact": agg("dump_root", "exact"),
        "dump_root_branch": agg("dump_root", "branch_ok"),
        "oracle_exact": agg("oracle", "exact"),
    }


def eval_llm_holdout(
    holdout_tasks: list[dict],
    model: str,
    limit: int,
    out_dir: Path,
) -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "probe_llm_placer", ROOT / "scripts" / "probe_llm_placer.py"
    )
    probe = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(probe)

    from harness.manager import make_chat_client

    tasks = holdout_tasks[:limit] if limit else holdout_tasks
    client = make_chat_client("openai")
    rows = []
    pt = ct = 0
    for t in tasks:
        roots = t["roots"]
        pred = probe.call_placer(
            client,
            model=model,
            roots=roots,
            max_depth=5,
            text=t["text"],
            kind="place",
            cue=None,
            from_path=None,
            existing_dirs=t.get("existing_dirs") or [[r] for r in roots],
        )
        pt += pred["usage"]["prompt_tokens"]
        ct += pred["usage"]["completion_tokens"]
        sc = soft_score(t["gold_path"], pred["path"])
        rows.append(
            {
                "id": t["id"],
                "tree_id": t.get("tree_id"),
                "gold_path": t["gold_path"],
                "pred_path": pred["path"],
                **sc,
            }
        )
        mark = "OK" if sc["exact"] else ("SOFT" if sc["soft_hit"] else "MISS")
        print(
            f"{mark} {t['id']} gold={path_key(t['gold_path'])} "
            f"pred={path_key(pred['path'])}",
            flush=True,
        )

    n = len(rows) or 1
    summary = {
        "n": len(rows),
        "model": model,
        "path_exact": sum(r["exact"] for r in rows) / n,
        "path_soft": sum(r["soft_hit"] for r in rows) / n,
        "branch_ok": sum(r["branch_ok"] for r in rows) / n,
        "chat_prompt_tokens": pt,
        "chat_completion_tokens": ct,
        "chat_cost_usd_est": round((pt * 2.5 + ct * 10) / 1e6, 5),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "llm_holdout_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "multitree_synth_smoke",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "multitree_smoke")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--llm", action="store_true", help="gpt-4o on holdout place tasks")
    ap.add_argument("--llm-model", default="gpt-4o")
    ap.add_argument("--llm-limit", type=int, default=0, help="0=all holdout")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    holdout = load_jsonl(args.data / "place_holdout.jsonl")

    print("=== twins (subtree vs flat) ===", flush=True)
    twin_sum, twin_rows = eval_twins(args.data, args.k, args.embed_model)
    print(json.dumps(twin_sum, indent=2), flush=True)
    (args.out / "twin_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in twin_rows) + "\n"
    )

    print("=== path baselines (holdout trees) ===", flush=True)
    base = eval_path_baselines(holdout)
    print(json.dumps(base, indent=2), flush=True)

    llm_sum = None
    if args.llm:
        print("=== LLM placer holdout ===", flush=True)
        llm_sum = eval_llm_holdout(
            holdout, args.llm_model, args.llm_limit, args.out
        )
        print(json.dumps(llm_sum, indent=2), flush=True)

    report = {
        "data": str(args.data),
        "twins": twin_sum,
        "path_baselines_holdout": base,
        "llm_holdout": llm_sum,
        "note": "Lead: twin_intrusion gap + holdout path_soft/branch with dirs. User-dir transfer separate.",
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2))
    print("wrote", args.out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
