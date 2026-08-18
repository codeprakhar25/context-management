#!/usr/bin/env python3
"""Run conflict_v0 with real LLMv0 (OpenAI) vs AlwaysADD / RuleV0.

Requires OPENAI_API_KEY (loads phase0/.env if present).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# load .env
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from harness.embed import Embedder  # noqa: E402
from harness.manager import AlwaysADD, LLMv0, ManagerInput, RuleV0  # noqa: E402
from harness.metrics import answer_in_hits, exact_match, token_f1  # noqa: E402
from harness.store import HierStore  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_one(name: str, cases: list[dict], out_dir: Path, embedder: Embedder) -> dict:
    results = []
    op_ok = 0
    bank_sizes = []

    if name == "AlwaysADD":
        mgr: AlwaysADD | RuleV0 | LLMv0 = AlwaysADD()
    elif name == "RuleV0":
        mgr = RuleV0(embedder=embedder, top_m=5)
    elif name == "LLMv0":
        mgr = LLMv0(embedder=embedder, top_m=8)
    else:
        raise ValueError(name)

    for case in cases:
        with tempfile.TemporaryDirectory() as td:
            store = HierStore(Path(td) / "t.sqlite")
            for s in case["seeds"]:
                store.create(s)

            inp = ManagerInput(
                text=case["incoming"]["text"],
                project=case["incoming"]["project"],
                fact_id=f"in_{case['id']}",
            )
            dec = mgr.decide(inp, store)
            logs = store.apply_ops(dec.ops, manager=mgr.name)

            pred_ops = [l["op"] for l in logs]
            gold = case["gold_op"]
            gold_tid = case.get("gold_target_id")

            def op_matches(log_row: dict) -> bool:
                if log_row["op"] != gold:
                    return False
                if gold_tid and gold in ("UPDATE", "DELETE", "NOOP"):
                    return log_row.get("fact_id") == gold_tid
                return True

            correct = any(op_matches(l) for l in logs)
            pred_op = next((l["op"] for l in logs if op_matches(l)), pred_ops[0] if pred_ops else "NOOP")
            if correct:
                op_ok += 1

            valid = store.read_all(valid_only=True)
            qa = case.get("qa") or {}
            pack = [
                f["text"]
                for f in valid
                if f.get("project") == case["incoming"]["project"]
            ]
            gold_ans = qa.get("gold_answer", "")
            aih = answer_in_hits(gold_ans, pack) if gold_ans else 0.0
            pred = gold_ans if aih else (pack[0] if pack else "")
            row = {
                "case_id": case["id"],
                "type": case["type"],
                "manager": name,
                "gold_op": gold,
                "pred_ops": pred_ops,
                "pred_op": pred_op,
                "op_correct": correct,
                "n_valid": len(valid),
                "answer_in_hits": aih,
                "em": exact_match(pred, gold_ans) if gold_ans else 0.0,
                "f1": token_f1(pred, gold_ans) if gold_ans else 0.0,
                "raw_llm": dec.raw_llm,
                "ops_log": logs,
            }
            results.append(row)
            bank_sizes.append(len(valid))
            store.close()

    n = len(cases)
    by_type: dict[str, list] = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r["op_correct"])
    summary = {
        "manager": name,
        "n": n,
        "op_accuracy": op_ok / n if n else 0.0,
        "mean_n_valid": float(np.mean(bank_sizes)) if bank_sizes else 0.0,
        "mean_answer_in_hits": float(np.mean([r["answer_in_hits"] for r in results])),
        "by_type_op_acc": {t: sum(xs) / len(xs) for t, xs in by_type.items()},
        "pred_op_hist": dict(Counter(r["pred_op"] for r in results)),
        "manager_stats": mgr.stats() if isinstance(mgr, LLMv0) else None,
        "embed_stats": embedder.stats(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"results_{name}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + "\n"
    )
    (out_dir / f"summary_{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "conflict_v0")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "conflict_v0_llm")
    ap.add_argument("--managers", default="AlwaysADD,RuleV0,LLMv0")
    args = ap.parse_args()

    if not args.data.joinpath("cases.jsonl").exists():
        from scripts.build_conflict_v0 import main as build

        build()

    cases = load_jsonl(args.data / "cases.jsonl")
    embedder = Embedder(
        cache_path=ROOT / "runs" / "_embed_cache" / "conflict_v0_manager.json"
    )
    summaries = {}
    for name in [m.strip() for m in args.managers.split(",") if m.strip()]:
        print("===", name, flush=True)
        summaries[name] = run_one(name, cases, args.out, embedder)
        print(json.dumps(summaries[name], indent=2), flush=True)

    (args.out / "summaries.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
