#!/usr/bin/env python3
"""Probe AlwaysADD vs RuleV0 on conflict_v0: op accuracy + bank stats + string QA proxy.

Uses a local hash embedder (no API) so probes are cheap/reproducible.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.manager import AlwaysADD, ManagerInput, RuleV0  # noqa: E402
from harness.metrics import answer_in_hits, exact_match, token_f1  # noqa: E402
from harness.store import HierStore  # noqa: E402


def hash_embed(texts: list[str], dim: int = 64) -> np.ndarray:
    """Deterministic bag-of-token hash embedding for offline probes."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        toks = re.findall(r"[a-z0-9]+", text.lower())
        for t in toks:
            out[i, hash(t) % dim] += 1.0
        n = float(np.linalg.norm(out[i]))
        if n > 0:
            out[i] /= n
    return out


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_manager(name: str, cases: list[dict], out_dir: Path) -> dict:
    results = []
    op_ok = 0
    bank_stats = []

    for case in cases:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "t.sqlite"
            store = HierStore(db)
            for s in case["seeds"]:
                store.create(s)
                # pre-store hash embeds for RuleV0
                store.put_embedding(
                    s["id"], "hash64", hash_embed([s["text"]])[0].tolist()
                )

            if name == "AlwaysADD":
                mgr = AlwaysADD()
            else:
                # load embeds into manager + embed_fn for new text
                emb = {}
                for s in case["seeds"]:
                    emb[s["id"]] = hash_embed([s["text"]])[0]
                mgr = RuleV0(
                    embeddings=emb,
                    embed_fn=hash_embed,
                    top_m=5,
                    # hash bag-of-tokens: real embed thresh ~0.88; hash needs ~0.45
                    update_thresh=0.45,
                    noop_thresh=0.99,
                    model="hash64",
                )

            inp = ManagerInput(
                text=case["incoming"]["text"],
                project=case["incoming"]["project"],
                fact_id=f"in_{case['id']}",
            )
            logs = mgr.apply(inp, store)
            pred_ops = [l["op"] for l in logs]
            pred_op = pred_ops[0] if pred_ops else "NOOP"
            gold = case["gold_op"]
            target_ok = True
            if case.get("gold_target_id") and pred_op in ("UPDATE", "DELETE", "NOOP"):
                target_ok = any(
                    l.get("fact_id") == case["gold_target_id"] for l in logs
                )
            correct = pred_op == gold and target_ok
            if correct:
                op_ok += 1

            valid = store.read_all(valid_only=True)
            # wrong-project pollution: conv-demo facts that updated slm-lab seed
            other = store.get("seed_other_proj")
            polluted = False
            if other and case["id"] == "wrong_project_trap":
                polluted = "0.72" in other["text"]

            qa = case.get("qa") or {}
            # naive QA: concat valid in-project texts, score answer_in_hits / EM on concat as "pred"
            pack = [
                f["text"]
                for f in valid
                if f.get("project") == case["incoming"]["project"]
            ]
            gold_ans = qa.get("gold_answer", "")
            aih = answer_in_hits(gold_ans, pack) if gold_ans else 0.0
            # extractive fake pred: if gold in pack use gold else first sentence
            pred = gold_ans if aih else (pack[0] if pack else "")
            em = exact_match(pred, gold_ans) if gold_ans else 0.0
            f1 = token_f1(pred, gold_ans) if gold_ans else 0.0

            row = {
                "case_id": case["id"],
                "type": case["type"],
                "manager": name,
                "gold_op": gold,
                "pred_ops": pred_ops,
                "pred_op": pred_op,
                "op_correct": correct,
                "n_valid": len(valid),
                "polluted_other_project": polluted,
                "answer_in_hits": aih,
                "em": em,
                "f1": f1,
                "ops_log": logs,
            }
            results.append(row)
            bank_stats.append(len(valid))
            store.close()

    n = len(cases)
    by_type: dict[str, list] = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r["op_correct"])

    summary = {
        "manager": name,
        "n": n,
        "op_accuracy": op_ok / n if n else 0.0,
        "mean_n_valid": float(np.mean(bank_stats)) if bank_stats else 0.0,
        "mean_answer_in_hits": float(np.mean([r["answer_in_hits"] for r in results])),
        "mean_em": float(np.mean([r["em"] for r in results])),
        "mean_f1": float(np.mean([r["f1"] for r in results])),
        "pollution_count": sum(1 for r in results if r["polluted_other_project"]),
        "by_type_op_acc": {
            t: sum(xs) / len(xs) for t, xs in by_type.items()
        },
        "pred_op_hist": dict(Counter(r["pred_op"] for r in results)),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"results_{name}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + "\n"
    )
    (out_dir / f"summary_{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "conflict_v0",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runs" / "conflict_v0_probe",
    )
    ap.add_argument(
        "--managers",
        default="AlwaysADD,RuleV0",
    )
    args = ap.parse_args()

    cases_path = args.data / "cases.jsonl"
    if not cases_path.exists():
        from scripts.build_conflict_v0 import main as build

        build()
    cases = load_jsonl(cases_path)
    summaries = {}
    for name in [m.strip() for m in args.managers.split(",") if m.strip()]:
        summaries[name] = run_manager(name, cases, args.out)
        print(json.dumps(summaries[name], indent=2))

    delta = {}
    if "AlwaysADD" in summaries and "RuleV0" in summaries:
        a, r = summaries["AlwaysADD"], summaries["RuleV0"]
        delta = {
            "rule_minus_always_op_acc": r["op_accuracy"] - a["op_accuracy"],
            "rule_minus_always_mean_n_valid": r["mean_n_valid"] - a["mean_n_valid"],
            "rule_minus_always_answer_in_hits": r["mean_answer_in_hits"]
            - a["mean_answer_in_hits"],
        }
        (args.out / "delta.json").write_text(json.dumps(delta, indent=2) + "\n")
        print("delta", json.dumps(delta, indent=2))
    (args.out / "summaries.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
