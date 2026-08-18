#!/usr/bin/env python3
"""Can LOCOMO's QA reward see a false invalidation?

Memory-R1 trains its Memory Manager on R = EM(y_pred, y_gold) over LOCOMO QA
(eq. 5), with no per-operation term. So a DELETE is punished only if some
question's gold answer depends on the deleted memory.

This measures whether that dependency exists:

  A. fact coverage       how many memories are required by ANY question at all.
                         A memory no question needs is free to delete.
  B. evidence arity      how many questions need >=2 memories. A 1-evidence
                         question cannot punish destroying the *other* member
                         of a confusable pair.
  C. pair coverage       the load-bearing one. Among confusable pairs (same
                         conversation, same speaker, cosine >= tau — the shape
                         a manager reads as "contradiction, DELETE+ADD"), how
                         many are jointly required by a single question?
                         That fraction is the upper bound on how often EM can
                         distinguish "merged correctly" from "destroyed one".

Read-only. Uses the embeddings already cached in hierstore.sqlite.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "locomo"


def speaker_of(fact_id: str) -> str:
    parts = fact_id.split("__")
    return parts[2] if len(parts) > 2 else "?"


def load() -> tuple[list[dict], np.ndarray, list[dict]]:
    con = sqlite3.connect(DATA / "hierstore.sqlite")
    facts = [
        {"id": i, "text": t, "project": p}
        for i, t, p in con.execute("select id,text,project from facts")
    ]
    idx = {f["id"]: k for k, f in enumerate(facts)}
    vecs = np.zeros((len(facts), 1536), dtype=np.float32)
    got = 0
    for fid, blob in con.execute("select id,vector from embeddings"):
        if fid in idx:
            vecs[idx[fid]] = np.frombuffer(blob, dtype=np.float32)
            got += 1
    con.close()
    n = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(n == 0, 1.0, n)
    queries = [
        json.loads(l)
        for l in (DATA / "queries.jsonl").read_text().splitlines()
        if l.strip()
    ]
    print(f"facts={len(facts)} embedded={got} queries={len(queries)}")
    return facts, vecs, queries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", default="0.70,0.75,0.80,0.85")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "locomo_reward_blindness.json")
    args = ap.parse_args()

    facts, vecs, queries = load()
    fid = [f["id"] for f in facts]
    idx = {f: k for k, f in enumerate(fid)}
    report: dict = {}

    # ---- A. fact coverage -------------------------------------------------
    needed = Counter()
    for q in queries:
        for g in q.get("gold_ids") or []:
            needed[g] += 1
    covered = sum(1 for f in fid if needed[f] > 0)
    q_with_gold = sum(1 for q in queries if q.get("gold_ids"))
    report["A_fact_coverage"] = {
        "n_facts": len(fid),
        "facts_required_by_some_question": covered,
        "frac": round(covered / len(fid), 4),
        "facts_no_question_needs": len(fid) - covered,
        "n_queries": len(queries),
        "queries_with_gold_ids": q_with_gold,
    }

    # ---- B. evidence arity ------------------------------------------------
    arity = Counter(len(q.get("gold_ids") or []) for q in queries)
    multi = sum(v for k, v in arity.items() if k >= 2)
    report["B_evidence_arity"] = {
        "distribution": {str(k): arity[k] for k in sorted(arity)},
        "queries_needing_2plus_memories": multi,
        "frac_of_joined": round(multi / max(q_with_gold, 1), 4),
    }

    # ---- C. confusable-pair coverage --------------------------------------
    # gold sets, for joint-requirement lookup
    gold_sets = [set(q["gold_ids"]) for q in queries if len(q.get("gold_ids") or []) >= 2]
    pair_needed: set[frozenset] = set()
    for gs in gold_sets:
        gl = sorted(gs)
        for a in range(len(gl)):
            for b in range(a + 1, len(gl)):
                pair_needed.add(frozenset((gl[a], gl[b])))

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for k, f in enumerate(facts):
        groups[(f["project"], speaker_of(f["id"]))].append(k)

    taus = [float(t) for t in args.taus.split(",")]
    report["C_pair_coverage"] = {}
    examples: dict[str, list] = {}
    for tau in taus:
        n_pairs = 0
        n_joint = 0
        n_either_needed = 0
        ex: list = []
        for _, rows in groups.items():
            if len(rows) < 2:
                continue
            sub = vecs[rows]
            sim = sub @ sub.T
            for a in range(len(rows)):
                for b in range(a + 1, len(rows)):
                    if sim[a, b] < tau:
                        continue
                    ia, ib = fid[rows[a]], fid[rows[b]]
                    n_pairs += 1
                    if needed[ia] > 0 or needed[ib] > 0:
                        n_either_needed += 1
                    if frozenset((ia, ib)) in pair_needed:
                        n_joint += 1
                    elif len(ex) < 6:
                        ex.append(
                            {
                                "cos": round(float(sim[a, b]), 3),
                                "a": facts[rows[a]]["text"],
                                "b": facts[rows[b]]["text"],
                                "a_needed_by_n_q": needed[ia],
                                "b_needed_by_n_q": needed[ib],
                            }
                        )
        report["C_pair_coverage"][f"tau_{tau}"] = {
            "confusable_pairs": n_pairs,
            "pairs_jointly_required_by_one_question": n_joint,
            "frac_joint": round(n_joint / n_pairs, 4) if n_pairs else None,
            "pairs_where_neither_member_is_ever_needed": n_pairs - n_either_needed,
        }
        examples[f"tau_{tau}"] = ex
    report["C_examples_unpunished"] = examples

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "C_examples_unpunished"}, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
