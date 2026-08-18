#!/usr/bin/env python3
"""Build labeled conflict_v0 pack for write-policy probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "conflict_v0"

# Seed bank (preloaded before each case's incoming claim)
SEEDS = [
    {
        "id": "seed_buddy",
        "text": "Andrew adopted a dog named Buddy from a shelter.",
        "project": "conv-demo",
        "path": ["project", "conv-demo"],
        "kind": "claim",
        "t": "2026-01-01T00:00:00Z",
    },
    {
        "id": "seed_city",
        "text": "Caroline lives in Seattle.",
        "project": "conv-demo",
        "path": ["project", "conv-demo"],
        "kind": "claim",
        "t": "2026-01-01T00:00:00Z",
    },
    {
        "id": "seed_job",
        "text": "Melanie works as a nurse.",
        "project": "conv-demo",
        "path": ["project", "conv-demo"],
        "kind": "claim",
        "t": "2026-01-01T00:00:00Z",
    },
    {
        "id": "seed_other_proj",
        "text": "F1 score on the retrieval task is 0.50.",
        "project": "slm-lab",
        "path": ["project", "slm-lab"],
        "kind": "claim",
        "t": "2026-01-01T00:00:00Z",
    },
]

# Cases: each starts from SEEDS (or subset), then one incoming ManagerInput + gold ops
CASES = [
    {
        "id": "complement_dogs",
        "type": "complement",
        "seed_ids": ["seed_buddy", "seed_city", "seed_job"],
        "incoming": {
            "text": "Andrew later adopted another dog named Scout.",
            "project": "conv-demo",
        },
        "gold_op": "UPDATE",
        "gold_target_id": "seed_buddy",
        "notes": "Buddy+Scout should consolidate (Memory-R1 poster child), not DELETE+ADD",
        "qa": {
            "id": "q_dogs",
            "text": "How many dogs did Andrew adopt?",
            "project": "conv-demo",
            "type": "local",
            "gold_answer": "2",
            "gold_ids": ["seed_buddy"],
        },
    },
    {
        "id": "contradict_city",
        "type": "contradict",
        "seed_ids": ["seed_buddy", "seed_city", "seed_job"],
        "incoming": {
            "text": "Caroline no longer lives in Seattle; she moved to Portland.",
            "project": "conv-demo",
        },
        "gold_op": "UPDATE",
        "gold_target_id": "seed_city",
        "notes": "True correction → UPDATE same id (not keep both cities)",
        "qa": {
            "id": "q_city",
            "text": "Where does Caroline live?",
            "project": "conv-demo",
            "type": "local",
            "gold_answer": "Portland",
            "gold_ids": ["seed_city"],
        },
    },
    {
        "id": "exact_dupe",
        "type": "dupe",
        "seed_ids": ["seed_buddy", "seed_city", "seed_job"],
        "incoming": {
            "text": "Andrew adopted a dog named Buddy from a shelter.",
            "project": "conv-demo",
        },
        "gold_op": "NOOP",
        "gold_target_id": "seed_buddy",
        "notes": "Exact duplicate → NOOP",
        "qa": {
            "id": "q_buddy",
            "text": "What dog did Andrew adopt from a shelter?",
            "project": "conv-demo",
            "type": "local",
            "gold_answer": "Buddy",
            "gold_ids": ["seed_buddy"],
        },
    },
    {
        "id": "new_topic",
        "type": "new_topic",
        "seed_ids": ["seed_buddy", "seed_city", "seed_job"],
        "incoming": {
            "text": "Jon started learning the piano last month.",
            "project": "conv-demo",
        },
        "gold_op": "ADD",
        "gold_target_id": None,
        "notes": "Unrelated new fact → ADD",
        "qa": {
            "id": "q_piano",
            "text": "What instrument did Jon start learning?",
            "project": "conv-demo",
            "type": "local",
            "gold_answer": "piano",
            "gold_ids": [],
        },
    },
    {
        "id": "wrong_project_trap",
        "type": "wrong_project",
        "seed_ids": ["seed_buddy", "seed_city", "seed_job", "seed_other_proj"],
        "incoming": {
            "text": "F1 score on the retrieval task is 0.72.",
            "project": "conv-demo",
        },
        "gold_op": "ADD",
        "gold_target_id": None,
        "notes": "Similar text lives in slm-lab; scoped manager must ADD under conv-demo, not UPDATE other project",
        "qa": {
            "id": "q_f1_demo",
            "text": "What F1 is reported for the retrieval task in this conversation?",
            "project": "conv-demo",
            "type": "local",
            "gold_answer": "0.72",
            "gold_ids": [],
        },
    },
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seeds_by_id = {s["id"]: s for s in SEEDS}
    (args.out / "seeds.jsonl").write_text(
        "\n".join(json.dumps(s) for s in SEEDS) + "\n"
    )
    cases_out = []
    queries = []
    for c in CASES:
        row = {
            **c,
            "seeds": [seeds_by_id[i] for i in c["seed_ids"]],
        }
        cases_out.append(row)
        if c.get("qa"):
            queries.append(c["qa"])

    (args.out / "cases.jsonl").write_text(
        "\n".join(json.dumps(c) for c in cases_out) + "\n"
    )
    (args.out / "queries.jsonl").write_text(
        "\n".join(json.dumps(q) for q in queries) + "\n"
    )
    meta = {
        "n_seeds": len(SEEDS),
        "n_cases": len(CASES),
        "types": sorted({c["type"] for c in CASES}),
        "architecture": "family_A_fact_bank",
        "gold_ops": ["ADD", "UPDATE", "DELETE", "NOOP"],
    }
    (args.out / "META.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.out} cases={len(CASES)}")


if __name__ == "__main__":
    main()
