#!/usr/bin/env python3
"""Build conflict_v1: ~200 labeled write-policy cases, 80/20 dev/test split.

Types:
  complement      — gold UPDATE (same id consolidate)
  control         — gold DELETE(old)+ADD(new) superseded
  condition       — both true different scope; gold ADD under incoming project
  dupe            — gold NOOP
  new_topic       — gold ADD
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "conflict_v1"

PEOPLE = [
    "Alex", "Blake", "Casey", "Drew", "Eden", "Finley", "Gray", "Harper",
    "Indie", "Jordan", "Kai", "Logan", "Morgan", "Noah", "Oakley", "Parker",
    "Quinn", "Riley", "Sage", "Taylor", "Uma", "Val", "Wren", "Yael", "Zion",
]
CITIES_A = [
    "Seattle", "Portland", "Austin", "Denver", "Boston", "Chicago", "Miami",
    "Atlanta", "Phoenix", "Dallas",
]
CITIES_B = [
    "Toronto", "Vancouver", "Montreal", "Ottawa", "Calgary", "Berlin",
    "Paris", "Madrid", "Lisbon", "Dublin",
]
PETS = ["dog", "cat", "rabbit", "parrot", "hamster"]
PET_NAMES_A = ["Buddy", "Luna", "Max", "Coco", "Bear"]
PET_NAMES_B = ["Scout", "Nala", "Rex", "Mochi", "Pixel"]
JOBS = ["nurse", "teacher", "engineer", "designer", "chef", "lawyer", "barista"]
HOBBIES = ["piano", "climbing", "pottery", "chess", "running", "photography"]
METRICS = ["F1", "EM", "recall", "precision", "BLEU"]


def _id(prefix: str, i: int) -> str:
    return f"{prefix}_{i:04d}"


def make_cases(n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    cases: list[dict] = []
    # roughly balanced
    types = (
        ["complement"] * (n // 5)
        + ["control"] * (n // 5)
        + ["condition"] * (n // 5)
        + ["dupe"] * (n // 5)
        + ["new_topic"] * (n - 4 * (n // 5))
    )
    rng.shuffle(types)

    for i, typ in enumerate(types):
        person = PEOPLE[i % len(PEOPLE)]
        other = PEOPLE[(i + 3) % len(PEOPLE)]
        proj = f"proj_{(i % 8):02d}"
        other_proj = f"proj_{((i + 1) % 8):02d}"
        seed_id = _id("seed", i)
        city_a = CITIES_A[i % len(CITIES_A)]
        city_b = CITIES_B[i % len(CITIES_B)]
        pet = PETS[i % len(PETS)]
        pn_a = PET_NAMES_A[i % len(PET_NAMES_A)]
        pn_b = PET_NAMES_B[i % len(PET_NAMES_B)]
        job = JOBS[i % len(JOBS)]
        hobby = HOBBIES[i % len(HOBBIES)]
        metric = METRICS[i % len(METRICS)]
        v_old = round(0.40 + (i % 10) * 0.02, 2)
        v_new = round(v_old + 0.15, 2)

        if typ == "complement":
            seed_text = f"{person} adopted a {pet} named {pn_a}."
            incoming = f"{person} later adopted another {pet} named {pn_b}."
            gold_ops = [{"op": "UPDATE", "target_id": seed_id}]
            notes = "complement: consolidate pets"
        elif typ == "control":
            seed_text = f"{person} lives in {city_a}."
            incoming = f"{person} no longer lives in {city_a}; moved to {city_b}."
            gold_ops = [
                {"op": "DELETE", "target_id": seed_id},
                {"op": "ADD", "target_id": None},
            ]
            notes = "control: superseded — DELETE+ADD"
        elif typ == "condition":
            # same metric claim true under different project scopes
            seed_text = f"{metric} on the retrieval task is {v_old}."
            incoming = f"{metric} on the retrieval task is {v_new}."
            # seed lives in OTHER project; incoming under proj → gold ADD (both keep)
            seed_proj = other_proj
            gold_ops = [{"op": "ADD", "target_id": None}]
            notes = "condition-scoped: both true different projects"
            seeds = [
                {
                    "id": seed_id,
                    "text": seed_text,
                    "project": seed_proj,
                    "path": ["project", seed_proj],
                    "kind": "claim",
                    "t": "2026-01-01T00:00:00Z",
                },
                {
                    "id": f"{seed_id}_local",
                    "text": f"{person} works as a {job}.",
                    "project": proj,
                    "path": ["project", proj],
                    "kind": "claim",
                    "t": "2026-01-01T00:00:00Z",
                },
            ]
            cases.append(
                {
                    "id": f"c_{i:04d}_{typ}",
                    "type": typ,
                    "split": None,
                    "seeds": seeds,
                    "incoming": {"text": incoming, "project": proj},
                    "gold_ops": gold_ops,
                    "notes": notes,
                }
            )
            continue
        elif typ == "dupe":
            seed_text = f"{person} works as a {job}."
            incoming = f"{person} works as a {job}."
            gold_ops = [{"op": "NOOP", "target_id": seed_id}]
            notes = "exact duplicate"
        else:  # new_topic
            seed_text = f"{person} works as a {job}."
            incoming = f"{other} started learning {hobby}."
            gold_ops = [{"op": "ADD", "target_id": None}]
            notes = "unrelated new topic"

        seeds = [
            {
                "id": seed_id,
                "text": seed_text,
                "project": proj,
                "path": ["project", proj],
                "kind": "claim",
                "t": "2026-01-01T00:00:00Z",
            },
            {
                "id": f"{seed_id}_b",
                "text": f"{other} lives in {city_a}." if typ != "control" else f"{other} works as a {job}.",
                "project": proj,
                "path": ["project", proj],
                "kind": "claim",
                "t": "2026-01-01T00:00:00Z",
            },
        ]
        # sprinkle a global fact
        if i % 4 == 0:
            seeds.append(
                {
                    "id": f"{seed_id}_g",
                    "text": "User prefers short answers.",
                    "project": None,
                    "path": ["global"],
                    "kind": "claim",
                    "t": "2026-01-01T00:00:00Z",
                }
            )

        cases.append(
            {
                "id": f"c_{i:04d}_{typ}",
                "type": typ,
                "split": None,
                "seeds": seeds,
                "incoming": {"text": incoming, "project": proj},
                "gold_ops": gold_ops,
                "notes": notes,
            }
        )
    return cases


def assign_split(cases: list[dict], seed: int = 7, test_frac: float = 0.2) -> None:
    rng = random.Random(seed + 1)
    by_type: dict[str, list] = {}
    for c in cases:
        by_type.setdefault(c["type"], []).append(c)
    for typ, rows in by_type.items():
        rng.shuffle(rows)
        n_test = max(1, int(round(len(rows) * test_frac)))
        for i, c in enumerate(rows):
            c["split"] = "test" if i < n_test else "dev"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cases = make_cases(args.n, seed=args.seed)
    assign_split(cases, seed=args.seed)
    blob = "\n".join(json.dumps(c) for c in cases) + "\n"
    (args.out / "cases.jsonl").write_text(blob)
    h = hashlib.sha256(blob.encode()).hexdigest()[:16]
    meta = {
        "n": len(cases),
        "n_dev": sum(1 for c in cases if c["split"] == "dev"),
        "n_test": sum(1 for c in cases if c["split"] == "test"),
        "types": sorted({c["type"] for c in cases}),
        "by_type": {
            t: sum(1 for c in cases if c["type"] == t)
            for t in sorted({c["type"] for c in cases})
        },
        "gold_control": "DELETE+ADD",
        "cases_sha256_16": h,
        "seed": args.seed,
    }
    (args.out / "META.json").write_text(json.dumps(meta, indent=2) + "\n")
    # default RuleV0 thresh placeholder — tune on dev only
    (args.out / "rulev0_thresh.json").write_text(
        json.dumps(
            {
                "update_thresh": 0.88,
                "noop_thresh": 0.97,
                "tuned_on": "unset",
                "note": "Set via scripts/tune_rulev0_dev.py on conflict_v1 split=dev only",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
