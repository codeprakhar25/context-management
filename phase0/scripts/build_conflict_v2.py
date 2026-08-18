#!/usr/bin/env python3
"""Build conflict_v2: same-scope condition cases the manager can actually see.

Why v2
------
In v1 the `condition` class put the two claims in *different* projects, so
`scoped_candidates()` filtered the twin out and the manager never saw a pair.
Every manager scored 1.0 or 0.125 there for reasons unrelated to conditions.

v2 puts both claims in the SAME project. The condition lives in the text
("in the frontend repo", "on staging", "while on the SF rotation"). Retrieval
surfaces both, so the manager has to decide.

Types (all same-project):
  condition   two claims, different condition, both true  -> ADD  (keep both)
  supersede   two claims, SAME condition, old one dead    -> DELETE+ADD
  dupe        same condition, same value, reworded        -> NOOP
  new_topic   unrelated claim from another domain         -> ADD

`condition` and `supersede` are built from the same domains, templates and
vocabulary. That pairing is the point: a manager that always ADDs aces
condition and fails supersede; one that always merges does the reverse.
Neither can be passed by a policy that ignores the condition slot.

Surface form is varied on purpose — v1's condition class was one sentence
shape 40 times, so a high score could not be distinguished from string
reflex. Each domain here has 3-4 phrasings plus separate supersession
phrasings, drawn independently per case.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "conflict_v2"

PEOPLE = [
    "Priya", "Sam", "Ines", "Tomas", "Mei", "Omar", "Lena", "Diego",
    "Aisha", "Noor", "Yuki", "Rafa", "Hana", "Ivan", "Zoë", "Karim",
]

DOMAINS: list[dict] = [
    {
        "name": "pkgmgr",
        "conds": [
            "the frontend repo", "legacy-api", "the mobile app", "docs-site",
            "billing-svc", "edge-proxy", "the admin panel", "data-pipeline",
        ],
        "vals": ["pnpm", "npm", "yarn", "bun"],
        "templates": [
            "In {cond} we use {val}.",
            "{cond} uses {val} as its package manager.",
            "Package manager for {cond}: {val}.",
            "For {cond}, install deps with {val}.",
        ],
        "supersede": [
            "{cond} has migrated off {old}; it now uses {val}.",
            "We switched {cond} from {old} to {val}.",
            "{cond} no longer uses {old} — it is on {val} now.",
        ],
    },
    {
        "name": "datastore",
        "conds": [
            "staging", "production", "the dev sandbox", "the EU region",
            "the canary cluster", "CI",
        ],
        "vals": ["Postgres 14", "Postgres 16", "MySQL 8", "SQLite", "Aurora"],
        "templates": [
            "{cond} runs {val}.",
            "The database in {cond} is {val}.",
            "{val} backs {cond}.",
            "In {cond}, the store is {val}.",
        ],
        "supersede": [
            "{cond} was upgraded from {old} to {val}.",
            "{cond} no longer runs {old}; it is on {val}.",
            "The {old} instance in {cond} was replaced by {val}.",
        ],
    },
    {
        "name": "platform",
        "conds": [
            "iOS", "Android", "the web client", "the desktop app", "the CLI",
        ],
        "vals": [
            "a native share sheet", "a web-view fallback", "a modal dialog",
            "an inline drawer",
        ],
        "templates": [
            "On {cond}, sharing opens {val}.",
            "{cond} shows {val} when sharing.",
            "Share flow on {cond}: {val}.",
            "Sharing from {cond} uses {val}.",
        ],
        "supersede": [
            "{cond} replaced {old} with {val} for sharing.",
            "Sharing on {cond} moved off {old} to {val}.",
            "{cond} no longer uses {old} when sharing — now {val}.",
        ],
    },
    {
        "name": "standup",
        "conds": [
            "when working from Berlin", "while on the SF rotation",
            "during on-call weeks", "on Fridays", "when travelling",
        ],
        "vals": [
            "async standups", "live standups", "written updates only",
            "no standup at all",
        ],
        "templates": [
            "{person} prefers {val} {cond}.",
            "{cond}, {person} wants {val}.",
            "{person}'s standup preference {cond} is {val}.",
            "{person} asked for {val} {cond}.",
        ],
        "supersede": [
            "{person} no longer wants {old} {cond}; now {val}.",
            "{person} changed their {cond} preference from {old} to {val}.",
            "Drop {old} {cond} for {person} — it is {val} now.",
        ],
    },
    {
        "name": "evalsplit",
        "conds": [
            "the val split", "the test split", "the held-out set",
            "the 3600s bin", "the 600s bin",
        ],
        "vals": ["0.54", "0.61", "0.47", "0.68", "0.72"],
        "templates": [
            "Recall on {cond} is {val}.",
            "{cond} scores {val} recall.",
            "On {cond} we measure {val} recall.",
            "Recall, {cond}: {val}.",
        ],
        "supersede": [
            "Recall on {cond} was re-measured at {val}, not {old}.",
            "The {old} figure for {cond} was wrong; it is {val}.",
            "Correction — {cond} recall is {val}, superseding {old}.",
        ],
    },
    {
        "name": "mergerule",
        "conds": [
            "as a reviewer", "as an author", "for external contributors",
            "for the release branch", "for hotfixes",
        ],
        "vals": [
            "two approvals", "one approval", "no approval",
            "a security sign-off",
        ],
        "templates": [
            "{cond}, merging requires {val}.",
            "Merge rule {cond}: {val}.",
            "{cond} you need {val} to merge.",
            "Merging {cond} is gated on {val}.",
        ],
        "supersede": [
            "The rule {cond} changed from {old} to {val}.",
            "{cond} no longer needs {old}; it needs {val}.",
            "We dropped {old} {cond} in favour of {val}.",
        ],
    },
]


def _fill(tpl: str, *, cond: str, val: str, person: str, old: str = "") -> str:
    return tpl.format(cond=cond, val=val, person=person, old=old)


def _distractor(rng: random.Random, exclude: str, person: str) -> str:
    dom = rng.choice([d for d in DOMAINS if d["name"] != exclude])
    return _fill(
        rng.choice(dom["templates"]),
        cond=rng.choice(dom["conds"]),
        val=rng.choice(dom["vals"]),
        person=person,
    )


def make_cases(n: int, seed: int = 11) -> list[dict]:
    rng = random.Random(seed)
    per = n // 4
    types = (
        ["condition"] * per
        + ["supersede"] * per
        + ["dupe"] * per
        + ["new_topic"] * (n - 3 * per)
    )
    rng.shuffle(types)

    cases: list[dict] = []
    for i, typ in enumerate(types):
        dom = DOMAINS[i % len(DOMAINS)]
        person = PEOPLE[i % len(PEOPLE)]
        proj = f"proj_{(i % 6):02d}"
        seed_id = f"s2_{i:04d}"

        cond_a, cond_b = rng.sample(dom["conds"], 2)
        val_a, val_b = rng.sample(dom["vals"], 2)
        t_seed = rng.choice(dom["templates"])
        t_in = rng.choice(dom["templates"])

        if typ == "condition":
            seed_text = _fill(t_seed, cond=cond_a, val=val_a, person=person)
            incoming = _fill(t_in, cond=cond_b, val=val_b, person=person)
            gold_ops = [{"op": "ADD", "target_id": None}]
            notes = f"{dom['name']}: both true — {cond_a!r} vs {cond_b!r}"
        elif typ == "supersede":
            seed_text = _fill(t_seed, cond=cond_a, val=val_a, person=person)
            incoming = _fill(
                rng.choice(dom["supersede"]),
                cond=cond_a,
                val=val_b,
                person=person,
                old=val_a,
            )
            gold_ops = [
                {"op": "DELETE", "target_id": seed_id},
                {"op": "ADD", "target_id": None},
            ]
            notes = f"{dom['name']}: same condition {cond_a!r}, {val_a} -> {val_b}"
        elif typ == "dupe":
            seed_text = _fill(t_seed, cond=cond_a, val=val_a, person=person)
            t_alt = rng.choice([t for t in dom["templates"] if t != t_seed])
            incoming = _fill(t_alt, cond=cond_a, val=val_a, person=person)
            gold_ops = [{"op": "NOOP", "target_id": seed_id}]
            notes = f"{dom['name']}: same condition + same value, reworded"
        else:  # new_topic
            seed_text = _fill(t_seed, cond=cond_a, val=val_a, person=person)
            incoming = _distractor(rng, dom["name"], person)
            gold_ops = [{"op": "ADD", "target_id": None}]
            notes = f"{dom['name']} seed, unrelated incoming"

        seeds = [
            {
                "id": seed_id,
                "text": seed_text,
                "project": proj,
                "path": ["project", proj],
                "kind": "claim",
                "t": "2026-01-01T00:00:00Z",
            }
        ]
        # 1-2 same-project distractors so RELATED is never a single item
        for j in range(1 + (i % 2)):
            seeds.append(
                {
                    "id": f"{seed_id}_d{j}",
                    "text": _distractor(rng, dom["name"], PEOPLE[(i + j + 1) % len(PEOPLE)]),
                    "project": proj,
                    "path": ["project", proj],
                    "kind": "claim",
                    "t": "2026-01-01T00:00:00Z",
                }
            )

        cases.append(
            {
                "id": f"v2_{i:04d}_{typ}",
                "type": typ,
                "domain": dom["name"],
                "split": None,
                "seeds": seeds,
                "incoming": {"text": incoming, "project": proj},
                "gold_ops": gold_ops,
                "notes": notes,
            }
        )
    return cases


def assign_split(cases: list[dict], seed: int = 11, test_frac: float = 0.2) -> None:
    """Stratify on (type, domain) so no domain lands wholly in one split."""
    rng = random.Random(seed + 1)
    buckets: dict[tuple[str, str], list] = {}
    for c in cases:
        buckets.setdefault((c["type"], c["domain"]), []).append(c)
    for rows in buckets.values():
        rng.shuffle(rows)
        n_test = max(1, int(round(len(rows) * test_frac)))
        for i, c in enumerate(rows):
            c["split"] = "test" if i < n_test else "dev"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cases = make_cases(args.n, seed=args.seed)
    assign_split(cases, seed=args.seed)
    blob = "\n".join(json.dumps(c) for c in cases) + "\n"
    (args.out / "cases.jsonl").write_text(blob)

    types = sorted({c["type"] for c in cases})
    meta = {
        "n": len(cases),
        "n_dev": sum(1 for c in cases if c["split"] == "dev"),
        "n_test": sum(1 for c in cases if c["split"] == "test"),
        "types": types,
        "by_type": {t: sum(1 for c in cases if c["type"] == t) for t in types},
        "by_domain": {
            d["name"]: sum(1 for c in cases if c["domain"] == d["name"])
            for d in DOMAINS
        },
        "scope": "same-project (condition carried in text)",
        "gold_supersede": "DELETE+ADD",
        "unique_incoming_texts": len({c["incoming"]["text"] for c in cases}),
        "cases_sha256_16": hashlib.sha256(blob.encode()).hexdigest()[:16],
        "seed": args.seed,
    }
    (args.out / "META.json").write_text(json.dumps(meta, indent=2) + "\n")
    (args.out / "rulev0_thresh.json").write_text(
        json.dumps(
            {
                "update_thresh": 0.78,
                "noop_thresh": 0.92,
                "tuned_on": "unset — carried over from conflict_v1 real-embed",
                "embed": "text-embedding-3-small",
                "note": "Re-tune on conflict_v2 split=dev only; never on test.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
