#!/usr/bin/env python3
"""Ingest LoCoMo observations into HierStore via CREATE (append_only).

Also writes queries.jsonl for Flat vs Hier eval.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.store import HierStore

# Memory-R1 / OmniMemEval: drop adversarial (category 5)
ADVERSARIAL_CATEGORY = 5


def iter_observations(sample: dict) -> list[tuple[str, str, str]]:
    """Yield (fact_id, text, t_hint) from observation tree."""
    sid = sample["sample_id"]
    out: list[tuple[str, str, str]] = []
    obs = sample.get("observation") or {}
    for sess_key, speakers in obs.items():
        if not isinstance(speakers, dict):
            continue
        for speaker, items in speakers.items():
            if not isinstance(items, list):
                continue
            for i, item in enumerate(items):
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    text = item[0]
                    evid = item[1] if len(item) > 1 else str(i)
                elif isinstance(item, str):
                    text, evid = item, str(i)
                else:
                    continue
                text = (text or "").strip()
                if not text:
                    continue
                fid = f"{sid}__{sess_key}__{speaker}__{evid}__{i}".replace(" ", "_")
                out.append((fid, text, sess_key))
    return out


def fallback_turns(sample: dict) -> list[tuple[str, str, str]]:
    """If no observations, use conversation turns."""
    sid = sample["sample_id"]
    conv = sample.get("conversation") or {}
    out: list[tuple[str, str, str]] = []
    for key, val in conv.items():
        if not key.startswith("session_") or key.endswith("date_time"):
            continue
        if not isinstance(val, list):
            continue
        for i, turn in enumerate(val):
            if not isinstance(turn, dict):
                continue
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            speaker = turn.get("speaker", "unk")
            dia = turn.get("dia_id", str(i))
            fid = f"{sid}__{key}__{speaker}__{dia}"
            out.append((fid, f"{speaker}: {text}", key))
    return out


def evidence_to_gold_ids(fact_ids: list[str], evidence: list | None) -> list[str]:
    """Map LoCoMo evidence tokens like 'D1:3' → fact ids containing __D1:3__."""
    if not evidence:
        return []
    out: list[str] = []
    for e in evidence:
        token = f"__{e}__"
        for fid in fact_ids:
            # fid shape: {sid}__{sess}__{speaker}__{evid}__{i}
            if token in f"__{fid}__":
                out.append(fid)
    # unique, preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for fid in out:
        if fid not in seen:
            seen.add(fid)
            uniq.append(fid)
    return uniq


def build_queries(
    samples: list[dict],
    fact_ids_by_sample: dict[str, list[str]] | None = None,
    exclude_adversarial: bool = True,
) -> list[dict]:
    qs: list[dict] = []
    n = 0
    fact_ids_by_sample = fact_ids_by_sample or {}
    for sample in samples:
        sid = sample["sample_id"]
        fids = fact_ids_by_sample.get(sid, [])
        for q in sample.get("qa") or []:
            cat = q.get("category")
            if exclude_adversarial and cat == ADVERSARIAL_CATEGORY:
                continue
            ans = q.get("answer")
            if ans is None or ans == "":
                continue
            n += 1
            evid = q.get("evidence") or []
            gold_ids = evidence_to_gold_ids(fids, evid)
            qs.append(
                {
                    "id": f"q_{sid}_{n:04d}",
                    "text": q["question"],
                    "type": f"cat{cat}",
                    "project": sid,
                    "gold_ids": gold_ids,
                    "gold_answer": str(ans),
                    "category": cat,
                    "evidence": evid,
                    "notes": (
                        "locomo QA; gold_ids joined from evidence"
                        if gold_ids
                        else "locomo QA; evidence unmatched"
                    ),
                }
            )
    return qs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--locomo",
        type=Path,
        default=ROOT / "data" / "locomo" / "locomo10.json",
    )
    ap.add_argument(
        "--db",
        type=Path,
        default=ROOT / "data" / "locomo" / "hierstore.sqlite",
    )
    ap.add_argument(
        "--queries-out",
        type=Path,
        default=ROOT / "data" / "locomo" / "queries.jsonl",
    )
    ap.add_argument("--sample-id", type=str, default=None, help="ingest only one conversation")
    ap.add_argument("--replace", action="store_true", help="delete existing db first")
    args = ap.parse_args()

    if not args.locomo.exists():
        raise SystemExit(f"missing {args.locomo}; download locomo10.json first")

    samples = json.loads(args.locomo.read_text())
    if args.sample_id:
        samples = [s for s in samples if s["sample_id"] == args.sample_id]
        if not samples:
            raise SystemExit(f"sample_id not found: {args.sample_id}")

    if args.replace and args.db.exists():
        args.db.unlink()

    n_facts = 0
    fact_ids_by_sample: dict[str, list[str]] = {}
    with HierStore(args.db) as store:
        assert store.conflict_policy() == "append_only"
        for sample in samples:
            sid = sample["sample_id"]
            rows = iter_observations(sample)
            if not rows:
                rows = fallback_turns(sample)
            fact_ids_by_sample[sid] = []
            for fid, text, t_hint in rows:
                try:
                    store.create(
                        {
                            "id": fid,
                            "text": text,
                            "path": ["project", sid],
                            "project": sid,
                            "kind": "observation",
                            "t": t_hint,
                        }
                    )
                    n_facts += 1
                    fact_ids_by_sample[sid].append(fid)
                except ValueError:
                    # duplicate id — still index for gold join
                    fact_ids_by_sample[sid].append(fid)

    queries = build_queries(samples, fact_ids_by_sample=fact_ids_by_sample)
    args.queries_out.parent.mkdir(parents=True, exist_ok=True)
    with args.queries_out.open("w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    n_with_gold = sum(1 for q in queries if q["gold_ids"])
    meta = {
        "db": str(args.db),
        "n_conversations": len(samples),
        "n_facts_created": n_facts,
        "n_queries": len(queries),
        "n_queries_with_gold_ids": n_with_gold,
        "gold_ids_coverage": round(n_with_gold / len(queries), 4) if queries else 0.0,
        "conflict_policy": "append_only",
        "excluded_category": ADVERSARIAL_CATEGORY,
        "sample_ids": [s["sample_id"] for s in samples],
    }
    (args.db.parent / "INGEST_META.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
