#!/usr/bin/env python3
"""Read FI out of an in-progress ingest, without touching the running process.

HierStore commits each op to its `ops_log` table as it happens, so the sqlite
file is a live view of the run. `ops.jsonl` is only written at exit — this
rebuilds the equivalent rows from `ops_log` plus the source observations.

Only conversations that finished are scored. Ingest walks conversations in
order, so every project except the last one seen in `ops_log` is complete; the
last is mid-flight and is dropped.

Read-only, no API calls. Safe to run at any time.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.write_metrics import fi_metrics  # noqa: E402
from scripts.ingest_locomo import fallback_turns, iter_observations  # noqa: E402

DATA = ROOT / "data" / "locomo"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", type=Path, required=True)
    ap.add_argument("--locomo", type=Path, default=DATA / "locomo10.json")
    ap.add_argument("--queries", type=Path, default=DATA / "queries.jsonl")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.arm_dir / 'store.sqlite'}?mode=ro", uri=True)
    ops = [
        {"op": r[0], "fact_id": r[1], "project": r[2]}
        for r in con.execute("SELECT op, fact_id, project FROM ops_log ORDER BY rowid")
    ]
    if not ops:
        raise SystemExit("no ops yet")

    seen_order: list[str] = []
    for o in ops:
        if o["project"] and (not seen_order or seen_order[-1] != o["project"]):
            if o["project"] not in seen_order:
                seen_order.append(o["project"])
    in_flight = seen_order[-1]
    complete = set(seen_order[:-1])

    samples = {s["sample_id"]: s for s in json.loads(args.locomo.read_text())}
    incoming_by_conv = {
        sid: [fid for fid, _, _ in (iter_observations(s) or fallback_turns(s))]
        for sid, s in samples.items()
    }

    # ops.jsonl equivalent, restricted to finished conversations
    ops_rows = [
        {"op": o["op"].upper(), "fact_id": o["fact_id"], "incoming_id": None}
        for o in ops
        if o["project"] in complete
    ]
    # every observation of a finished conversation was offered to the manager
    fed = {fid for c in complete for fid in incoming_by_conv.get(c, [])}
    ops_rows += [{"op": "_FED", "fact_id": None, "incoming_id": fid} for fid in fed]

    valid_ids = {
        r[0]
        for r in con.execute("SELECT id FROM facts WHERE valid = 1 AND project IN (%s)"
                             % ",".join("?" * len(complete)), tuple(complete))
    } if complete else set()

    queries = [json.loads(l) for l in args.queries.read_text().splitlines() if l.strip()]
    queries = [q for q in queries if q["project"] in complete]
    required = {g for q in queries for g in (q.get("gold_ids") or [])}

    fi = fi_metrics(required, valid_ids, ops_rows)
    counts: dict[str, int] = {}
    for o in ops_rows:
        if o["op"] != "_FED":
            counts[o["op"]] = counts.get(o["op"], 0) + 1
    print(
        json.dumps(
            {
                "complete_conversations": sorted(complete),
                "in_flight_conversation": in_flight,
                "observations_fed": len(fed),
                "op_counts": counts,
                "facts_valid": len(valid_ids),
                "queries_covered": len(queries),
                "fi": fi,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
