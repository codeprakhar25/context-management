#!/usr/bin/env python3
"""Sequential LoCoMo ingest THROUGH a memory manager.

The existing ingest (scripts/ingest_locomo.py) is append_only — it CREATEs every
observation and no manager ever runs. That store cannot exhibit false
invalidation because nothing is ever deleted.

Here each observation is fed to the manager one at a time, in session order,
against the store built so far. The manager chooses ADD / UPDATE / DELETE /
NOOP. One store per (arm, conversation).

Gold join
---------
`ManagerInput.fact_id` pre-assigns the incoming observation's ORIGINAL id, so a
manager that ADDs keeps the id LoCoMo's `evidence` field points at and
queries.jsonl `gold_ids` still resolve. Ids only go missing when the manager
chose not to add — which is exactly what we want to count.

Outputs per arm:
  <out>/<arm>/store.sqlite     final memory bank (all conversations)
  <out>/<arm>/ops.jsonl        every op, with the incoming observation id
  <out>/<arm>/ingest.json      op counts + manager stats + cost
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
from harness.store import HierStore  # noqa: E402
from scripts.ingest_locomo import fallback_turns, iter_observations  # noqa: E402

DATA = ROOT / "data" / "locomo"


def session_dates(sample: dict) -> dict[str, str]:
    """session_1_observation -> the session's wall-clock date.

    LoCoMo observations are written in speaker-relative time ("recently", "last
    Saturday") while cat2 gold answers are absolute dates. Without the session
    timestamp the reader cannot answer a temporal question no matter which
    memories survive, which floors QA for every arm and makes the QA signal
    useless for comparing memory banks. Mem0/Memory-R1 pipelines attach it too.
    """
    conv = sample.get("conversation") or {}
    out: dict[str, str] = {}
    for key, val in conv.items():
        if key.endswith("_date_time") and isinstance(val, str):
            stem = key[: -len("_date_time")]
            out[f"{stem}_observation"] = val
    return out


def build_manager(arm: str, embedder, thresh: dict, chat_model: str, provider: str, prompt: str):
    if arm == "AlwaysADD":
        return AlwaysADD()
    if arm == "RuleV0":
        return RuleV0(
            embedder=embedder,
            update_thresh=thresh["update_thresh"],
            noop_thresh=thresh["noop_thresh"],
        )
    if arm == "LLMv0":
        return LLMv0(
            embedder=embedder,
            top_m=8,
            chat_model=chat_model,
            chat_provider=provider,
            prompt=prompt,
        )
    raise ValueError(arm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="AlwaysADD", choices=["AlwaysADD", "RuleV0", "LLMv0"])
    ap.add_argument("--chat-model", default="gpt-4o")
    ap.add_argument("--chat-provider", default="openai", choices=["openai", "openrouter"])
    ap.add_argument("--prompt", default="neutral", choices=["neutral", "cond"])
    ap.add_argument("--locomo", type=Path, default=DATA / "locomo10.json")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "locomo_managed")
    ap.add_argument("--tag", default=None, help="arm dir name; defaults to arm[_model]")
    ap.add_argument("--limit-convs", type=int, default=None, help="smoke test: first N conversations")
    ap.add_argument("--limit-obs", type=int, default=None, help="smoke test: first N observations per conversation")
    ap.add_argument("--no-date-prefix", action="store_true", help="disable session-date stamping (ablation)")
    ap.add_argument("--replace", action="store_true")
    args = ap.parse_args()

    tag = args.tag or (
        args.arm if args.arm != "LLMv0" else f"LLMv0_{args.chat_model.replace('/', '-')}_{args.prompt}"
    )
    outdir = args.out / tag
    outdir.mkdir(parents=True, exist_ok=True)
    db_path = outdir / "store.sqlite"
    if args.replace and db_path.exists():
        db_path.unlink()
    if db_path.exists():
        raise SystemExit(f"{db_path} exists — pass --replace to overwrite")

    samples = json.loads(args.locomo.read_text())
    if args.limit_convs:
        samples = samples[: args.limit_convs]

    thresh_path = DATA / "rulev0_thresh.json"
    thresh = (
        json.loads(thresh_path.read_text())
        if thresh_path.exists()
        else {"update_thresh": 0.78, "noop_thresh": 0.92}
    )
    embedder = None
    if args.arm in ("RuleV0", "LLMv0"):
        embedder = Embedder(cache_path=ROOT / "runs" / "_embed_cache" / "locomo_managed.json")

    mgr = build_manager(args.arm, embedder, thresh, args.chat_model, args.chat_provider, args.prompt)
    print(f"arm={tag} manager={mgr.name} convs={len(samples)}", flush=True)

    op_counts: Counter = Counter()
    ops_rows: list[dict] = []
    n_obs = 0
    t0 = time.perf_counter()

    with HierStore(db_path) as store:
        for si, sample in enumerate(samples):
            sid = sample["sample_id"]
            rows = iter_observations(sample) or fallback_turns(sample)
            if args.limit_obs:
                rows = rows[: args.limit_obs]
            dates = session_dates(sample)
            for oi, (fid, text, t_hint) in enumerate(rows):
                stamp = dates.get(t_hint)
                if stamp and not args.no_date_prefix:
                    text = f"[{stamp}] {text}"
                inp = ManagerInput(
                    text=text,
                    project=sid,
                    path=["project", sid],
                    kind="observation",
                    t=dates.get(t_hint) or t_hint,
                    fact_id=fid,  # keep LoCoMo's id so gold_ids still resolve
                )
                dec = mgr.decide(inp, store)
                logs = store.apply_ops(dec.ops, manager=mgr.name)
                if hasattr(mgr, "invalidate_embed_cache"):
                    mgr.invalidate_embed_cache([l["fact_id"] for l in logs if l.get("fact_id")])
                n_obs += 1
                for l in logs:
                    op_counts[l["op"].upper()] += 1
                    ops_rows.append(
                        {
                            "conv": sid,
                            "obs_index": oi,
                            "incoming_id": fid,
                            "incoming_text": text,
                            "op": l["op"].upper(),
                            "fact_id": l.get("fact_id"),
                            "before_text": l.get("before_text"),
                            "after_text": l.get("after_text"),
                        }
                    )
                if n_obs % 100 == 0:
                    el = time.perf_counter() - t0
                    print(f"  [{si+1}/{len(samples)}] obs={n_obs} ops={dict(op_counts)} {el:.0f}s", flush=True)
        final_valid = len(store.read_all(valid_only=True))
        final_all = len(store.read_all(valid_only=False))

    (outdir / "ops.jsonl").write_text("\n".join(json.dumps(r) for r in ops_rows) + "\n")
    meta = {
        "arm": tag,
        "manager": mgr.name,
        "n_conversations": len(samples),
        "n_observations_fed": n_obs,
        "date_prefix": not args.no_date_prefix,
        "op_counts": dict(op_counts),
        "final_facts_valid": final_valid,
        "final_facts_total": final_all,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "db": str(db_path),
    }
    if isinstance(mgr, LLMv0):
        meta["manager_stats"] = mgr.stats()
    (outdir / "ingest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
