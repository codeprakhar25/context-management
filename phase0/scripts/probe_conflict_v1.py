#!/usr/bin/env python3
"""Probe managers on conflict_v1 (dev or test). Reports op_acc + invalidation rates."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

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
from harness.write_metrics import (  # noqa: E402
    aggregate_write,
    invalidation_flags,
    ops_match_lenient,
    ops_match_strict,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def run_manager(
    name: str,
    cases: list[dict],
    *,
    embedder: Embedder | None,
    thresh: dict,
    use_hash: bool,
    chat_model: str = "gpt-4o-mini",
    chat_provider: str = "openai",
    prompt: str = "neutral",
) -> tuple[list[dict], dict]:
    results = []

    def hash_embed(texts: list[str], dim: int = 64) -> np.ndarray:
        import re

        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for t in re.findall(r"[a-z0-9]+", text.lower()):
                out[i, hash(t) % dim] += 1.0
            n = float(np.linalg.norm(out[i]))
            if n > 0:
                out[i] /= n
        return out

    if name == "AlwaysADD":
        mgr: AlwaysADD | RuleV0 | LLMv0 = AlwaysADD()
    elif name == "RuleV0":
        if use_hash:
            mgr = RuleV0(
                embed_fn=hash_embed,
                update_thresh=thresh.get("update_thresh", 0.45),
                noop_thresh=thresh.get("noop_thresh", 0.99),
                model="hash64",
            )
        else:
            assert embedder is not None
            mgr = RuleV0(
                embedder=embedder,
                update_thresh=thresh["update_thresh"],
                noop_thresh=thresh["noop_thresh"],
            )
    elif name == "LLMv0":
        assert embedder is not None
        mgr = LLMv0(
            embedder=embedder,
            top_m=8,
            chat_model=chat_model,
            chat_provider=chat_provider,
            prompt=prompt,
        )
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
            if hasattr(mgr, "invalidate_embed_cache"):
                mgr.invalidate_embed_cache(
                    [l["fact_id"] for l in logs if l.get("fact_id")]
                )
            valid_ids = {f["id"] for f in store.read_all(valid_only=True)}
            inv = invalidation_flags(case, logs, valid_ids)
            row = {
                "case_id": case["id"],
                "type": case["type"],
                "split": case.get("split"),
                "manager": name,
                "gold_ops": case["gold_ops"],
                "pred_ops": [l["op"] for l in logs],
                "pred_ops_full": [
                    {"op": l["op"], "fact_id": l.get("fact_id")} for l in logs
                ],
                "op_correct": ops_match_lenient(case["gold_ops"], logs),
                "op_correct_strict": ops_match_strict(case["gold_ops"], logs),
                "raw_llm": getattr(dec, "raw_llm", None),
                **inv,
            }
            results.append(row)
            store.close()

    summary = aggregate_write(results)
    summary["manager"] = name
    if isinstance(mgr, LLMv0):
        summary["manager_stats"] = mgr.stats()
    return results, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "conflict_v1")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "conflict_v1_probe")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--managers", default="AlwaysADD,RuleV0")
    ap.add_argument("--hash-embed", action="store_true", help="offline RuleV0")
    ap.add_argument("--real-embed", action="store_true")
    ap.add_argument("--chat-model", default="gpt-4o-mini", help="LLMv0 backbone")
    ap.add_argument(
        "--chat-provider",
        default="openai",
        choices=["openai", "openrouter"],
        help="openai=OPENAI_API_KEY; openrouter=OPENROUTER_API_KEY (chat only)",
    )
    ap.add_argument("--prompt", default="neutral", choices=["neutral", "cond"])
    ap.add_argument("--types", default=None, help="comma list, e.g. condition,new_topic")
    ap.add_argument("--tag", default=None, help="override output filename tag")
    args = ap.parse_args()

    cases_path = args.data / "cases.jsonl"
    if not cases_path.exists():
        if args.data.name == "conflict_v2":
            from scripts.build_conflict_v2 import main as build
        else:
            from scripts.build_conflict_v1 import main as build
        build()
    cases = load_jsonl(cases_path)
    if args.split != "all":
        cases = [c for c in cases if c.get("split") == args.split]
    if args.types:
        want = {t.strip() for t in args.types.split(",") if t.strip()}
        cases = [c for c in cases if c.get("type") in want]

    # Prefer real-tuned thresh when scoring with real embeds
    if args.real_embed and (args.data / "rulev0_thresh_real.json").exists():
        thresh_path = args.data / "rulev0_thresh_real.json"
    else:
        thresh_path = args.data / "rulev0_thresh.json"
    thresh = json.loads(thresh_path.read_text()) if thresh_path.exists() else {
        "update_thresh": 0.88,
        "noop_thresh": 0.97,
    }
    print(f"thresh_file={thresh_path} thresh={thresh}", flush=True)
    print(
        f"chat_provider={args.chat_provider} chat_model={args.chat_model} "
        f"prompt={args.prompt}",
        flush=True,
    )

    use_hash = args.hash_embed or not args.real_embed
    embedder = None
    if args.real_embed or "LLMv0" in args.managers.split(","):
        cache_name = f"{args.data.name}.json"
        embedder = Embedder(
            cache_path=ROOT / "runs" / "_embed_cache" / cache_name
        )
        use_hash = bool(args.hash_embed)

    tag = args.tag or ("real" if args.real_embed else "hash")
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name in [m.strip() for m in args.managers.split(",") if m.strip()]:
        print("===", name, "n=", len(cases), "embed=", tag, flush=True)
        uh = True if (name == "RuleV0" and not args.real_embed) else False
        if name == "RuleV0" and args.real_embed:
            uh = False
        results, summary = run_manager(
            name, cases, embedder=embedder, thresh=thresh, use_hash=uh,
            chat_model=args.chat_model, chat_provider=args.chat_provider,
            prompt=args.prompt,
        )
        summary["embed_mode"] = "hash64" if uh else (
            embedder.model if embedder else "none"
        )
        summary["thresh"] = {
            "update_thresh": thresh.get("update_thresh"),
            "noop_thresh": thresh.get("noop_thresh"),
            "tuned_on": thresh.get("tuned_on"),
            "embed": thresh.get("embed"),
        }
        summaries[name] = summary
        (args.out / f"results_{name}_{args.split}_{tag}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in results) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)

    (args.out / f"summaries_{args.split}_{tag}.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
