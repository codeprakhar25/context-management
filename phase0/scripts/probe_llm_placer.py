#!/usr/bin/env python3
"""LLM path placer probe — path fidelity vs gold.

OpenAI chat only (default gpt-4o). No OpenRouter for this arm.
Scores exact + soft path fidelity (same root / prefix / LCP).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from harness.manager import make_chat_client  # noqa: E402
from harness.store import HierStore, Op, MAX_DEPTH, path_key  # noqa: E402

DEFAULT_TASKS = ROOT / "data" / "storage_oracle" / "place_tasks.jsonl"

PLACER_SYSTEM = """You are a memory path placer for a hard folder tree.

Rules:
- Fixed roots only: {roots}
- Max folder depth: {max_depth} (count folder segments; leaf text is not a segment)
- One fact lives under exactly one path
- Prefer an existing directory from existing_dirs when it fits
- Only mkdir a new path if nothing existing is appropriate
- If unsure, use ["inbox"] and low confidence
- Prefer short sensible paths (depth 2–4)

Return ONLY JSON (no markdown):
{{"path": ["root", "..."], "confidence": 0.0}}
"""


def dirs_from_store(store_path: Path | None, roots: list[str]) -> list[list[str]]:
    """Load existing dirs from a frozen HierStore (fairer placer / SFT context)."""
    if store_path is None or not store_path.exists():
        return [[r] for r in roots]
    with HierStore(store_path, roots=roots) as store:
        return store.list_dirs()


def load_tasks(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def parse_path_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    # first {...}
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise ValueError(f"no JSON object in: {raw[:200]}")
    obj = json.loads(m.group(0))
    path = obj.get("path")
    if not isinstance(path, list) or not all(isinstance(x, str) for x in path):
        raise ValueError(f"bad path: {path}")
    conf = obj.get("confidence", None)
    return {"path": path, "confidence": conf, "raw": raw}


def call_placer(
    client,
    *,
    model: str,
    roots: list[str],
    max_depth: int,
    text: str,
    kind: str,
    cue: str | None,
    from_path: list[str] | None,
    existing_dirs: list[list[str]],
) -> dict:
    sys_p = PLACER_SYSTEM.format(roots=roots, max_depth=max_depth)
    user = {
        "task": kind,
        "fact_text": text,
        "existing_dirs": existing_dirs,
        "from_path": from_path,
        "cue": cue,
    }
    messages = [
        {"role": "system", "content": sys_p},
        {
            "role": "user",
            "content": json.dumps(user, indent=2)
            + "\n\nChoose the destination path. JSON only.",
        },
    ]
    # Retry transient server/rate-limit errors. Without this a parallel run on a
    # large prompt silently turns into a partial, timing-selected sample: the
    # first flat+descriptions pass lost 185/294 items to 429s at 15k tokens/call.
    last_err: Exception | None = None
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.0, messages=messages
            )
            break
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            transient = any(
                s in msg for s in ("429", "rate_limit", "500", "502", "503", "504",
                                   "overloaded", "timeout", "Timeout", "Connection")
            )
            if not transient or attempt == 5:
                raise
            last_err = e
            time.sleep(min(2 ** attempt, 30) + random.random() * 2)
    else:  # pragma: no cover - loop always breaks or raises
        raise last_err  # type: ignore[misc]
    raw = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    parsed = parse_path_json(raw)
    parsed["usage"] = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }
    return parsed


def prep_store(roots: list[str], from_path: list[str] | None, text: str, fact_id: str) -> HierStore:
    td = tempfile.mkdtemp(prefix="placer_")
    store = HierStore(Path(td) / "t.sqlite", roots=roots, strict_dirs=False)
    # keep tmp path on store for cleanup optional
    store._tmp_dir = td  # type: ignore[attr-defined]
    if from_path:
        store.mkdir(from_path)
        store.apply_ops(
            [Op(op="ADD", fact_id=fact_id, text=text, path=from_path)],
            manager="setup",
        )
    return store


def lcp_len(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def soft_relation(gold: list[str], pred: list[str]) -> str:
    """exact | gold_prefix | pred_prefix | same_root | diff_root | empty."""
    if not pred:
        return "empty"
    if gold == pred:
        return "exact"
    if len(gold) <= len(pred) and pred[: len(gold)] == gold:
        return "gold_prefix"  # pred deeper under gold
    if len(pred) <= len(gold) and gold[: len(pred)] == pred:
        return "pred_prefix"  # pred shallower ancestor of gold
    if gold[0] == pred[0]:
        return "same_root"
    return "diff_root"


def score_one(
    gold: list[str],
    pred_path: list[str],
    *,
    roots: list[str],
    max_depth: int = MAX_DEPTH,
) -> dict:
    exact = pred_path == gold
    root_ok = bool(pred_path) and pred_path[0] in roots
    depth_ok = 0 < len(pred_path) <= max_depth
    same_root = bool(pred_path) and bool(gold) and pred_path[0] == gold[0]
    rel = soft_relation(gold, pred_path)
    # soft_hit: exact OR either is prefix of other (same branch)
    soft_hit = rel in ("exact", "gold_prefix", "pred_prefix")
    # branch_ok: at least same root (life-area correct)
    branch_ok = same_root
    lcp = lcp_len(gold, pred_path)
    denom = max(len(gold), len(pred_path), 1)
    return {
        "exact": exact,
        "soft_hit": soft_hit,
        "branch_ok": branch_ok,
        "same_root": same_root,
        "soft_relation": rel,
        "lcp": lcp,
        "lcp_ratio": lcp / denom,
        "root_ok": root_ok,
        "depth_ok": depth_ok,
        "gold_path": gold,
        "pred_path": pred_path,
        "pred_key": path_key(pred_path) if pred_path else "",
        "gold_key": path_key(gold),
    }


def summarize(rows: list[dict], *, model: str, provider: str, pt: int, ct: int) -> dict:
    n = len(rows)
    def rate(key: str) -> float:
        return sum(1 for r in rows if r.get(key)) / n if n else 0.0

    cost = (pt * 2.50 + ct * 10.00) / 1e6
    summary = {
        "n": n,
        "path_exact": rate("exact"),
        "n_exact": sum(1 for r in rows if r.get("exact")),
        "path_soft": rate("soft_hit"),
        "n_soft": sum(1 for r in rows if r.get("soft_hit")),
        "branch_ok": rate("branch_ok"),
        "mean_lcp_ratio": round(
            sum(r.get("lcp_ratio") or 0 for r in rows) / n, 4
        )
        if n
        else 0.0,
        "root_ok": rate("root_ok"),
        "depth_ok": rate("depth_ok"),
        "apply_ok": sum(1 for r in rows if not r.get("apply_err")) / n if n else 0.0,
        "soft_relation_counts": {},
        "model": model,
        "provider": provider,
        "chat_prompt_tokens": pt,
        "chat_completion_tokens": ct,
        "chat_cost_usd_est": round(cost, 5),
        "lead_metric": "path_soft (exact|prefix); branch_ok=same root; path_exact diagnostic",
        "by_kind": {},
    }
    from collections import Counter

    summary["soft_relation_counts"] = dict(
        Counter(r.get("soft_relation") for r in rows)
    )
    for kind in sorted({r["kind"] for r in rows}):
        sub = [r for r in rows if r["kind"] == kind]
        summary["by_kind"][kind] = {
            "n": len(sub),
            "path_exact": sum(1 for r in sub if r.get("exact")) / len(sub),
            "path_soft": sum(1 for r in sub if r.get("soft_hit")) / len(sub),
            "branch_ok": sum(1 for r in sub if r.get("branch_ok")) / len(sub),
        }
    return summary


def rescore_dir(out: Path) -> dict:
    """Recompute soft metrics on existing results.jsonl (no API)."""
    rows_path = out / "results.jsonl"
    old_sum = {}
    if (out / "summary.json").exists():
        old_sum = json.loads((out / "summary.json").read_text())
    rows_in = [
        json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()
    ]
    rows = []
    for r in rows_in:
        roots = r.get("roots") or ["work", "personal", "inbox"]
        # gold/pred already on row
        sc = score_one(
            r["gold_path"],
            r["pred_path"],
            roots=roots,
            max_depth=MAX_DEPTH,
        )
        nr = {**r, **sc}
        rows.append(nr)
    summary = summarize(
        rows,
        model=old_sum.get("model", "?"),
        provider=old_sum.get("provider", "?"),
        pt=int(old_sum.get("chat_prompt_tokens") or 0),
        ct=int(old_sum.get("chat_completion_tokens") or 0),
    )
    (out / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--provider", default="openai", choices=["openai"])
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "llm_placer")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--store",
        type=Path,
        default=None,
        help="HierStore sqlite — pass its dirs as existing_dirs (fairer smoke)",
    )
    ap.add_argument(
        "--rescore-only",
        action="store_true",
        help="Recompute soft metrics on --out/results.jsonl; no API calls",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.rescore_only:
        summary = rescore_dir(args.out)
        print(json.dumps(summary, indent=2), flush=True)
        print("rescored", args.out)
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing")

    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]

    # shared vocabulary from frozen snap (all tasks same roots usually)
    default_roots = tasks[0].get("roots") if tasks else ["work", "personal", "inbox"]
    shared_dirs = dirs_from_store(args.store, default_roots or ["work", "personal", "inbox"])
    if args.store:
        print(
            f"existing_dirs from {args.store}: n={len(shared_dirs)}",
            flush=True,
        )

    client = make_chat_client(args.provider)
    rows = []
    pt = ct = 0

    for task in tasks:
        roots = task.get("roots") or ["work", "personal", "inbox"]
        kind = task.get("kind", "place")
        fact_id = task.get("fact_id") or f"f_{task['id']}"
        store = prep_store(roots, task.get("from_path"), task["text"], fact_id)
        existing = shared_dirs if args.store else store.list_dirs()
        try:
            pred = call_placer(
                client,
                model=args.model,
                roots=roots,
                max_depth=MAX_DEPTH,
                text=task["text"],
                kind=kind,
                cue=task.get("cue"),
                from_path=task.get("from_path"),
                existing_dirs=existing,
            )
            pt += pred["usage"]["prompt_tokens"]
            ct += pred["usage"]["completion_tokens"]
            pred_path = pred["path"]
            apply_err = None
            try:
                if kind == "move":
                    store.mkdir(pred_path)
                    store.apply_ops(
                        [Op(op="MOVE", fact_id=fact_id, path=pred_path)],
                        manager="LLMplacer",
                    )
                else:
                    store.apply_ops(
                        [
                            Op(op="MKDIR", path=pred_path),
                            Op(
                                op="ADD",
                                fact_id=fact_id,
                                text=task["text"],
                                path=pred_path,
                                confidence=pred.get("confidence"),
                            ),
                        ],
                        manager="LLMplacer",
                    )
            except Exception as e:
                apply_err = f"{type(e).__name__}: {e}"

            sc = score_one(
                task["gold_path"], pred_path, roots=roots, max_depth=store.max_depth
            )
            row = {
                "id": task["id"],
                "kind": kind,
                "text": task["text"],
                "cue": task.get("cue"),
                "confidence": pred.get("confidence"),
                "apply_err": apply_err,
                "raw": pred.get("raw"),
                "roots": roots,
                **sc,
            }
        finally:
            store.close()

        rows.append(row)
        mark = "OK" if row.get("exact") else ("SOFT" if row.get("soft_hit") else "MISS")
        print(
            f"{mark} {task['id']} gold={path_key(task['gold_path'])} "
            f"pred={path_key(row.get('pred_path') or [])} rel={row['soft_relation']}",
            flush=True,
        )

    summary = summarize(
        rows, model=args.model, provider=args.provider, pt=pt, ct=ct
    )
    summary["store"] = str(args.store) if args.store else None
    summary["n_existing_dirs"] = len(shared_dirs) if args.store else None
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

