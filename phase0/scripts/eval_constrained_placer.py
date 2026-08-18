#!/usr/bin/env python3
"""B: constrain placer to existing_dirs (prompt + hard project).

Modes:
  prompt  — system says MUST pick exact member of existing_dirs
  project — after decode, snap invalid path to nearest existing dir (LCP)

Reports unconstrained-style metrics on projected paths; also invalid_rate
before project.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from harness.store import path_key  # noqa: E402

CONSTRAINED_SYSTEM = """You are a memory path placer for a hard folder tree.

Rules:
- Fixed roots only: {roots}
- Max folder depth: {max_depth}
- You MUST choose path as an EXACT entry from existing_dirs (copy the array)
- Do NOT invent new folders or mkdir
- If unsure, pick the best-fitting existing_dirs entry (often an inbox/dump/misc root)

Return ONLY JSON (no markdown):
{{"path": ["root", "..."], "confidence": 0.0}}
"""


def soft_score(gold: list[str], pred: list[str] | None) -> dict:
    if not pred:
        return {
            "exact": False,
            "soft_hit": False,
            "branch_ok": False,
            "soft_relation": "empty",
        }
    exact = gold == pred
    same_root = bool(gold) and bool(pred) and gold[0] == pred[0]
    if exact:
        rel = "exact"
    elif len(gold) <= len(pred) and pred[: len(gold)] == gold:
        rel = "gold_prefix"
    elif len(pred) <= len(gold) and gold[: len(pred)] == pred:
        rel = "pred_prefix"
    elif same_root:
        rel = "same_root"
    else:
        rel = "diff_root"
    return {
        "exact": exact,
        "soft_hit": rel in ("exact", "gold_prefix", "pred_prefix"),
        "branch_ok": same_root,
        "soft_relation": rel,
    }


def lcp_len(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def project_to_existing(pred: list[str] | None, existing: list[list[str]]) -> list[str]:
    if not existing:
        return pred or []
    keys = {path_key(d): d for d in existing}
    if pred and path_key(pred) in keys:
        return list(pred)
    # maximize LCP; then minimize |len(pred)-len(cand)|; then shorter; then lex
    best = None
    best_key = None
    for d in existing:
        lc = lcp_len(pred or [], d)
        gap = abs(len(pred or []) - len(d))
        key = (-lc, gap, len(d), path_key(d))
        if best_key is None or key < best_key:
            best_key = key
            best = d
    return list(best)  # type: ignore[arg-type]


def load_probe():
    spec = importlib.util.spec_from_file_location(
        "probe_llm_placer", ROOT / "scripts" / "probe_llm_placer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def call_constrained(client, probe, *, model, roots, max_depth, text, existing_dirs):
    sys_p = CONSTRAINED_SYSTEM.format(roots=roots, max_depth=max_depth)
    user = {
        "task": "place",
        "fact_text": text,
        "existing_dirs": existing_dirs,
        "from_path": None,
        "cue": None,
        "constraint": "path MUST be an exact element of existing_dirs",
    }
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": sys_p},
            {
                "role": "user",
                "content": json.dumps(user, indent=2)
                + "\n\nChoose destination path from existing_dirs only. JSON only.",
            },
        ],
    )
    raw = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    parsed = probe.parse_path_json(raw)
    parsed["usage"] = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }
    parsed["raw"] = raw
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "data" / "user_dir_snap" / "place_tasks_from_snap.jsonl",
    )
    ap.add_argument(
        "--store",
        type=Path,
        default=ROOT / "data" / "user_dir_snap" / "hierstore.sqlite",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--provider",
        choices=["openai", "fireworks"],
        default="openai",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--project",
        action="store_true",
        help="hard-snap invalid preds onto nearest existing_dirs",
    )
    args = ap.parse_args()

    probe = load_probe()
    tasks = [json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()]
    if args.limit:
        tasks = tasks[: args.limit]
    roots0 = tasks[0].get("roots") or ["work", "personal", "inbox"]
    shared = probe.dirs_from_store(args.store, roots0)
    exist_set = {path_key(d) for d in shared}
    print(f"existing_dirs n={len(shared)} project={args.project}", flush=True)

    if args.provider == "fireworks":
        key = os.environ.get("FIREWORKS_API_KEY") or os.environ.get("FIREWORKS_API")
        if not key:
            raise SystemExit("missing FIREWORKS_API")
        client = OpenAI(api_key=key, base_url="https://api.fireworks.ai/inference/v1")
    else:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("missing OPENAI_API_KEY")
        client = OpenAI(api_key=key, base_url="https://api.openai.com/v1")

    rows = []
    pt = ct = 0
    invalid = 0
    parse_fail = 0
    for t in tasks:
        roots = t.get("roots") or roots0
        try:
            pred = call_constrained(
                client,
                probe,
                model=args.model,
                roots=roots,
                max_depth=5,
                text=t["text"],
                existing_dirs=shared,
            )
            raw_path = pred["path"]
            pt += pred["usage"]["prompt_tokens"]
            ct += pred["usage"]["completion_tokens"]
        except Exception as e:  # noqa: BLE001
            raw_path = None
            parse_fail += 1
            pred = {"raw": str(e)}
            print(f"FAIL {t['id']} {e}", flush=True)

        in_set = bool(raw_path and path_key(raw_path) in exist_set)
        if not in_set:
            invalid += 1
        final = (
            project_to_existing(raw_path, shared)
            if args.project
            else (list(raw_path) if raw_path else None)
        )
        sc = soft_score(t["gold_path"], final)
        rows.append(
            {
                "id": t["id"],
                "gold_path": t["gold_path"],
                "pred_raw": raw_path,
                "pred_path": final,
                "in_existing_dirs": in_set,
                **sc,
            }
        )
        mark = "OK" if sc["exact"] else ("SOFT" if sc["soft_hit"] else "MISS")
        print(
            f"{mark} {'IN' if in_set else 'OUT'} {t['id']} "
            f"gold={path_key(t['gold_path'])} pred={path_key(final) if final else None}",
            flush=True,
        )

    n = len(rows) or 1
    summary = {
        "n": len(rows),
        "model": args.model,
        "provider": args.provider,
        "constrain": "prompt+project" if args.project else "prompt",
        "n_existing_dirs": len(shared),
        "invalid_rate": invalid / n,
        "n_invalid": invalid,
        "parse_fail": parse_fail,
        "path_exact": sum(r["exact"] for r in rows) / n,
        "path_soft": sum(r["soft_hit"] for r in rows) / n,
        "branch_ok": sum(r["branch_ok"] for r in rows) / n,
        "chat_prompt_tokens": pt,
        "chat_completion_tokens": ct,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
