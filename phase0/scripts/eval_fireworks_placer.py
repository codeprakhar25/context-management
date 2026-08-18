#!/usr/bin/env python3
"""Holdout path eval for Fireworks fine-tuned placer LoRA."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

import importlib.util

from harness.store import max_depth_from_store, path_key  # noqa: E402


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "probe_llm_placer", ROOT / "scripts" / "probe_llm_placer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "data" / "multitree_synth_smoke" / "place_holdout.jsonl",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "fireworks_placer_smoke")
    ap.add_argument(
        "--model",
        default=(
            "accounts/prakharkhatri123-edp/models/placer-smoke-llama31-8b"
            "#accounts/prakharkhatri123-edp/deployments/placer-smoke-llama31-8b-live"
        ),
        help="Fireworks model id; live-merge needs model#deployment or deployment id",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--provider",
        default="fireworks",
        choices=["fireworks", "openrouter", "openai"],
        help="openrouter serves the untuned base weights; Fireworks dropped "
        "llama-3.1-8b from serverless and the LoRA deployments are stopped. "
        "openai is here so every arm goes through one prompt/parser/scorer",
    )
    ap.add_argument(
        "--store",
        type=Path,
        default=None,
        help="HierStore sqlite — shared existing_dirs for all tasks (user-dir transfer)",
    )
    ap.add_argument(
        "--descriptions",
        type=Path,
        default=None,
        help="folder_descriptions.json — renders each folder as 'path + what "
        "belongs here'. Combined with the flat (no-retrieval) prompt this is the "
        "grounding-only cell of the retrieval x grounding 2x2; see PLACER_FINDINGS. "
        "Only valid where the descriptions were built from files the eval never "
        "sees (item split only) — see gen_folder_descriptions.py",
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--max-depth",
        type=int,
        default=0,
        help="0 = read MAX_DEPTH from --store, else max gold-path length",
    )
    args = ap.parse_args()

    if args.provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API")
        base_url = "https://openrouter.ai/api/v1"
    elif args.provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        base_url = None
    else:
        key = os.environ.get("FIREWORKS_API_KEY") or os.environ.get("FIREWORKS_API")
        base_url = "https://api.fireworks.ai/inference/v1"
    if not key:
        raise SystemExit(f"missing API key for provider={args.provider}")

    client = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
    probe = _load_probe()
    tasks = [json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()]
    if args.limit:
        tasks = tasks[: args.limit]

    shared_dirs = None
    if args.store:
        shared_dirs = probe.dirs_from_store(
            args.store, tasks[0].get("roots") or ["work", "personal", "inbox"]
        )
        print(f"existing_dirs from {args.store}: n={len(shared_dirs)}", flush=True)

    if args.max_depth:
        max_depth = args.max_depth
    elif args.store:
        max_depth = max_depth_from_store(args.store)
    else:
        max_depth = max((len(t["gold_path"]) for t in tasks), default=5)

    if args.descriptions:
        descs = json.loads(args.descriptions.read_text())["descriptions"]
        # keep the path first so the expected answer format is unchanged; folders
        # with no training files simply carry no description
        def _decorate(dirs):
            out = []
            for d in dirs:
                k = "/" + "/".join(d)
                out.append({"path": d, "contains": descs[k]} if descs.get(k) else {"path": d})
            return out

        if shared_dirs is not None:
            shared_dirs = _decorate(shared_dirs)
        n_desc = sum(1 for d in (shared_dirs or []) if isinstance(d, dict) and "contains" in d)
        print(f"descriptions attached to {n_desc}/{len(shared_dirs or [])} folders", flush=True)

    rows: list = [None] * len(tasks)
    usage = [0, 0, 0]  # prompt tokens, completion tokens, parse failures
    lock = threading.Lock()

    def run_one(i: int) -> None:
        t = tasks[i]
        roots = t["roots"]
        if shared_dirs is not None:
            existing = shared_dirs
        else:
            existing = t.get("existing_dirs") or [[r] for r in roots]
            if args.descriptions:
                existing = _decorate(existing)
        try:
            pred = probe.call_placer(
                client,
                model=args.model,
                roots=roots,
                max_depth=max_depth,
                text=t["text"],
                kind="place",
                cue=None,
                from_path=None,
                existing_dirs=existing,
            )
            path = pred["path"]
            raw = pred.get("raw", "")
            with lock:
                usage[0] += pred["usage"]["prompt_tokens"]
                usage[1] += pred["usage"]["completion_tokens"]
        except Exception as e:  # noqa: BLE001
            path = None
            raw = str(e)
            with lock:
                usage[2] += 1
            print(f"FAIL {t['id']} {e}", flush=True)

        sc = soft_score(t["gold_path"], path)
        rows[i] = {
            "id": t["id"],
            "tree_id": t.get("tree_id"),
            "gold_path": t["gold_path"],
            "pred_path": path,
            "raw": raw[:500] if isinstance(raw, str) else raw,
            **sc,
        }
        mark = "OK" if sc["exact"] else ("SOFT" if sc["soft_hit"] else "MISS")
        print(
            f"{mark} {t['id']} gold={path_key(t['gold_path'])} "
            f"pred={path_key(path) if path else None}",
            flush=True,
        )

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run_one, range(len(tasks))))
    else:
        for i in range(len(tasks)):
            run_one(i)
    rows = [r for r in rows if r is not None]
    pt, ct, parse_fail = usage

    n = len(rows) or 1
    summary = {
        "n": len(rows),
        "model": args.model,
        "provider": args.provider,
        "tasks": str(args.tasks),
        "max_depth": max_depth,
        "path_exact": sum(r["exact"] for r in rows) / n,
        "path_soft": sum(r["soft_hit"] for r in rows) / n,
        "branch_ok": sum(r["branch_ok"] for r in rows) / n,
        "parse_fail": parse_fail,
        "descriptions": str(args.descriptions) if args.descriptions else None,
        "chat_prompt_tokens": pt,
        "chat_completion_tokens": ct,
        "mean_prompt_tokens": round(pt / n, 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "holdout_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", args.out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
