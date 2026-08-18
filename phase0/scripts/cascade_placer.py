#!/usr/bin/env python3
"""Cascade placer: retrieve a folder shortlist, then let the model pick from it.

Everything about the call is held identical to the flat baselines — same system
prompt, same user object, same parser, same scorer — so the ONLY variable is
what goes in `existing_dirs`: a retrieved shortlist instead of all 365 folders.

Two candidate sources, because they fail in opposite places (see
scripts/candidate_recall.py):
  note  folders of the nearest training notes. recall@20 = 0.983 on folders with
        training examples, exactly 0.000 on folders without.
  path  folders whose path string embeds nearest to the note. recall@20 = 0.677
        on unseen folders — the only source that reaches them at all.

The shortlist is also 26x smaller than the full list (236 vs 6251 tokens), which
is what makes this the only design that survives a tree of a few thousand folders.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import importlib.util  # noqa: E402

from harness.embed import Embedder  # noqa: E402
from harness.store import max_depth_from_store  # noqa: E402
from openai import OpenAI  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "probe_llm_placer", ROOT / "scripts" / "probe_llm_placer.py"
)
probe = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(probe)

_ev = importlib.util.spec_from_file_location(
    "eval_fireworks_placer", ROOT / "scripts" / "eval_fireworks_placer.py"
)
evmod = importlib.util.module_from_spec(_ev)
assert _ev and _ev.loader
_ev.loader.exec_module(evmod)
soft_score = evmod.soft_score


def path_text(path: list[str]) -> str:
    return f"folder: {' / '.join(path)} (topic: {path[-1].replace('-', ' ').replace('_', ' ')})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--mode", default="note", choices=["note", "path", "union"])
    ap.add_argument("--n", type=int, default=20, help="shortlist size")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--provider", default="openai", choices=["openai", "openrouter"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--descriptions",
        type=Path,
        default=None,
        help="folder_descriptions.json — renders each candidate as 'path — what "
        "belongs here' instead of a bare path. Only valid when the descriptions "
        "were built from files the eval never sees; see gen_folder_descriptions.py",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--max-depth",
        type=int,
        default=0,
        help="0 = read MAX_DEPTH from the store (corpus B is 8, corpus A is 5)",
    )
    args = ap.parse_args()

    load = lambda p: [json.loads(l) for l in p.read_text().splitlines() if l.strip()]  # noqa: E731
    train, val = load(args.train), load(args.val)
    if args.limit:
        val = val[: args.limit]
    con = sqlite3.connect(args.store)
    all_dirs = [json.loads(r[0]) for r in con.execute("select path_json from dirs")]
    max_depth = args.max_depth or max_depth_from_store(args.store)

    key = lambda p: "/" + "/".join(p)  # noqa: E731
    unkey = lambda k: [x for x in k.split("/") if x]  # noqa: E731

    emb = Embedder(cache_path=ROOT / "runs" / "_embed_cache" / "candidate_recall.json")

    def norm(x):
        a = np.array(x, dtype=np.float32)
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)

    vv = norm(emb.embed_texts([v["text"] for v in val]))
    tv = norm(emb.embed_texts([t["text"] for t in train]))
    dv = norm(emb.embed_texts([path_text(d) for d in all_dirs]))
    train_dirs = [key(t["gold_path"]) for t in train]
    dir_keys = [key(d) for d in all_dirs]
    note_order = np.argsort(-(vv @ tv.T), axis=1)
    path_order = np.argsort(-(vv @ dv.T), axis=1)

    def dedup(seq):
        seen, out = set(), []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def cands(i: int) -> list[str]:
        n = args.n
        note_c = dedup(train_dirs[j] for j in note_order[i][: n * 8])
        path_c = dedup(dir_keys[j] for j in path_order[i][: n * 2])
        if args.mode == "note":
            return note_c[:n]
        if args.mode == "path":
            return path_c[:n]
        out: list[str] = []
        for a, b in zip(note_c + [None] * n, path_c + [None] * n):
            for x in (a, b):
                if x is not None and x not in out:
                    out.append(x)
            if len(out) >= n:
                break
        return out[:n]

    if args.provider == "openrouter":
        client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    rows: list[dict] = [None] * len(val)  # type: ignore[list-item]
    tok = [0, 0]
    lock = threading.Lock()

    descs: dict[str, str] = {}
    if args.descriptions:
        descs = json.loads(args.descriptions.read_text())["descriptions"]

    def run_one(i: int) -> None:
        v = val[i]
        ckeys = cands(i)
        shortlist: list = [unkey(k) for k in ckeys]
        if descs:
            # keep the path as the first element so the answer format is unchanged
            shortlist = [
                {"path": unkey(k), "contains": descs[k]} if descs.get(k) else {"path": unkey(k)}
                for k in ckeys
            ]
        in_list = key(v["gold_path"]) in set(ckeys)
        try:
            pred = probe.call_placer(
                client,
                model=args.model,
                roots=v["roots"],
                max_depth=max_depth,
                text=v["text"],
                kind="place",
                cue=None,
                from_path=None,
                existing_dirs=shortlist,
            )
            path, raw = pred["path"], pred.get("raw", "")
            with lock:
                tok[0] += pred["usage"]["prompt_tokens"]
                tok[1] += pred["usage"]["completion_tokens"]
        except Exception as e:  # noqa: BLE001
            path, raw = None, str(e)
        sc = soft_score(v["gold_path"], path) if path else {
            "exact": False, "soft_hit": False, "branch_ok": False, "soft_relation": "empty"
        }
        rows[i] = {
            "id": v["id"],
            "gold_path": v["gold_path"],
            "pred_path": path,
            "gold_in_shortlist": in_list,
            "n_candidates": len(shortlist),
            "raw": raw[:300] if isinstance(raw, str) else raw,
            **sc,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_one, range(len(val))))
    rows = [r for r in rows if r is not None]

    n = len(rows) or 1
    in_short = [r for r in rows if r["gold_in_shortlist"]]
    summary = {
        "mode": args.mode,
        "n_candidates": args.n,
        "model": args.model,
        "provider": args.provider,
        "max_depth": max_depth,
        "val": str(args.val),
        "n": len(rows),
        "candidate_recall": round(len(in_short) / n, 4),
        "path_exact": round(sum(r["exact"] for r in rows) / n, 4),
        "path_soft": round(sum(r["soft_hit"] for r in rows) / n, 4),
        "branch_ok": round(sum(r["branch_ok"] for r in rows) / n, 4),
        # how well it picks GIVEN the answer was on the menu — separates the
        # retrieval ceiling from the model's ability to choose
        "exact_when_gold_in_shortlist": round(
            sum(r["exact"] for r in in_short) / max(len(in_short), 1), 4
        ),
        "n_parse_fail": sum(1 for r in rows if r["pred_path"] is None),
        "chat_prompt_tokens": tok[0],
        "chat_completion_tokens": tok[1],
        "mean_prompt_tokens": round(tok[0] / n, 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "holdout_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
