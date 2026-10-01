#!/usr/bin/env python3
"""Folder-description retrieval on the folder-disjoint split, no picker.

Reviewer question: on folders that have never been used, is the cascade's LLM
step needed, or would retrieving against a description of each folder do as
well? This is that baseline. Each folder gets one line describing what belongs
in it, the note is embedded, and the nearest description is the prediction.

LEAK BOUNDARY. On the folder-disjoint split a held-out folder contains only
validation notes, so any description built from its contents would encode the
answer. Descriptions here are generated from the PATH STRING ALONE, for every
folder in the tree, seen or unseen, so all candidates are described the same
way. (gen_folder_descriptions.py builds from training notes and must not be
used on this split.)

Outputs per-vault holdout_results.jsonl in the same shape as the other arms,
plus a pooled summary. Descriptions are cached per vault, so a rerun only pays
for embeddings it has not seen.
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
sys.path.insert(0, str(ROOT / "scripts"))

env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from openai import OpenAI  # noqa: E402

from harness.embed import Embedder  # noqa: E402
from knn_placer_baseline import soft_score  # noqa: E402
from vault_level_robustness import CACHE, CachedEmbeddings  # noqa: E402

SYSTEM = """You write one-line descriptions of folders in a note store.

You are given only the folder's path. Write ONE line (max 15 words) saying what
kind of note most plausibly belongs in this folder, judging from the folder and
parent names. No preamble, no quotes, just the line."""


def describe(client: OpenAI, folders: list[list[str]], model: str, workers: int) -> tuple[dict, list[int]]:
    out: dict[str, str] = {}
    tok = [0, 0]
    lock = threading.Lock()

    def one(p: list[str]) -> None:
        k = "/" + "/".join(p)
        try:
            r = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": f"Folder: {k}"}],
            )
            desc = (r.choices[0].message.content or "").strip().strip('"')
            with lock:
                tok[0] += r.usage.prompt_tokens
                tok[1] += r.usage.completion_tokens
        except Exception as e:  # noqa: BLE001
            desc = ""
            print(f"FAIL {k}: {e}", flush=True)
        out[k] = desc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, folders))
    return out, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-vaults", type=int, default=0, help="smoke test")
    args = ap.parse_args()

    load = lambda p: [json.loads(l) for l in p.read_text().splitlines() if l.strip()]  # noqa: E731
    snaps = sorted(d for d in args.build.iterdir() if d.is_dir() and (d / "fold_val.jsonl").exists())
    if args.limit_vaults:
        snaps = snaps[: args.limit_vaults]

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    note_emb = CachedEmbeddings(CACHE)  # val notes are already embedded; never re-paid
    desc_emb = Embedder(cache_path=ROOT / "runs" / "_embed_cache" / "folder_desc_pathonly.json")
    args.out.mkdir(parents=True, exist_ok=True)
    tok_all = [0, 0]
    pooled = {"n": 0, "exact": 0, "soft": 0, "n_empty_desc": 0}

    for snap in snaps:
        vdir = args.out / snap.name
        vdir.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(snap / "hierstore.sqlite")
        dirs = [json.loads(r[0]) for r in con.execute("select path_json from dirs")]
        con.close()

        dfile = vdir / "descriptions.json"
        if dfile.exists():
            descs = json.loads(dfile.read_text())["descriptions"]
        else:
            descs, tok = describe(client, dirs, args.model, args.workers)
            tok_all[0] += tok[0]
            tok_all[1] += tok[1]
            dfile.write_text(json.dumps({"model": args.model, "source": "path string only",
                                         "descriptions": descs}, indent=2) + "\n")

        keys = ["/" + "/".join(p) for p in dirs]
        # an empty description falls back to the path itself rather than an empty string
        texts = [f"{k}: {descs.get(k) or k}" for k in keys]
        dv = np.asarray(desc_emb.embed_texts(texts), dtype=np.float32)
        dv /= np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9

        val = load(snap / "fold_val.jsonl")
        vv = note_emb([v["text"] for v in val])
        top1 = np.argmax(vv @ dv.T, axis=1)
        rows = []
        for i, v in enumerate(val):
            pred = dirs[top1[i]]
            rows.append({"id": v["id"], "gold_path": v["gold_path"], "pred_path": pred,
                         **soft_score(v["gold_path"], pred)})
        (vdir / "holdout_results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        n = len(rows)
        summ = {"n": n, "path_exact": round(sum(r["exact"] for r in rows) / n, 4),
                "path_soft": round(sum(r["soft_hit"] for r in rows) / n, 4),
                "n_folders": len(dirs), "n_empty_desc": sum(1 for k in keys if not descs.get(k))}
        (vdir / "summary.json").write_text(json.dumps(summ, indent=2) + "\n")
        pooled["n"] += n
        pooled["exact"] += sum(r["exact"] for r in rows)
        pooled["soft"] += sum(r["soft_hit"] for r in rows)
        pooled["n_empty_desc"] += summ["n_empty_desc"]
        print(f"  {snap.name:<44} n={n:<4} exact={summ['path_exact']:.3f}", flush=True)

    cost = tok_all[0] / 1e6 * 0.15 + tok_all[1] / 1e6 * 0.60 + desc_emb.cost_usd()
    out = {"build": str(args.build), "model": args.model, "n_vaults": len(snaps), "n_val": pooled["n"],
           "path_exact": round(pooled["exact"] / max(pooled["n"], 1), 4),
           "path_soft": round(pooled["soft"] / max(pooled["n"], 1), 4),
           "n_empty_desc": pooled["n_empty_desc"], "cost_usd_est": round(cost, 4)}
    (args.out / "pooled.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
