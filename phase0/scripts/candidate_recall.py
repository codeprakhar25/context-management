#!/usr/bin/env python3
"""Recall gate for a cascade placer: is the gold folder even in the shortlist?

A cascade (retrieve candidate folders -> model picks one) can never beat the
recall of its candidate generator. Measure that ceiling before spending anything
on the picking half.

Three ways to propose candidates:

  note     folders of the nearest TRAINING NOTES. This is what the kNN baseline
           does. Structurally cannot reach a folder holding no training note,
           which is why kNN scores exactly 0.000 on the folder-disjoint split.
  path     folders whose PATH STRING embeds nearest to the note. Reaches any
           folder in the tree, examples or not — the only mode with a chance on
           unseen folders.
  union    interleave both, note-side first.

Reports recall@N and the prompt-size win: a shortlist of N paths versus the full
`existing_dirs` block every arm currently carries.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
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

from harness.embed import Embedder  # noqa: E402


def path_text(path: list[str]) -> str:
    """Folder rendered for embedding. Segments carry most of the signal, so the
    leaf is repeated — `.../gateway` is more about gateway than about work."""
    return f"folder: {' / '.join(path)} (topic: {path[-1].replace('-', ' ').replace('_', ' ')})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[5, 10, 20, 30, 50])
    args = ap.parse_args()

    load = lambda p: [json.loads(l) for l in p.read_text().splitlines() if l.strip()]  # noqa: E731
    train, val = load(args.train), load(args.val)
    con = sqlite3.connect(args.store)
    all_dirs = [json.loads(r[0]) for r in con.execute("select path_json from dirs")]

    key = lambda p: "/" + "/".join(p)  # noqa: E731
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

    def cands(i: int, mode: str, n: int) -> list[str]:
        note_c = dedup(train_dirs[j] for j in note_order[i][: n * 8])
        path_c = dedup(dir_keys[j] for j in path_order[i][: n * 2])
        if mode == "note":
            return note_c[:n]
        if mode == "path":
            return path_c[:n]
        # Interleave, so neither side can starve the other. Filling note-first
        # would hand the union note's structural blindness to unseen folders.
        out: list[str] = []
        for a, b in zip(note_c + [None] * n, path_c + [None] * n):
            for x in (a, b):
                if x is not None and x not in out:
                    out.append(x)
            if len(out) >= n:
                break
        return out[:n]

    report: dict = {"n_val": len(val), "n_folders_in_tree": len(all_dirs), "recall": {}}
    for mode in ("note", "path", "union"):
        report["recall"][mode] = {}
        for n in args.n:
            hit = sum(1 for i, v in enumerate(val) if key(v["gold_path"]) in cands(i, mode, n))
            report["recall"][mode][str(n)] = round(hit / len(val), 4)

    full_tok = len(json.dumps(all_dirs, indent=2)) // 4
    report["prompt_tokens_full_dir_list"] = full_tok
    report["prompt_tokens_shortlist"] = {
        str(n): len(json.dumps([d for d in dir_keys[:n]], indent=2)) // 4 for n in args.n
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
