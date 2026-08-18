#!/usr/bin/env python3
"""One-line description per folder, generated from the TRAINING files it holds.

The cascade's bottleneck is no longer retrieval — on the item split the gold
folder is in the top-20 shortlist 98.3% of the time, yet only 65% get picked.
The model sees 20 bare paths like `work/openclaw/docs/gateway` and has to guess
what belongs there. A one-line summary is the missing signal.

LEAK BOUNDARY — read before reusing this. Descriptions are built ONLY from files
in `--train`. On an item-split (stratified within folder) every val folder also
holds training files, so this is clean. On a folder-disjoint split the held-out
folders contain *nothing but* val files, so any content-derived description would
encode the answer; there, descriptions must come from the path alone or from an
external source, and this script must not be used.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from openai import OpenAI  # noqa: E402

SYSTEM = """You write one-line descriptions of folders in a note store.

Given the folder path and excerpts from notes filed there, write ONE line (max 15
words) saying what kind of note belongs in this folder. Describe the topic and
kind of content, not the specific notes. No preamble, no quotes, just the line."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max-notes", type=int, default=4, help="excerpts per folder")
    ap.add_argument("--chars", type=int, default=300, help="chars per excerpt")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    train = [json.loads(l) for l in args.train.read_text().splitlines() if l.strip()]
    key = lambda p: "/" + "/".join(p)  # noqa: E731
    by_dir: dict[str, list[str]] = collections.defaultdict(list)
    for t in train:
        by_dir[key(t["gold_path"])].append(t["text"])

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    folders = sorted(by_dir)
    out: dict[str, str] = {}
    tok = [0, 0]
    lock = threading.Lock()

    def one(f: str) -> None:
        notes = by_dir[f][: args.max_notes]
        body = "\n\n".join(f"- {n[: args.chars]}" for n in notes)
        try:
            r = client.chat.completions.create(
                model=args.model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Folder: {f}\n\nNotes filed here:\n{body}"},
                ],
            )
            desc = (r.choices[0].message.content or "").strip().strip('"')
            with lock:
                tok[0] += r.usage.prompt_tokens
                tok[1] += r.usage.completion_tokens
        except Exception as e:  # noqa: BLE001
            desc = ""
            print(f"FAIL {f}: {e}", flush=True)
        out[f] = desc

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, folders))

    cost = tok[0] / 1e6 * 0.15 + tok[1] / 1e6 * 0.60  # gpt-4o-mini
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "source_train": str(args.train),
                "model": args.model,
                "n_folders": len(folders),
                "n_empty": sum(1 for v in out.values() if not v),
                "prompt_tokens": tok[0],
                "completion_tokens": tok[1],
                "cost_usd_est": round(cost, 4),
                "descriptions": out,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{len(folders)} folders described, ~${cost:.3f}")
    for f in folders[:5]:
        print(f"  {f}\n     -> {out[f]}")


if __name__ == "__main__":
    main()
