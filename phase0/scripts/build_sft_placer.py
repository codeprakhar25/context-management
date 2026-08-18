#!/usr/bin/env python3
"""Build OpenAI chat-SFT JSONL for path placer from user-dir snap gold.

Train/val split stratified by root. Each example includes existing_dirs from
the frozen HierStore (same prompt as probe_llm_placer).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.store import MAX_DEPTH  # noqa: E402


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "probe_llm_placer", ROOT / "scripts" / "probe_llm_placer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def to_openai_row(
    task: dict,
    existing_dirs: list[list[str]],
    max_depth: int,
    placer_system: str,
) -> dict:
    roots = task.get("roots") or ["work", "personal", "inbox"]
    system = placer_system.format(roots=roots, max_depth=max_depth)
    user_obj = {
        "task": task.get("kind", "place"),
        "fact_text": task["text"],
        "existing_dirs": existing_dirs,
        "from_path": task.get("from_path"),
        "cue": task.get("cue"),
    }
    user = (
        json.dumps(user_obj, indent=2)
        + "\n\nChoose the destination path. JSON only."
    )
    assistant = json.dumps(
        {"path": task["gold_path"], "confidence": 0.9},
        ensure_ascii=False,
    )
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {
            "id": task["id"],
            "gold_path": task["gold_path"],
            "root": task["gold_path"][0] if task.get("gold_path") else None,
        },
    }


def split_stratified(
    tasks: list[dict], val_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_root: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        r = (t.get("gold_path") or ["?"])[0]
        by_root[r].append(t)
    train, val = [], []
    for _r, group in by_root.items():
        rng.shuffle(group)
        if len(group) >= 3:
            n_val = max(1, int(round(len(group) * val_frac)))
        else:
            n_val = 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


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
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "sft_placer")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    probe = _load_probe()
    tasks = [
        json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()
    ]
    roots = tasks[0].get("roots") if tasks else ["work", "personal", "inbox"]
    existing = probe.dirs_from_store(
        args.store, roots or ["work", "personal", "inbox"]
    )
    train_t, val_t = split_stratified(tasks, args.val_frac, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    train_rows = [
        to_openai_row(t, existing, MAX_DEPTH, probe.PLACER_SYSTEM) for t in train_t
    ]
    val_rows = [
        to_openai_row(t, existing, MAX_DEPTH, probe.PLACER_SYSTEM) for t in val_t
    ]

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "\n".join(json.dumps({"messages": r["messages"]}, ensure_ascii=False) for r in rows)
            + ("\n" if rows else "")
        )

    write_jsonl(args.out / "train.jsonl", train_rows)
    write_jsonl(args.out / "val.jsonl", val_rows)
    (args.out / "val_meta.jsonl").write_text(
        "\n".join(json.dumps(r["meta"]) for r in val_rows) + "\n"
    )
    val_by_id = {t["id"]: t for t in val_t}
    (args.out / "val_tasks.jsonl").write_text(
        "\n".join(
            json.dumps(val_by_id[r["meta"]["id"]]) for r in val_rows
        )
        + "\n"
    )

    meta = {
        "n_tasks": len(tasks),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_existing_dirs": len(existing),
        "val_frac": args.val_frac,
        "seed": args.seed,
        "store": str(args.store),
        "tasks": str(args.tasks),
    }
    (args.out / "META.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
