#!/usr/bin/env python3
"""Stratified 100-item gold-label sample from corpus A, for the annotation study.

Draws from item-split val (folders seen in training) and folder-split val
(folders never seen — occupancy bucket "0" by construction). These are two
independent random splits of the same underlying task pool, so the same
physical note can land in both splits' val sets — dedupe by id before
sampling or the same item can appear twice under two different bucket labels.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("probe_llm_placer", ROOT / "scripts" / "probe_llm_placer.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", type=Path, default=ROOT / "data" / "user_dir_snap_v2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plan", default="0:35,1-2:18,3-9:25,10+:22")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-ids", type=Path, default=None,
                    help="JSON file mapping id -> bucket for items an annotator "
                    "already judged; pins each id to the bucket it was actually "
                    "shown under (an id can legitimately exist in more than one "
                    "split's pool with a different bucket, so a bare id list is "
                    "not enough to know which context was judged)")
    args = ap.parse_args()

    plan = {}
    for pair in args.plan.split(","):
        k, v = pair.split(":")
        plan[k] = int(v)

    train = [json.loads(l) for l in (args.snap / "train.jsonl").read_text().splitlines() if l.strip()]
    val = [json.loads(l) for l in (args.snap / "val.jsonl").read_text().splitlines() if l.strip()]
    fold_val = [json.loads(l) for l in (args.snap / "fold_val.jsonl").read_text().splitlines() if l.strip()]

    cnt = defaultdict(int)
    for t in train:
        cnt[tuple(t["gold_path"])] += 1

    def item_bucket(gp: list[str]) -> str:
        n = cnt[tuple(gp)]
        return "1-2" if n <= 2 else "3-9" if n <= 9 else "10+"

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for t in val:
        b = item_bucket(t["gold_path"])
        by_bucket[b].append({**t, "split": "item", "occupancy_bucket": b})
    for t in fold_val:
        by_bucket["0"].append({**t, "split": "folder", "occupancy_bucket": "0"})

    # id -> bucket it was already judged under (pins the context; an id can
    # legitimately sit in more than one split's pool with a different bucket)
    keep_map: dict[str, str] = {}
    if args.keep_ids and args.keep_ids.exists():
        keep_map = json.loads(args.keep_ids.read_text())

    rng = random.Random(args.seed)
    sample: list[dict] = []
    used_ids: set[str] = set()
    for bucket in ("1-2", "3-9", "10+", "0"):
        n = plan.get(bucket, 0)
        pool = [t for t in by_bucket[bucket] if t["id"] not in used_ids]
        rng.shuffle(pool)
        forced = [t for t in pool if keep_map.get(t["id"]) == bucket]
        rest = [t for t in pool if keep_map.get(t["id"]) != bucket]
        take = (forced + rest)[:max(n, len(forced))]
        sample.extend(take)
        used_ids.update(t["id"] for t in take)
    rng.shuffle(sample)

    existing_dirs = probe.dirs_from_store(args.snap / "hierstore.sqlite", ["work", "personal", "inbox"])
    existing_dirs_str = ["/".join(d) for d in existing_dirs]

    items = [{
        "id": t["id"], "text": t["text"], "gold_path": t["gold_path"],
        "occupancy_bucket": t["occupancy_bucket"], "split": t["split"],
    } for t in sample]

    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), "duplicate id survived sampling"

    out = {
        "corpus": "A", "existing_dirs": existing_dirs_str, "n_items": len(items),
        "bucket_plan": plan, "items": items,
    }
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    from collections import Counter
    print(f"wrote {len(items)} unique items -> {args.out}")
    print(Counter(i["occupancy_bucket"] for i in items))


if __name__ == "__main__":
    main()
