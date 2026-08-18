#!/usr/bin/env python3
"""Clean a user-dir snapshot into a train/val split fit to train and score on.

Two filters, both of which otherwise show up as fake error in every arm:

  duplicate text  Sweeping whole repos pulls in copied and vendored files, so
                  the identical note body sits in two different folders. No
                  model can be right on both; the pair is irreducible error.

  depth-truncated `path_for_file` caps paths at MAX_DEPTH by keeping the root
                  plus the tail, so a natural depth-8 file gets a mangled gold
                  path that never existed on disk.

Split is stratified by gold folder. Folders with <3 files contribute entirely
to train, which keeps every val item in a folder the model has seen — the
question under test is whether a model learns THIS tree's conventions, not
whether it can invent unseen folders.

Outputs <snap>/{place_tasks_clean,train,val}.jsonl + split.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def natural_depth(task: dict, sources: dict[str, list[Path]]) -> int | None:
    """Folder depth the file actually has on disk, before MAX_DEPTH capping."""
    f = Path(task["source_file"])
    root = task["gold_path"][0]
    for src in sources.get(root, []):
        try:
            rel = f.relative_to(src)
        except ValueError:
            continue
        area = 1 if src.name.replace("_", "-").lower() != root else 0
        parts = rel.parent.parts if str(rel.parent) != "." else ()
        return 1 + area + len(parts)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", type=Path, default=ROOT / "data" / "user_dir_snap_v2")
    ap.add_argument("--val-frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--split-by",
        default="item",
        choices=["item", "folder"],
        help="item: stratified inside each folder, so every val folder was seen "
        "in training — tests learning THIS tree's conventions. folder: whole "
        "folders held out, so no val folder was ever trained on — tests placing "
        "into a folder the model knows exists but has never seen used.",
    )
    ap.add_argument("--out-prefix", default="", help="prefix for output filenames")
    args = ap.parse_args()

    cfg = json.loads((args.snap / "sources.json").read_text())
    max_depth = int(cfg.get("max_depth") or 5)
    sources = {r: [Path(s) for s in v] for r, v in cfg["sources"].items()}
    tasks = [
        json.loads(l)
        for l in (args.snap / "place_tasks_from_snap.jsonl").read_text().splitlines()
        if l.strip()
    ]

    text_count = collections.Counter(t["text"] for t in tasks)
    n_dup = n_trunc = 0
    clean = []
    for t in tasks:
        if text_count[t["text"]] > 1:
            n_dup += 1
            continue
        d = natural_depth(t, sources)
        if d is not None and d > max_depth:
            n_trunc += 1
            continue
        clean.append(t)

    key = lambda p: "/" + "/".join(p)  # noqa: E731
    rng = random.Random(args.seed)
    by_dir: dict[str, list[dict]] = collections.defaultdict(list)
    for t in clean:
        by_dir[key(t["gold_path"])].append(t)

    train, val = [], []
    if args.split_by == "folder":
        # Whole folders to one side or the other. Greedy fill to the target
        # fraction over a shuffled folder order, so val folder sizes stay mixed
        # rather than being all-tiny or all-huge.
        folders = list(by_dir)
        rng.shuffle(folders)
        target = args.val_frac * len(clean)
        for d in folders:
            if len(val) < target:
                val.extend(by_dir[d])
            else:
                train.extend(by_dir[d])
    else:
        for group in by_dir.values():
            rng.shuffle(group)
            n_val = max(1, round(len(group) * args.val_frac)) if len(group) >= 3 else 0
            val.extend(group[:n_val])
            train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)

    seen = {key(t["gold_path"]) for t in train}
    counts = collections.Counter(key(t["gold_path"]) for t in clean)
    meta = {
        "snap": str(args.snap),
        "n_raw": len(tasks),
        "n_dropped_duplicate_text": n_dup,
        "n_dropped_depth_truncated": n_trunc,
        "n_clean": len(clean),
        "val_frac": args.val_frac,
        "split_by": args.split_by,
        "seed": args.seed,
        "n_train": len(train),
        "n_val": len(val),
        "n_gold_dirs": len(counts),
        "n_singleton_dirs": sum(1 for v in counts.values() if v == 1),
        "majority_class_baseline": round(counts.most_common(1)[0][1] / len(clean), 4),
        "val_in_seen_folder": sum(1 for t in val if key(t["gold_path"]) in seen),
        "roots": collections.Counter(t["gold_path"][0] for t in clean),
    }

    pre = args.out_prefix
    for name, rows in (("place_tasks_clean", clean), ("train", train), ("val", val)):
        (args.snap / f"{pre}{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
    (args.snap / f"{pre}split.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
