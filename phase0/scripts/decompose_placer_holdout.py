#!/usr/bin/env python3
"""Break placer accuracy out by gold class.

60% of the synth gold paths end in the literal segment `results` (every
generated tree carries `<root>/<area>/results`). An overall holdout number is
therefore dominated by whether the model learned that one filing convention,
which is not the same skill as reading a note and choosing a folder. Split it:

  ends_results  the majority convention
  depth1        root-only ("inbox" catch-all)
  other         the actual placement decisions

`other` is the number to compare across arms, and the one that lines up with
real-tree transfer.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def gold_class(gold: list[str]) -> str:
    if gold and gold[-1] == "results":
        return "ends_results"
    if len(gold) == 1:
        return "depth1"
    return "other"


def load(path: Path) -> list[dict]:
    for name in ("holdout_results.jsonl", "results.jsonl", "llm_holdout_results.jsonl"):
        p = path / name if path.is_dir() else path
        if p.exists():
            return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    raise SystemExit(f"no result rows under {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="label=path/to/run_dir")
    args = ap.parse_args()

    table: dict[str, dict[str, tuple[int, int, int]]] = {}
    for spec in args.arms:
        label, _, p = spec.partition("=")
        rows = load(Path(p or label))
        grp: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
        for r in rows:
            c = gold_class(r["gold_path"])
            grp[c][0] += bool(r.get("exact"))
            grp[c][1] += bool(r.get("soft_hit"))
            grp[c][2] += 1
        grp["ALL"] = [
            sum(bool(r.get("exact")) for r in rows),
            sum(bool(r.get("soft_hit")) for r in rows),
            len(rows),
        ]
        table[label] = {k: tuple(v) for k, v in grp.items()}

    classes = ["ends_results", "depth1", "other", "ALL"]
    w = max(len(l) for l in table) + 2
    print(f"{'arm':<{w}}" + "".join(f"{c:>18}" for c in classes))
    for label, grp in table.items():
        cells = []
        for c in classes:
            if c not in grp:
                cells.append(f"{'-':>18}")
                continue
            ex, soft, n = grp[c]
            cells.append(f"{ex/n:>10.3f} (n={n})")
        print(f"{label:<{w}}" + "".join(cells))
    print("\ncell = exact accuracy. `other` is the placement-skill column.")


if __name__ == "__main__":
    main()
