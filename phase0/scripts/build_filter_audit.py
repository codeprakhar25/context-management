#!/usr/bin/env python3
"""H3 vault-filter audit sheets (PLACER_FINDINGS.md, H3): did the is_software classifier
admit/reject the right repos?

Draws 40 repos from data/vaults_classified.json, 20 per classifier label, seed 0. Each
annotator labels each repo from its GitHub page as a personal note collection or not. The
classifier label is NOT on the sheet; it goes to a separate key file. Item order is
shuffled per annotator (seeded). The draw is over all 132 classified rows (not just the 27
that made it into corpus B) because the audit is of the classifier itself.

No API calls, no network: the annotators open the URLs.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANNOTATORS = ["annotator_1", "annotator_2"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classified", type=Path, default=ROOT / "data" / "vaults_classified.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "annotation_B")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = json.loads(args.classified.read_text())["rows"]
    rng = random.Random(args.seed)
    items = []
    for is_sw in (False, True):
        pool = sorted((r["full_name"] for r in rows if bool(r["is_software"]) == is_sw))
        assert len(pool) >= args.per_class, f"only {len(pool)} rows with is_software={is_sw}"
        items += [(n, is_sw) for n in rng.sample(pool, args.per_class)]
    rng.shuffle(items)

    key = {}
    sheet = []
    for i, (name, is_sw) in enumerate(items):
        code = f"H3-{i + 1:03d}"
        key[code] = {"full_name": name, "is_software": is_sw}
        sheet.append([code, f"https://github.com/{name}"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "h3_key.json").write_text(json.dumps(key, indent=2))
    for a in ANNOTATORS:
        order = list(sheet)
        random.Random(f"{args.seed}/{a}/h3").shuffle(order)
        p = args.out_dir / a / "h3_filter_audit.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow(["item", "github_url", "personal_note_collection (personal / not)", "notes"])
            w.writerows([[code, url, "", ""] for code, url in order])
    n_sw = sum(k["is_software"] for k in key.values())
    print(f"wrote {len(key)} repos ({len(key) - n_sw} classifier-personal, {n_sw} classifier-software) -> {args.out_dir}")


if __name__ == "__main__":
    main()
