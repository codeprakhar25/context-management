#!/usr/bin/env python3
"""Pick the corpus-B vaults from a probe file and pin them to commit SHAs.

Corpus B is public personal markdown knowledge bases -- people's own note stores,
with folder hierarchies they invented for themselves. That is the setting the
placer is actually for, and it is the one thing our private corpus cannot be
audited on.

Input is the output of classify_vaults.py, which has already dropped the software
repos. Two filters remain, both because a large repo is not the same as a useful
one:

  denylist  Auto-converted published reference works (SRD rulebooks, TTRPG
            compendia, documentation-generator sites) pass every structural check
            and are exactly wrong for us: their hierarchy is a book's table of
            contents, not a personal folksonomy. Left in, the single largest one
            would also outweigh every real vault combined.
  size      A per-vault note cap, so one big vault cannot dominate the pooled
            coverage curve.

Licensing: we redistribute this manifest and the build scripts, never the note
text, so the corpus is manifest-only in the same sense as a URL list. Licenses
are recorded per vault rather than used as a filter. These are real people's
personal notes -- report aggregate statistics, do not quote notes verbatim, and
honor takedown requests against the manifest.

Output is pinned to commit SHAs, so the corpus is reconstructible after the
upstream repos change.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Auto-converted published works, product docs, and empty scaffolds. Matched
# against the full name, case-insensitive. These survive classify_vaults.py
# because they genuinely contain no code -- a converted rulebook is pure markdown.
DENY_PATTERNS = [
    r"srd",              # d20 System Reference Documents
    r"dnd5e",
    r"pathfinder",
    r"ttrpg",
    r"old-school-essentials",
    r"-template$", r"template-", r"starter-kit", r"_template",
    r"awesome-",
    r"retypeapp",        # documentation-generator product site
    r"srid/neuron",      # note-taking tool's own docs
    r"dataview-cards",   # plugin demo vault, not a used note store
]


def sh(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, timeout=60).stdout.strip()


def denied(full_name: str) -> str | None:
    low = full_name.lower()
    for pat in DENY_PATTERNS:
        if re.search(pat, low):
            return pat
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-vaults", type=int, default=25)
    ap.add_argument("--min-notes", type=int, default=60)
    ap.add_argument(
        "--max-notes-per-vault",
        type=int,
        default=250,
        help="notes sampled per vault at build time; caps how far one large vault "
        "can dominate the pooled coverage curve",
    )
    ap.add_argument("--require-obsidian", action="store_true",
                    help="keep only repos with a .obsidian/ directory")
    args = ap.parse_args()

    probe = json.loads(args.probe.read_text())
    rows = [r for r in probe["rows"] if r["usable"]]

    kept, rejected = [], []
    for r in rows:
        why = None
        if r.get("is_software"):
            why = f"software: manifests={r.get('manifests')} code={r.get('n_code')}"
        elif (pat := denied(r["full_name"])):
            why = f"denylist:{pat}"
        elif r["n_md_in_dirs"] < args.min_notes:
            why = f"notes={r['n_md_in_dirs']}"
        elif args.require_obsidian and not r["has_obsidian"]:
            why = "no .obsidian/"
        (rejected if why else kept).append({**r, "reject_reason": why})

    # Spread the selection across vault sizes rather than taking the biggest N:
    # the coverage curve needs folders at every occupancy, and big vaults are
    # skewed toward well-populated folders.
    kept.sort(key=lambda r: r["n_md_in_dirs"])
    if len(kept) > args.max_vaults:
        step = len(kept) / args.max_vaults
        kept = [kept[int(i * step)] for i in range(args.max_vaults)]

    print(f"probed {len(probe['rows'])}  usable {len(rows)}  selected {len(kept)}",
          file=sys.stderr)
    for r in kept:
        sha = sh(["gh", "api", f"repos/{r['full_name']}/commits/HEAD", "--jq", ".sha"])
        r["commit"] = sha
        r["clone_url"] = f"https://github.com/{r['full_name']}.git"
        print(f"  {r['full_name']:<48} {r['n_md_in_dirs']:>5} notes "
              f"{r['n_dirs']:>4} dirs  {sha[:8]}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "source_probe": str(args.probe),
        "n_probed": len(probe["rows"]),
        "n_usable": len(rows),
        "n_selected": len(kept),
        "max_notes_per_vault": args.max_notes_per_vault,
        "license_policy": "manifest-only; licenses recorded per vault, not used "
                          "as a filter. Note text is never redistributed.",
        "deny_patterns": DENY_PATTERNS,
        "total_notes_selected": sum(r["n_md_in_dirs"] for r in kept),
        "total_dirs_selected": sum(r["n_dirs"] for r in kept),
        "vaults": kept,
        "rejected_sample": rejected[:60],
    }, indent=2) + "\n")
    print(f"\n{len(kept)} vaults  "
          f"{sum(r['n_md_in_dirs'] for r in kept)} notes  "
          f"{sum(r['n_dirs'] for r in kept)} folders -> {args.out}")


if __name__ == "__main__":
    main()
