#!/usr/bin/env python3
"""Domain-shift multitree synth — closer to messy personal/work notes.

Anti-goals vs mid_hard:
  - short template one-liners → markdown-ish multi-line notes
  - clean area names only → mix file-like dir names (readme.md, handoff.md)
  - explicit area tags → weaker cues buried in prose
  - still: whole-tree holdout, exact gold for SFT, never use user_dir_snap texts
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.store import HierStore, Op  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "build_multitree_synth", ROOT / "scripts" / "build_multitree_synth.py"
)
_base = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_base)
LAYOUTS = _base.LAYOUTS


def fid(tree_id: str, *parts: str) -> str:
    h = hashlib.sha1(f"{tree_id}|{'|'.join(parts)}".encode()).hexdigest()[:10]
    return f"{tree_id}_{h}"


# multi-line note bodies; {area} optional / weak
NOTE_BODIES_A = [
    """# Status — {area}

Last touch: keep going until freeze.
Open: wiring + eval harness.
Do **not** assume yesterday's volume still mounts.

## Next
- finish curve
- bank numbers
- update table
""",
    """## Handoff ({area})

Pod still warm. Credits tight.
Azure = cold store; re-stage if needed.

Checklist:
1. dump scores
2. plot compare
3. write note in tree
""",
    """@notes {area}

scratch from today — metric dipped after merge.
someone should ping infra about disk.

```
run: python scripts/dump_scores.py
```
""",
    """# FOCUS {area}

Status: long job running.
Later: new day pod + fresh volume.

| item | state |
|------|-------|
| stage | hot |
| eval | pending |
""",
    """README fragment ({area})

This folder is the working set for the current pass.
If unsure where a blob goes, park under misc/inbox — but try here first.
""",
]

NOTE_BODIES_B = [
    """### {area} working notes

Overnight sweep paged twice.
Action: ask about disk pressure before Monday.

Don't ship until ablation lands.
""",
    """# {area} — weekly

Blockers listed below. Owners fuzzy.

- deploy red after config
- GPU quota mid-run
- bill spiked (pod left warm)
""",
    """thread dump / {area}

Primary metric dropped post-merge.
NaNs when batching long inputs.

Park results under results/ if you mkdir.
""",
    """## scratchpad

re: {area} review ritual for newcomers
budget lines this quarter still wrong

untitled — move later if needed
""",
    """# Agent context map ({area})

**Context** = token pack for this call.
**Memory** = durable store outside the window.

Short field baseline. Update when the run finishes.
""",
]

LIFE_NOTES_A = [
    "calendar nudge #{n} under {root}/{area} — dentist/tax/errand style",
    "personal sticky ({area}): pay / renew / reply #{n}",
    "FYI {area} life-admin #{n}",
]
LIFE_NOTES_B = [
    "remind me ({area}) item {n}",
    "{area} chore list — #{n}",
    "parked life note {n} → {area}",
]

DUMP_A = [
    "asdf untitled {tid} {n}\n\nrandom paste, no home yet",
    "zzz\nscratch {n}\n({tid})",
]
DUMP_B = [
    "misc blob {n}\n\n# no title\njust parking",
    "noise {tid}-{n} — delete later?",
]

# file-like extra dirs under twin_root (matches weird user-dir shape)
FILEISH = ["readme.md", "handoff.md", "notes.md", "TODO.md", "results.md"]


def pick(xs: list, holdout: bool, rng: random.Random):
    part = xs[1::2] if holdout else xs[0::2]
    return rng.choice(part or xs)


def build_tree(layout: dict, out_dir: Path, rng: random.Random, *, holdout: bool) -> dict:
    tid = layout["id"]
    tdir = out_dir / tid
    tdir.mkdir(parents=True, exist_ok=True)
    db = tdir / "hierstore.sqlite"
    if db.exists():
        db.unlink()

    roots = layout["roots"]
    twin_root = layout["twin_root"]
    areas = layout["areas"]
    store = HierStore(db, roots=roots, max_depth=5)
    ops: list[Op] = []
    place_tasks: list[dict] = []

    for a in areas:
        ops.append(Op(op="MKDIR", path=[twin_root, a]))
        ops.append(Op(op="MKDIR", path=[twin_root, a, "results"]))
    # file-ish siblings (confusable with real snaps)
    fileish_dirs = rng.sample(FILEISH, k=min(3, len(FILEISH)))
    for name in fileish_dirs:
        ops.append(Op(op="MKDIR", path=[twin_root, name]))

    life_root = roots[1]
    for a in layout["life_areas"]:
        ops.append(Op(op="MKDIR", path=[life_root, a]))
        tmpl = pick(LIFE_NOTES_B if holdout else LIFE_NOTES_A, holdout, rng)
        text = tmpl.format(root=life_root, area=a, n=rng.randint(1, 99))
        fact_id = fid(tid, "life", a, text)
        path = [life_root, a]
        ops.append(Op(op="ADD", fact_id=fact_id, text=text, path=path))
        place_tasks.append(
            {
                "id": f"place_{fact_id}",
                "kind": "place",
                "text": text,
                "gold_path": path,
                "roots": roots,
                "tree_id": tid,
            }
        )

    dump_root = roots[2]
    ops.append(Op(op="MKDIR", path=[dump_root]))
    dump_text = pick(DUMP_B if holdout else DUMP_A, holdout, rng).format(
        tid=tid, n=rng.randint(1000, 9999)
    )
    dump_id = fid(tid, "dump", dump_text)
    ops.append(Op(op="ADD", fact_id=dump_id, text=dump_text, path=[dump_root]))
    place_tasks.append(
        {
            "id": f"place_{dump_id}",
            "kind": "place",
            "text": dump_text,
            "gold_path": [dump_root],
            "roots": roots,
            "tree_id": tid,
        }
    )

    bodies = NOTE_BODIES_B if holdout else NOTE_BODIES_A
    # unique-ish notes per area + some into results / fileish
    for a in areas:
        for i in range(5):
            body = pick(bodies, holdout, rng).format(area=a)
            # weaken: sometimes strip explicit area from title by paraphrasing path only via gold
            if rng.random() < 0.35:
                body = body.replace(a, "this track").replace(a.title(), "this track")
            if i % 3 == 0:
                path = [twin_root, a, "results"]
            elif i % 3 == 1 and fileish_dirs and rng.random() < 0.25:
                path = [twin_root, rng.choice(fileish_dirs)]
            else:
                path = [twin_root, a]
            fact_id = fid(tid, "note", a, str(i), body[:80])
            ops.append(Op(op="ADD", fact_id=fact_id, text=body, path=path))
            place_tasks.append(
                {
                    "id": f"place_{fact_id}",
                    "kind": "place",
                    "text": body,
                    "gold_path": path,
                    "roots": roots,
                    "tree_id": tid,
                }
            )

    # twin-ish near-dups across first two areas (prose, weak tags)
    a0, a1 = areas[0], areas[1]
    twin_bodies = [
        "Deploy failed Friday after the config change. Keep under results.",
        "GPU quota hit mid-run; paused until Monday.",
        "Primary metric dropped after the last merge — log it.",
    ]
    for bi, body in enumerate(twin_bodies):
        body = pick(
            [body, body + " (follow-up)", "Update: " + body.lower()],
            holdout,
            rng,
        )
        for a in (a0, a1):
            text = f"{body}\n\ncontext: active workstream ~{a}"
            path = [twin_root, a, "results"]
            fact_id = fid(tid, "twin", a, str(bi))
            ops.append(Op(op="ADD", fact_id=fact_id, text=text, path=path))
            place_tasks.append(
                {
                    "id": f"place_{fact_id}",
                    "kind": "place",
                    "text": text,
                    "gold_path": path,
                    "roots": roots,
                    "tree_id": tid,
                }
            )

    store.apply_ops(ops, manager="domain_shift_synth")
    existing_dirs = store.list_dirs()
    snap = store.snapshot()
    store.close()
    for t in place_tasks:
        t["existing_dirs"] = existing_dirs

    (tdir / "place_tasks.jsonl").write_text(
        "\n".join(json.dumps(t) for t in place_tasks) + "\n"
    )
    meta = {
        "tree_id": tid,
        "roots": roots,
        "holdout_tree": holdout,
        "n_facts": snap["valid_count"],
        "n_dirs": len(existing_dirs),
        "n_place_tasks": len(place_tasks),
        "db": str(db),
    }
    (tdir / "META.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "multitree_domain_shift")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n-holdout", type=int, default=10)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    ids = [L["id"] for L in LAYOUTS]
    holdout = set(ids[-args.n_holdout :])
    metas = [
        build_tree(L, args.out, rng, holdout=L["id"] in holdout) for L in LAYOUTS
    ]

    train, hold = [], []
    for m in metas:
        tasks = [
            json.loads(l)
            for l in (args.out / m["tree_id"] / "place_tasks.jsonl")
            .read_text()
            .splitlines()
            if l.strip()
        ]
        (hold if m["tree_id"] in holdout else train).extend(tasks)

    tr_txt, ho_txt = {t["text"] for t in train}, {t["text"] for t in hold}
    leak = len(tr_txt & ho_txt)
    (args.out / "place_train.jsonl").write_text(
        "\n".join(json.dumps(t) for t in train) + "\n"
    )
    (args.out / "place_holdout.jsonl").write_text(
        "\n".join(json.dumps(t) for t in hold) + "\n"
    )
    summary = {
        "n_trees": len(metas),
        "train_trees": [i for i in ids if i not in holdout],
        "holdout_trees": sorted(holdout),
        "n_place_train": len(train),
        "n_place_holdout": len(hold),
        "n_facts_total": sum(m["n_facts"] for m in metas),
        "seed": args.seed,
        "variant": "domain_shift_markdown",
        "exact_text_overlap_train_holdout": leak,
        "lock": "B+D; domain-shift toward user-dir style; never train on user snap",
        "trees": metas,
    }
    (args.out / "META.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "trees"}, indent=2))


if __name__ == "__main__":
    main()
