#!/usr/bin/env python3
"""Harder multitree synth — anti-template-leak (research placer).

Same tree layouts / path gold as mid, but:
  - large paraphrase banks (no shared exact strings train↔holdout)
  - area cues varied (not always \"[{area}]\")
  - life/dump text diversified

Holdout trees get paraphrase partition B; train gets A.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.store import HierStore, Op  # noqa: E402

# reuse LAYOUTS + fid from smoke/mid builder
_spec = importlib.util.spec_from_file_location(
    "build_multitree_synth", ROOT / "scripts" / "build_multitree_synth.py"
)
_base = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_base)
LAYOUTS = _base.LAYOUTS
fid = _base.fid

# Intent slots for unique facts — each has many paraphrases (must contain {area})
UNIQUE_BANK: dict[str, list[str]] = {
    "sync": [
        "Weekly sync notes for {area}: blockers and owners",
        "{area} standup writeup — who owns what this week",
        "Notes from the {area} sync; open blockers listed",
        "Catch-up doc after {area} weekly: owners + slips",
        "Thread dump: {area} meeting, action owners attached",
        "What we covered in {area} sync — blockers still open",
        "{area} cadence notes (owners, risks, next steps)",
        "Post-{area}-meeting scratch: owners and stuck items",
    ],
    "ship": [
        "Checklist before shipping {area} changes to prod",
        "Pre-prod gate for {area} — don't merge cold",
        "Ship checklist ({area}): tests, flags, rollback",
        "Go/no-go list ahead of {area} release",
        "{area} deploy readiness: boxes left unchecked",
        "Before {area} hits prod — verify this list",
        "Release hygiene for {area} (smoke + owners)",
        "Prod ship gate notes under {area}",
    ],
    "onboard": [
        "Onboarding doc: how we run {area} reviews",
        "New hire guide to {area} review process",
        "How reviews work in {area} — short orientation",
        "{area} review ritual explained for newcomers",
        "Bootstrapping into {area}: review norms",
        "Read this first if you join {area} reviews",
        "Orientation scrap for {area} review loop",
        "Starter notes: participating in {area} reviews",
    ],
    "retro": [
        "Retrospective themes from last {area} sprint",
        "What went wrong last {area} sprint (retro)",
        "{area} retro themes — keep / kill / try",
        "Sprint lookback for {area}: recurring pain",
        "Themes we keep hearing in {area} retros",
        "After-action on the last {area} cycle",
        "Retro digest ({area}): top complaints",
        "Last {area} sprint autopsy notes",
    ],
    "budget": [
        "Budget line items tied to {area} this quarter",
        "{area} spend lines for this quarter",
        "Quarterly budget rows that touch {area}",
        "Where {area} money goes this Q",
        "Cost map: {area} line items (current quarter)",
        "Finance scrap for {area} budget this quarter",
        "{area} cost centers / lines for Q",
        "This quarter's {area} budget crumbs",
    ],
}

# Twin semantics — paraphrase lists; even idx → train partition, odd → holdout
SHARED_BANK: dict[str, list[str]] = {
    "deploy": [
        "Deploy failed on Friday night after the config change",
        "Friday night deploy blew up once config landed",
        "Config change → Friday deploy red",
        "Ship died Friday evening post-config tweak",
        "After the config edit, Friday deploy cratered",
        "Night deploy failed Friday; blame the config diff",
        "Friday deploy outage traced to config change",
        "Config push, then Friday night deploy went south",
    ],
    "quota": [
        "Hit GPU quota mid-run; jobs paused until Monday",
        "GPU quota exhausted mid-job — parked until Monday",
        "Ran into GPU limit; workloads idle till Monday",
        "Quota wall on GPUs; nothing runs before Monday",
        "Mid-run GPU cap; resume Monday",
        "Jobs stalled: GPU quota gone until Monday",
        "Burned through GPU quota; Monday restart",
        "Paused on GPU quota — wait for Monday reset",
    ],
    "metric": [
        "Primary metric dropped after the last merge",
        "Main metric fell off a cliff post-merge",
        "After the merge, the headline metric tanked",
        "Merge landed; primary KPI went down",
        "Last merge hurt the primary metric",
        "KPI regression showed up right after merge",
        "Primary score slid once that merge hit",
        "Post-merge dip in the main metric",
    ],
    "meeting": [
        "Ship the ablation before the freeze next week",
        "Ablation needs to land before next week's freeze",
        "Get ablation done prior to the freeze",
        "Freeze next week — ablation must ship first",
        "Before freeze: finish the ablation",
        "Ablation is the blocker ahead of freeze week",
        "Need ablation out before the freeze window",
        "Next-week freeze means ablation ships now",
    ],
    "bug": [
        "NaNs appeared when batching long inputs",
        "Long-input batching started producing NaNs",
        "NaN spike when we batch long sequences",
        "Batching long examples → NaNs",
        "Seeing NaNs under long-input batches",
        "Long batches go NaN",
        "NaNs show up only on long batched inputs",
        "Batch path on long inputs yields NaNs",
    ],
    "bill": [
        "Cloud bill spiked after leaving the pod warm overnight",
        "Left the pod hot overnight — cloud bill jumped",
        "Warm pod overnight → ugly cloud invoice",
        "Overnight idle pod cooked the cloud bill",
        "Bill shock from a pod left running all night",
        "Forgot to stop the pod; overnight spend spiked",
        "Cloud cost blew up after overnight warm pod",
        "Pod stayed up overnight and the bill followed",
    ],
}

IDENTICAL_BANK: dict[str, list[str]] = {
    "alert": [
        "PagerDuty fired twice during the overnight sweep",
        "Two PagerDuty pages in the overnight sweep",
        "Overnight sweep: PagerDuty twice",
        "PD alerted twice overnight during sweep",
    ],
    "action": [
        "Action item: ping infra about disk pressure",
        "TODO — ask infra re disk pressure",
        "Follow up with infra on disk pressure",
        "Need infra pinged about disk pressure",
    ],
}

LIFE_BANK = [
    "Note under {root}/{area}: routine reminder {n}",
    "Reminder ({root}/{area}) #{n} — don't forget",
    "Small {root}/{area} note, ticket {n}",
    "{root}/{area} sticky: item {n}",
    "Parked under {root}/{area}: reminder {n}",
    "FYI {root}/{area} — routine #{n}",
]

DUMP_BANK = [
    "asdf scratch {tid} {n}",
    "random junk {tid}-{n}",
    "zzz parking lot {tid} {n}",
    "misc blob {n} ({tid})",
    "untitled scratchpad {tid} #{n}",
    "noise note {n} for {tid}",
]

AREA_CUES = [
    "[{area}]",
    "(area={area})",
    "— {area} track",
    "/ re:{area}",
    "· {area}",
    "#project-{area}",
]


def _pick_partition(phrases: list[str], holdout: bool, rng: random.Random) -> str:
    """Even indices = train partition, odd = holdout. Fallback shuffle if short."""
    if len(phrases) < 2:
        return phrases[0]
    part = phrases[1::2] if holdout else phrases[0::2]
    if not part:
        part = phrases
    return rng.choice(part)


def build_tree(
    layout: dict,
    out_dir: Path,
    rng: random.Random,
    *,
    holdout_tree: bool,
) -> dict:
    tid = layout["id"]
    tdir = out_dir / tid
    tdir.mkdir(parents=True, exist_ok=True)
    db = tdir / "hierstore.sqlite"
    if db.exists():
        db.unlink()

    roots = layout["roots"]
    store = HierStore(db, roots=roots, max_depth=5)
    ops: list[Op] = []
    place_tasks: list[dict] = []
    twin_queries: list[dict] = []

    twin_root = layout["twin_root"]
    areas = layout["areas"]
    for a in areas:
        ops.append(Op(op="MKDIR", path=[twin_root, a]))
        ops.append(Op(op="MKDIR", path=[twin_root, a, "results"]))

    life_root = roots[1]
    for a in layout["life_areas"]:
        ops.append(Op(op="MKDIR", path=[life_root, a]))
        tmpl = _pick_partition(LIFE_BANK, holdout_tree, rng)
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
    dump_tmpl = _pick_partition(DUMP_BANK, holdout_tree, rng)
    dump_text = dump_tmpl.format(tid=tid, n=rng.randint(1000, 9999))
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

    # unique facts — one paraphrase per intent×area, partition-locked
    for a in areas:
        for i, intent in enumerate(UNIQUE_BANK):
            tmpl = _pick_partition(UNIQUE_BANK[intent], holdout_tree, rng)
            text = tmpl.format(area=a)
            path = [twin_root, a] if i % 2 == 0 else [twin_root, a, "results"]
            fact_id = fid(tid, "uniq", a, intent, text)
            ops.append(Op(op="ADD", fact_id=fact_id, text=text, path=path))
            place_tasks.append(
                {
                    "id": f"place_{fact_id}",
                    "kind": "place",
                    "text": text,
                    "gold_path": path,
                    "roots": roots,
                    "tree_id": tid,
                    "intent": intent,
                }
            )

    a0, a1 = areas[0], areas[1]
    for tname, phrases in SHARED_BANK.items():
        body = _pick_partition(phrases, holdout_tree, rng)
        cue_tmpl = _pick_partition(AREA_CUES, holdout_tree, rng)
        for a in (a0, a1):
            text = f"{body} {cue_tmpl.format(area=a)}"
            path = [twin_root, a, "results"]
            fact_id = fid(tid, "shared", a, tname)
            ops.append(Op(op="ADD", fact_id=fact_id, text=text, path=path))
            place_tasks.append(
                {
                    "id": f"place_{fact_id}",
                    "kind": "place",
                    "text": text,
                    "gold_path": path,
                    "roots": roots,
                    "tree_id": tid,
                    "twin_key": tname,
                }
            )
        twin_queries.append(
            {
                "id": f"q_{tid}_shared_{tname}_{a0}",
                "question": body + "?",
                "active_path": [twin_root, a0],
                "gold_ids": [fid(tid, "shared", a0, tname)],
                "twin_id": fid(tid, "shared", a1, tname),
                "tree_id": tid,
                "kind": "shared",
            }
        )
        twin_queries.append(
            {
                "id": f"q_{tid}_shared_{tname}_{a1}",
                "question": body + "?",
                "active_path": [twin_root, a1],
                "gold_ids": [fid(tid, "shared", a1, tname)],
                "twin_id": fid(tid, "shared", a0, tname),
                "tree_id": tid,
                "kind": "shared",
            }
        )

    for tname, phrases in IDENTICAL_BANK.items():
        body = _pick_partition(phrases, holdout_tree, rng)
        for a in (a0, a1):
            path = [twin_root, a, "results"]
            fact_id = fid(tid, "ident", a, tname)
            ops.append(Op(op="ADD", fact_id=fact_id, text=body, path=path))
        twin_queries.append(
            {
                "id": f"q_{tid}_ident_{tname}_{a0}",
                "question": body + " Which project folder?",
                "active_path": [twin_root, a0],
                "gold_ids": [fid(tid, "ident", a0, tname)],
                "twin_id": fid(tid, "ident", a1, tname),
                "tree_id": tid,
                "kind": "identical",
            }
        )
        twin_queries.append(
            {
                "id": f"q_{tid}_ident_{tname}_{a1}",
                "question": body + " Which project folder?",
                "active_path": [twin_root, a1],
                "gold_ids": [fid(tid, "ident", a1, tname)],
                "twin_id": fid(tid, "ident", a0, tname),
                "tree_id": tid,
                "kind": "identical",
            }
        )

    store.apply_ops(ops, manager="multitree_synth_hard")
    existing_dirs = store.list_dirs()
    snap = store.snapshot()
    store.close()

    for t in place_tasks:
        t["existing_dirs"] = existing_dirs

    (tdir / "place_tasks.jsonl").write_text(
        "\n".join(json.dumps(t) for t in place_tasks) + "\n"
    )
    (tdir / "twin_queries.jsonl").write_text(
        "\n".join(json.dumps(q) for q in twin_queries) + "\n"
    )
    meta = {
        "tree_id": tid,
        "roots": roots,
        "holdout_tree": holdout_tree,
        "n_facts": snap["valid_count"],
        "n_dirs": len(existing_dirs),
        "n_place_tasks": len(place_tasks),
        "n_twin_queries": len(twin_queries),
        "twin_areas": [a0, a1],
        "db": str(db),
    }
    (tdir / "META.json").write_text(json.dumps(meta, indent=2))
    return meta


def leak_stats(train_tasks: list[dict], hold_tasks: list[dict]) -> dict:
    tr = {t["text"] for t in train_tasks}
    ho = {t["text"] for t in hold_tasks}
    return {
        "train_unique_texts": len(tr),
        "holdout_unique_texts": len(ho),
        "exact_text_overlap_train_holdout": len(tr & ho),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "multitree_synth_mid_hard",
    )
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n-holdout", type=int, default=10)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    ids = [L["id"] for L in LAYOUTS]
    holdout = set(ids[-args.n_holdout :])
    train_ids = [i for i in ids if i not in holdout]

    metas = []
    for layout in LAYOUTS:
        metas.append(
            build_tree(
                layout,
                args.out,
                rng,
                holdout_tree=layout["id"] in holdout,
            )
        )

    train_tasks, hold_tasks, twin_all = [], [], []
    for m in metas:
        tid = m["tree_id"]
        tdir = args.out / tid
        tasks = [
            json.loads(l)
            for l in (tdir / "place_tasks.jsonl").read_text().splitlines()
            if l.strip()
        ]
        twins = [
            json.loads(l)
            for l in (tdir / "twin_queries.jsonl").read_text().splitlines()
            if l.strip()
        ]
        twin_all.extend(twins)
        if tid in holdout:
            hold_tasks.extend(tasks)
        else:
            train_tasks.extend(tasks)

    leak = leak_stats(train_tasks, hold_tasks)
    (args.out / "place_train.jsonl").write_text(
        "\n".join(json.dumps(t) for t in train_tasks) + "\n"
    )
    (args.out / "place_holdout.jsonl").write_text(
        "\n".join(json.dumps(t) for t in hold_tasks) + "\n"
    )
    (args.out / "twin_queries_all.jsonl").write_text(
        "\n".join(json.dumps(q) for q in twin_all) + "\n"
    )

    summary = {
        "n_trees": len(metas),
        "train_trees": train_ids,
        "holdout_trees": sorted(holdout),
        "n_place_train": len(train_tasks),
        "n_place_holdout": len(hold_tasks),
        "n_twin_queries": len(twin_all),
        "n_facts_total": sum(m["n_facts"] for m in metas),
        "seed": args.seed,
        "variant": "hard_paraphrase_partition",
        "leak": leak,
        "lock": "B research + D synth-train; hard text anti-leak; user-dir transfer later",
        "trees": metas,
    }
    (args.out / "META.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "trees"}, indent=2))
    if leak["exact_text_overlap_train_holdout"] != 0:
        print("WARN: exact text leak across split", file=sys.stderr)


if __name__ == "__main__":
    main()
