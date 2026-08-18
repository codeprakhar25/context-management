#!/usr/bin/env python3
"""Multi-tree synth corpus (research placer — lock B+D).

~10 smoke / ~50 mid trees, each with its own root vocabulary (not always work/personal/inbox).
Per tree:
  - gold place tasks (exact path labels for later SFT)
  - confusable twins across sibling folders (subtree vs flat)
  - HierStore sqlite + existing_dirs snapshot

Trees split train/holdout by tree id (whole-tree holdout).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.store import HierStore, Op, path_key  # noqa: E402

# Each layout: tree_id, roots, areas under first "active" root for twins
LAYOUTS = [
    {
        "id": "lab_v0",
        "roots": ["lab", "life", "dump"],
        "twin_root": "lab",
        "areas": ["vision", "nlp", "systems"],
        "life_areas": ["health", "finance"],
    },
    {
        "id": "studio_v0",
        "roots": ["studio", "home", "inbox"],
        "twin_root": "studio",
        "areas": ["design", "code", "ops"],
        "life_areas": ["family", "errands"],
    },
    {
        "id": "clinic_v0",
        "roots": ["clinic", "admin", "scratch"],
        "twin_root": "clinic",
        "areas": ["cards", "labs", "billing"],
        "life_areas": ["hr", "facilities"],
    },
    {
        "id": "shop_v0",
        "roots": ["shop", "personal", "misc"],
        "twin_root": "shop",
        "areas": ["inventory", "support", "growth"],
        "life_areas": ["taxes", "notes"],
    },
    {
        "id": "game_v0",
        "roots": ["game", "team", "backlog"],
        "twin_root": "game",
        "areas": ["engine", "art", "liveops"],
        "life_areas": ["hiring", "legal"],
    },
    {
        "id": "news_v0",
        "roots": ["newsroom", "desk", "wire"],
        "twin_root": "newsroom",
        "areas": ["politics", "tech", "sports"],
        "life_areas": ["style", "ops"],
    },
    {
        "id": "robot_v0",
        "roots": ["robotics", "field", "inbox"],
        "twin_root": "robotics",
        "areas": ["perception", "control", "sim"],
        "life_areas": ["safety", "logistics"],
    },
    {
        "id": "edu_v0",
        "roots": ["course", "research", "inbox"],
        "twin_root": "course",
        "areas": ["lectures", "homework", "exams"],
        "life_areas": ["grants", "students"],
    },
    {
        "id": "fintech_v0",
        "roots": ["product", "risk", "dump"],
        "twin_root": "product",
        "areas": ["payments", "ledger", "fraud"],
        "life_areas": ["compliance", "people"],
    },
    {
        "id": "kitchen_v0",
        "roots": ["kitchen", "front", "notes"],
        "twin_root": "kitchen",
        "areas": ["menu", "suppliers", "prep"],
        "life_areas": ["scheduling", "repairs"],
    },
    {
        "id": "legal_v0",
        "roots": ['matters', 'admin', 'inbox'],
        "twin_root": "matters",
        "areas": ['litigation', 'contracts', 'ip'],
        "life_areas": ['billing', 'conflicts'],
    },
    {
        "id": "farm_v0",
        "roots": ['crops', 'barn', 'notes'],
        "twin_root": "crops",
        "areas": ['corn', 'cattle', 'equipment'],
        "life_areas": ['weather', 'supplies'],
    },
    {
        "id": "museum_v0",
        "roots": ['exhibits', 'ops', 'vault'],
        "twin_root": "exhibits",
        "areas": ['modern', 'ancient', 'kids'],
        "life_areas": ['security', 'donors'],
    },
    {
        "id": "airline_v0",
        "roots": ['flight', 'ground', 'misc'],
        "twin_root": "flight",
        "areas": ['crew', 'routes', 'safety'],
        "life_areas": ['union', 'facilities'],
    },
    {
        "id": "biotech_v0",
        "roots": ['lab', 'clinic', 'dump'],
        "twin_root": "lab",
        "areas": ['assays', 'animals', 'compounds'],
        "life_areas": ['irb', 'vendors'],
    },
    {
        "id": "podcast_v0",
        "roots": ['show', 'biz', 'inbox'],
        "twin_root": "show",
        "areas": ['episodes', 'guests', 'audio'],
        "life_areas": ['ads', 'legal'],
    },
    {
        "id": "nonprofit_v0",
        "roots": ['programs', 'ops', 'scratch'],
        "twin_root": "programs",
        "areas": ['youth', 'housing', 'food'],
        "life_areas": ['grants', 'board'],
    },
    {
        "id": "construction_v0",
        "roots": ['sites', 'office', 'misc'],
        "twin_root": "sites",
        "areas": ['plumbing', 'electrical', 'framing'],
        "life_areas": ['permits', 'hr'],
    },
    {
        "id": "sports_v0",
        "roots": ['team', 'front', 'inbox'],
        "twin_root": "team",
        "areas": ['offense', 'defense', 'scouting'],
        "life_areas": ['travel', 'media'],
    },
    {
        "id": "pharmacy_v0",
        "roots": ['rx', 'store', 'notes'],
        "twin_root": "rx",
        "areas": ['compounding', 'retail', 'delivery'],
        "life_areas": ['compliance', 'inventory'],
    },
    {
        "id": "architecture_v0",
        "roots": ['projects', 'studio', 'inbox'],
        "twin_root": "projects",
        "areas": ['residential', 'civic', 'interiors'],
        "life_areas": ['bids', 'codes'],
    },
    {
        "id": "security_v0",
        "roots": ['soc', 'eng', 'dump'],
        "twin_root": "soc",
        "areas": ['detect', 'respond', 'threatintel'],
        "life_areas": ['policy', 'vendors'],
    },
    {
        "id": "energy_v0",
        "roots": ['plant', 'grid', 'misc'],
        "twin_root": "plant",
        "areas": ['turbine', 'storage', 'control'],
        "life_areas": ['safety', 'reg'],
    },
    {
        "id": "fashion_v0",
        "roots": ['line', 'atelier', 'inbox'],
        "twin_root": "line",
        "areas": ['design', 'fabric', 'lookbook'],
        "life_areas": ['wholesale', 'pr'],
    },
    {
        "id": "film_v0",
        "roots": ['prod', 'post', 'notes'],
        "twin_root": "prod",
        "areas": ['camera', 'sound', 'set'],
        "life_areas": ['casting', 'legal'],
    },
    {
        "id": "hotel_v0",
        "roots": ['rooms', 'fnb', 'misc'],
        "twin_root": "rooms",
        "areas": ['front', 'housekeeping', 'events'],
        "life_areas": ['purchasing', 'hr'],
    },
    {
        "id": "transit_v0",
        "roots": ['rail', 'bus', 'inbox'],
        "twin_root": "rail",
        "areas": ['signals', 'rolling', 'stations'],
        "life_areas": ['schedules', 'safety'],
    },
    {
        "id": "insurance_v0",
        "roots": ['claims', 'uw', 'dump'],
        "twin_root": "claims",
        "areas": ['auto', 'home', 'life'],
        "life_areas": ['fraud', 'actuarial'],
    },
    {
        "id": "space_v0",
        "roots": ['mission', 'ground', 'scratch'],
        "twin_root": "mission",
        "areas": ['avionics', 'payload', 'ops'],
        "life_areas": ['range', 'comms'],
    },
    {
        "id": "music_v0",
        "roots": ['label', 'tour', 'inbox'],
        "twin_root": "label",
        "areas": ['aandr', 'mastering', 'promo'],
        "life_areas": ['royalties', 'tour'],
    },
    {
        "id": "civic_v0",
        "roots": ['city', 'public', 'notes'],
        "twin_root": "city",
        "areas": ['zoning', 'transit', 'parks'],
        "life_areas": ['budget', 'council'],
    },
    {
        "id": "mining_v0",
        "roots": ['pit', 'plant', 'misc'],
        "twin_root": "pit",
        "areas": ['drill', 'haul', 'process'],
        "life_areas": ['safety', 'env'],
    },
    {
        "id": "marine_v0",
        "roots": ['fleet', 'port', 'inbox'],
        "twin_root": "fleet",
        "areas": ['cargo', 'crew', 'maint'],
        "life_areas": ['customs', 'fuel'],
    },
    {
        "id": "telecom_v0",
        "roots": ['network', 'care', 'dump'],
        "twin_root": "network",
        "areas": ['core', 'radio', 'fiber'],
        "life_areas": ['sla', 'vendors'],
    },
    {
        "id": "agtech_v0",
        "roots": ['fields', 'lab', 'notes'],
        "twin_root": "fields",
        "areas": ['sensors', 'irrigation', 'yield'],
        "life_areas": ['seed', 'finance'],
    },
    {
        "id": "esports_v0",
        "roots": ['org', 'content', 'inbox'],
        "twin_root": "org",
        "areas": ['players', 'coaching', 'analytics'],
        "life_areas": ['sponsors', 'travel'],
    },
    {
        "id": "bakery_v0",
        "roots": ['bake', 'front', 'misc'],
        "twin_root": "bake",
        "areas": ['bread', 'pastry', 'cakes'],
        "life_areas": ['orders', 'suppliers'],
    },
    {
        "id": "zoo_v0",
        "roots": ['animals', 'ops', 'notes'],
        "twin_root": "animals",
        "areas": ['mammals', 'birds', 'reptiles'],
        "life_areas": ['vet', 'education'],
    },
    {
        "id": "library_v0",
        "roots": ['stacks', 'programs', 'inbox'],
        "twin_root": "stacks",
        "areas": ['fiction', 'ref', 'archives'],
        "life_areas": ['events', 'acq'],
    },
    {
        "id": "court_v0",
        "roots": ['dockets', 'clerk', 'misc'],
        "twin_root": "dockets",
        "areas": ['civil', 'criminal', 'family'],
        "life_areas": ['jury', 'records'],
    },
    {
        "id": "weather_v0",
        "roots": ['forecast', 'obs', 'inbox'],
        "twin_root": "forecast",
        "areas": ['models', 'radar', 'alerts'],
        "life_areas": ['climate', 'outreach'],
    },
    {
        "id": "drone_v0",
        "roots": ['fleet', 'ops', 'dump'],
        "twin_root": "fleet",
        "areas": ['mapping', 'delivery', 'inspect'],
        "life_areas": ['faa', 'battery'],
    },
    {
        "id": "brewing_v0",
        "roots": ['brew', 'tap', 'notes'],
        "twin_root": "brew",
        "areas": ['lager', 'ipa', 'sour'],
        "life_areas": ['inventory', 'events'],
    },
    {
        "id": "railfreight_v0",
        "roots": ['yard', 'dispatch', 'misc'],
        "twin_root": "yard",
        "areas": ['intermodal', 'bulk', 'reefer'],
        "life_areas": ['safety', 'crew'],
    },
    {
        "id": "semicon_v0",
        "roots": ['fab', 'test', 'inbox'],
        "twin_root": "fab",
        "areas": ['litho', 'etch', 'metrology'],
        "life_areas": ['yield', 'ehns'],
    },
    {
        "id": "ngo_aid_v0",
        "roots": ['field', 'hq', 'scratch'],
        "twin_root": "field",
        "areas": ['wash', 'health', 'shelter'],
        "life_areas": ['logistics', 'donors'],
    },
    {
        "id": "theater_v0",
        "roots": ['stage', 'box', 'inbox'],
        "twin_root": "stage",
        "areas": ['sets', 'lighting', 'costume'],
        "life_areas": ['casting', 'sponsors'],
    },
    {
        "id": "vet_v0",
        "roots": ['clinic', 'kennel', 'notes'],
        "twin_root": "clinic",
        "areas": ['surgery', 'dental', 'imaging'],
        "life_areas": ['pharmacy', 'billing'],
    },
    {
        "id": "cycling_v0",
        "roots": ['race', 'shop', 'misc'],
        "twin_root": "race",
        "areas": ['road', 'mtb', 'track'],
        "life_areas": ['sponsors', 'mech'],
    },
    {
        "id": "cloud_v0",
        "roots": ['platform', 'sre', 'dump'],
        "twin_root": "platform",
        "areas": ['compute', 'storage', 'network'],
        "life_areas": ['billing', 'security'],
    },
]

# Templates for unique (non-twin) facts — {area} filled
UNIQUE = [
    "Weekly sync notes for {area}: blockers and owners",
    "Checklist before shipping {area} changes to prod",
    "Onboarding doc: how we run {area} reviews",
    "Retrospective themes from last {area} sprint",
    "Budget line items tied to {area} this quarter",
]

# Near-dup twins — same body, thin area tag optional
SHARED_TWIN = [
    ("deploy", "Deploy failed on Friday night after the config change"),
    ("quota", "Hit GPU quota mid-run; jobs paused until Monday"),
    ("metric", "Primary metric dropped after the last merge"),
    ("meeting", "Ship the ablation before the freeze next week"),
    ("bug", "NaNs appeared when batching long inputs"),
    ("bill", "Cloud bill spiked after leaving the pod warm overnight"),
]

IDENTICAL_TWIN = [
    ("alert", "PagerDuty fired twice during the overnight sweep"),
    ("action", "Action item: ping infra about disk pressure"),
]


def fid(tree_id: str, *parts: str) -> str:
    h = hashlib.sha1(f"{tree_id}|{'|'.join(parts)}".encode()).hexdigest()[:10]
    return f"{tree_id}_{h}"


def build_tree(layout: dict, out_dir: Path, rng: random.Random) -> dict:
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
    # ensure area folders
    for a in areas:
        ops.append(Op(op="MKDIR", path=[twin_root, a]))
        ops.append(Op(op="MKDIR", path=[twin_root, a, "results"]))

    # life / secondary root filler
    life_root = roots[1]
    for a in layout["life_areas"]:
        ops.append(Op(op="MKDIR", path=[life_root, a]))
        text = f"Note under {life_root}/{a}: routine reminder {rng.randint(1, 99)}"
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

    # dump/inbox root — unsure bucket
    dump_root = roots[2]
    ops.append(Op(op="MKDIR", path=[dump_root]))
    dump_text = f"asdf scratch {tid} {rng.randint(1000, 9999)}"
    dump_id = fid(tid, "dump", dump_text)
    ops.append(
        Op(op="ADD", fact_id=dump_id, text=dump_text, path=[dump_root])
    )
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

    # unique facts per area
    for a in areas:
        for i, tmpl in enumerate(UNIQUE):
            text = tmpl.format(area=a)
            path = [twin_root, a] if i % 2 == 0 else [twin_root, a, "results"]
            fact_id = fid(tid, "uniq", a, str(i), text)
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

    # twins across first two areas (confusable)
    a0, a1 = areas[0], areas[1]
    for tname, body in SHARED_TWIN:
        for a in (a0, a1):
            text = f"{body} [{a}]"
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

    for tname, body in IDENTICAL_TWIN:
        for a in (a0, a1):
            path = [twin_root, a, "results"]
            fact_id = fid(tid, "ident", a, tname)
            ops.append(Op(op="ADD", fact_id=fact_id, text=body, path=path))
            # no place_task: identical text ⇒ ambiguous gold without active_path
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

    store.apply_ops(ops, manager="multitree_synth")
    existing_dirs = store.list_dirs()
    snap = store.snapshot()
    store.close()

    # attach existing_dirs to place tasks
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
        "n_facts": snap["valid_count"],
        "n_dirs": len(existing_dirs),
        "n_place_tasks": len(place_tasks),
        "n_twin_queries": len(twin_queries),
        "twin_areas": [a0, a1],
        "db": str(db),
    }
    (tdir / "META.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "multitree_synth_mid",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-holdout", type=int, default=10, help="whole trees for holdout")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    metas = []
    for layout in LAYOUTS:
        metas.append(build_tree(layout, args.out, rng))

    ids = [m["tree_id"] for m in metas]
    holdout = set(ids[-args.n_holdout :])  # last N = holdout
    train_ids = [i for i in ids if i not in holdout]

    # aggregate place tasks
    train_tasks, hold_tasks = [], []
    twin_all = []
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
        "lock": "B research + D synth-train / soft+subtree eval; user-dir transfer later",
        "trees": metas,
    }
    (args.out / "META.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "trees"}, indent=2))


if __name__ == "__main__":
    main()
