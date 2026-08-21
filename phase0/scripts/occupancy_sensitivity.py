#!/usr/bin/env python3
"""Occupancy-bucket sensitivity check for corpus B, item-stratified split.

Rebuilds the occupancy table from per-item run artifacts, then re-runs it with
outlier vaults excluded, to test whether the method crossing in Figure 1 depends
on a small number of pathological vaults.

Also reports the cascade's parent-prediction ("pred_prefix") rate by bucket,
which is the suspected mechanism behind its dense-bucket collapse.

No API calls; reads only files already on disk.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.bm25 import BM25Index  # noqa: E402

BUILD = ROOT / "data" / "vaults_build"
RUNS = ROOT / "runs"

# per-item run artifacts: arm -> (run dir, per-vault subdir suffix)
ARMS = {
    "flat": ("vaultB_flat", "__item"),
    "kNN": ("vaultB_knn_k5only", "__item"),
    "cascade": ("vaultB_cascade", "__item_note20"),
    "LoRA": ("vaultB_lora_item", ""),
}

# vaults called out in PLACER_FINDINGS.md as driving the cascade's item-pool loss
OUTLIERS = ["TheRoadOfSO", "anthonyamar", "ManadayM"]

BUCKET_ORDER = ["1-2", "3-9", "10+"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def path_key(p: list[str]) -> str:
    return "/" + "/".join(p)


def parse_key(k: str) -> list[str]:
    return [s for s in k.split("/") if s]


def bucket(n: int) -> str:
    if n <= 0:
        return "0"
    if n <= 2:
        return "1-2"
    if n <= 9:
        return "3-9"
    return "10+"


def vault_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "val.jsonl").exists())


def bm25_preds(train: list[dict], val: list[dict], k: int) -> list[list[str]]:
    facts = [
        {"id": t.get("id", str(i)), "text": t["text"], "path": t["gold_path"]}
        for i, t in enumerate(train)
    ]
    index = BM25Index(facts)
    out = []
    for v in val:
        votes: dict[str, float] = collections.defaultdict(float)
        for h in index.retrieve_flat(v["text"], k):
            if h.score > 0:
                votes[path_key(h.path)] += h.score
        out.append(parse_key(max(votes.items(), key=lambda x: x[1])[0]) if votes else [])
    return out


def collect(k: int) -> list[dict]:
    """One row per corpus-B item-split validation item, with every arm's verdict."""
    rows: list[dict] = []
    for d in vault_dirs(BUILD):
        vault = d.name
        train, val = load_jsonl(d / "train.jsonl"), load_jsonl(d / "val.jsonl")
        support = collections.Counter(path_key(t["gold_path"]) for t in train)

        majority = (
            parse_key(
                collections.Counter(
                    path_key(t["gold_path"]) for t in train
                ).most_common(1)[0][0]
            )
            if train
            else []
        )
        bm25 = bm25_preds(train, val, k) if train else [[] for _ in val]

        # per-arm verdicts, keyed by item id
        arm_exact: dict[str, dict[str, bool]] = {}
        arm_rel: dict[str, dict[str, str]] = {}
        for arm, (run, suffix) in ARMS.items():
            f = RUNS / run / f"{vault}{suffix}" / "holdout_results.jsonl"
            if not f.exists():
                continue
            recs = load_jsonl(f)
            arm_exact[arm] = {r["id"]: bool(r.get("exact")) for r in recs}
            arm_rel[arm] = {r["id"]: r.get("soft_relation", "") for r in recs}

        for v, bpred in zip(val, bm25):
            s = support[path_key(v["gold_path"])]
            row = {
                "vault": vault,
                "id": v["id"],
                "support": s,
                "bucket": bucket(s),
                "majority": v["gold_path"] == majority,
                "BM25": v["gold_path"] == bpred,
                "cascade_rel": arm_rel.get("cascade", {}).get(v["id"], ""),
            }
            for arm in ARMS:
                got = arm_exact.get(arm, {}).get(v["id"])
                row[arm] = got
            rows.append(row)
    return rows


def table(rows: list[dict], arms: list[str]) -> dict:
    by_b: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_b[r["bucket"]].append(r)

    def acc(rs: list[dict], arm: str):
        vals = [r[arm] for r in rs if r[arm] is not None]
        return (round(sum(vals) / len(vals), 4), len(vals)) if vals else (None, 0)

    out = {}
    for b in BUCKET_ORDER + ["all"]:
        rs = rows if b == "all" else by_b.get(b, [])
        if not rs:
            continue
        out[b] = {"n": len(rs), **{a: acc(rs, a)[0] for a in arms}}
    return out


def fmt(title: str, tab: dict, arms: list[str]) -> str:
    head = f"{'occ':<6}{'n':>7}" + "".join(f"{a:>10}" for a in arms)
    lines = [title, "-" * len(head), head]
    for b, row in tab.items():
        cells = "".join(
            f"{row[a]:>10.3f}" if row.get(a) is not None else f"{'--':>10}" for a in arms
        )
        lines.append(f"{b:<6}{row['n']:>7}{cells}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", type=Path, default=RUNS / "occupancy_sensitivity")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = collect(args.k)
    arms = ["majority", "flat", "BM25", "kNN", "cascade", "LoRA"]

    full = table(rows, arms)
    kept = [r for r in rows if not any(o in r["vault"] for o in OUTLIERS)]
    dropped = sorted({r["vault"] for r in rows} - {r["vault"] for r in kept})
    trimmed = table(kept, arms)

    print(fmt("ALL 27 VAULTS", full, arms))
    print()
    print(fmt(f"EXCLUDING {len(dropped)} OUTLIER VAULTS: {', '.join(dropped)}", trimmed, arms))

    # cascade failure-mode breakdown: how often is the prediction the gold's parent?
    print("\nCASCADE FAILURE MODE (share of items where pred is a prefix of gold)")
    print(f"{'occ':<6}{'n':>7}{'pred_prefix':>13}{'all vaults':>12}{'ex-outlier':>12}")
    for b in BUCKET_ORDER:
        a = [r for r in rows if r["bucket"] == b and r["cascade_rel"]]
        k_ = [r for r in kept if r["bucket"] == b and r["cascade_rel"]]
        pa = sum(r["cascade_rel"] == "pred_prefix" for r in a) / len(a) if a else 0
        pk = sum(r["cascade_rel"] == "pred_prefix" for r in k_) / len(k_) if k_ else 0
        print(f"{b:<6}{len(a):>7}{'':>13}{pa:>12.3f}{pk:>12.3f}")

    # per-vault cascade vs kNN in the dense bucket, to show the spread
    print("\nDENSE BUCKET (10+), per-vault cascade vs kNN, worst 8 by margin")
    dense = collections.defaultdict(list)
    for r in rows:
        if r["bucket"] == "10+":
            dense[r["vault"]].append(r)
    marg = []
    for v, rs in dense.items():
        c = [r["cascade"] for r in rs if r["cascade"] is not None]
        kk = [r["kNN"] for r in rs if r["kNN"] is not None]
        if c and kk:
            marg.append((sum(c) / len(c) - sum(kk) / len(kk), v, len(rs), sum(c) / len(c), sum(kk) / len(kk)))
    for m, v, n, c, kk in sorted(marg)[:8]:
        print(f"  {v:<42} n={n:<4} cascade={c:.3f}  kNN={kk:.3f}  margin={m:+.3f}")

    payload = {
        "k": args.k,
        "outliers_excluded": dropped,
        "all_vaults": full,
        "excluding_outliers": trimmed,
        "n_items_all": len(rows),
        "n_items_kept": len(kept),
    }
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nwrote", args.out / "summary.json")


if __name__ == "__main__":
    main()
