#!/usr/bin/env python3
"""Majority-class and BM25-kNN placement baselines. No API calls.

BM25-kNN is the lexical analogue of the embedding kNN arm: retrieve the k
nearest training notes and vote their folders, weighted by score.
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


def majority_pred(train: list[dict]) -> list[str]:
    counts = collections.Counter(path_key(t["gold_path"]) for t in train)
    return parse_key(counts.most_common(1)[0][0])


def bm25_preds(train: list[dict], val: list[dict], k: int) -> list[list[str]]:
    facts = [
        {"id": t.get("id", str(i)), "text": t["text"], "path": t["gold_path"]}
        for i, t in enumerate(train)
    ]
    index = BM25Index(facts)
    out = []
    for v in val:
        hits = index.retrieve_flat(v["text"], k)
        votes: dict[str, float] = collections.defaultdict(float)
        for h in hits:
            if h.score <= 0:
                continue
            votes[path_key(h.path)] += h.score
        if not votes:
            out.append([])
            continue
        out.append(parse_key(max(votes.items(), key=lambda x: x[1])[0]))
    return out


def eval_split(train: list[dict], val: list[dict], k: int) -> dict:
    maj = majority_pred(train) if train else []
    bm25 = bm25_preds(train, val, k) if train else [[] for _ in val]
    support = collections.Counter(path_key(t["gold_path"]) for t in train)
    rows = []
    for v, pred in zip(val, bm25):
        gold = v["gold_path"]
        s = support[path_key(gold)]
        rows.append(
            {
                "id": v["id"],
                "support": s,
                "bucket": bucket(s),
                "majority_exact": gold == maj,
                "bm25_exact": gold == pred,
            }
        )
    n = len(rows) or 1
    by_b: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        by_b[r["bucket"]].append(r)
    return {
        "n": len(rows),
        "majority": sum(r["majority_exact"] for r in rows) / n,
        "bm25": sum(r["bm25_exact"] for r in rows) / n,
        "by_bucket": {
            b: {
                "n": len(rs),
                "majority": sum(r["majority_exact"] for r in rs) / len(rs),
                "bm25": sum(r["bm25_exact"] for r in rs) / len(rs),
            }
            for b, rs in sorted(by_b.items())
        },
    }


def pool(parts: list[dict]) -> dict:
    n = sum(p["n"] for p in parts)
    maj_hits = sum(p["majority"] * p["n"] for p in parts)
    bm_hits = sum(p["bm25"] * p["n"] for p in parts)
    buckets: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    for p in parts:
        for b, st in p["by_bucket"].items():
            buckets[b][0] += st["n"]
            buckets[b][1] += st["majority"] * st["n"]
            buckets[b][2] += st["bm25"] * st["n"]
    return {
        "n": n,
        "majority": maj_hits / n if n else 0.0,
        "bm25": bm_hits / n if n else 0.0,
        "by_bucket": {
            b: {
                "n": c[0],
                "majority": c[1] / c[0],
                "bm25": c[2] / c[0],
            }
            for b, c in sorted(buckets.items())
        },
    }


def vault_dirs(root: Path) -> list[Path]:
    return sorted(
        d for d in root.iterdir() if d.is_dir() and (d / "val.jsonl").exists()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "lexical_baselines")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report: dict = {"k": args.k}

    a_train = load_jsonl(ROOT / "data/user_dir_snap_v2/train.jsonl")
    a_val = load_jsonl(ROOT / "data/user_dir_snap_v2/val.jsonl")
    report["A_item"] = eval_split(a_train, a_val, args.k)
    print("A item", json.dumps(report["A_item"], indent=2))

    b_parts = []
    for d in vault_dirs(ROOT / "data/vaults_build"):
        b_parts.append(
            eval_split(load_jsonl(d / "train.jsonl"), load_jsonl(d / "val.jsonl"), args.k)
        )
    report["B_item"] = pool(b_parts)
    print("B item", json.dumps(report["B_item"], indent=2))

    ap_parts = []
    aprime = ROOT / "data/vaultsA_build"
    if aprime.exists():
        for d in vault_dirs(aprime):
            ap_parts.append(
                eval_split(
                    load_jsonl(d / "train.jsonl"), load_jsonl(d / "val.jsonl"), args.k
                )
            )
        report["Aprime_item"] = pool(ap_parts)
        print("A' item", json.dumps(report["Aprime_item"], indent=2))

    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", args.out / "summary.json")


if __name__ == "__main__":
    main()
