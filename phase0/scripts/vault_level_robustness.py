#!/usr/bin/env python3
"""Vault-level robustness checks for the occupancy tables (corpus B, corpus A').

Answers the reviewer points that need no new model calls:

  1. Vault-clustered bootstrap intervals. Items within a vault are not
     independent, so intervals resample whole vaults, not items. Reported per
     arm and occupancy bucket, plus paired (cascade - kNN) and (LoRA - kNN)
     differences and the per-vault sign of the sparse/dense crossing.
  2. Soft metric alongside exact match: a prediction that is the gold folder's
     parent or child counts as a soft hit.
  3. kNN vote rules. A fixed k=5 vote lets a dense folder contribute up to five
     neighbours and a 1-note folder at most one. Re-scored with k=1 and with a
     nearest-folder-centroid rule, both of which remove that size advantage.
  4. Path-only baseline on the folder-disjoint split: the top-1 folder by
     path-string embedding, no LLM. The gap to the cascade is what the picker
     adds over its own retriever.
  5. Sensitivity to the vaults whose notes were capped at 250 in file order.

Reads only files already on disk. Embeddings come from the candidate-recall
cache; a cache miss is an error, so this script never calls an API.
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from candidate_recall import path_text  # noqa: E402
from knn_placer_baseline import soft_score  # noqa: E402
from occupancy_sensitivity import (  # noqa: E402
    OUTLIERS,
    bm25_preds,
    bucket,
    load_jsonl,
    path_key,
    vault_dirs,
)

RUNS = ROOT / "runs"
CACHE = RUNS / "_embed_cache" / "candidate_recall.json"
BUCKETS = ["1-2", "3-9", "10+", "all"]
K_GRID = [1, 3, 5, 8, 12]

CORPORA = {
    "B": {
        "build": ROOT / "data" / "vaults_build",
        "item_arms": {
            "flat": ("vaultB_flat", "__item"),
            "cascade": ("vaultB_cascade", "__item_note20"),
            "cascade_llama": ("vaultB_cascade_llama70b", "__item_note20"),
            "LoRA": ("vaultB_lora_item", ""),
            # follow-up runs R4, R5 (PLACER_FINDINGS.md, pre-specified 2026-10-01)
            "cascade_member": ("vaultB_cascade_mem", "__item_note20_mem2"),
            "LoRA_pervault_e1": ("vaultB_pervault_e1", ""),
            "LoRA_pervault_e3": ("vaultB_pervault_e3", ""),
            "LoRA_pooled_reserved": ("vaultB_pooled_reserved", ""),
        },
        "fold_arms": {
            "flat": ("vaultB_flat", "__folder"),
            "cascade": ("vaultB_cascade", "__folder_path20"),
            "cascade_llama": ("vaultB_cascade_llama70b", "__folder_path20"),
            "LoRA": ("vaultB_lora_fold", ""),
            "desc_top1": ("vaultB_fold_desc_top1", ""),
        },
    },
    "A_prime": {
        "build": ROOT / "data" / "vaultsA_build",
        "item_arms": {
            "cascade": ("vaultsA_cascade", "__item_note20"),
            "cascade_llama": ("vaultsA_cascade_llama70b", "__item_note20"),
            "LoRA": ("vaultsA_lora_item", ""),
            "flat": ("vaultsA_flat", "__item"),
            "LoRA_pervault_e1": ("vaultsA_pervault_e1", ""),
            "LoRA_pervault_e3": ("vaultsA_pervault_e3", ""),
            "LoRA_pooled_reserved": ("vaultsA_pooled_reserved", ""),
        },
        # R1: path@50 is the primary shortlist under the pre-specified recall rule
        "fold_arms": {
            "flat": ("vaultsA_flat", "__folder"),
            "cascade": ("vaultsA_cascade", "__folder_path50"),
            "cascade_path20": ("vaultsA_cascade", "__folder_path20"),
            "cascade_llama": ("vaultsA_cascade_llama70b", "__folder_path50"),
            "LoRA": ("vaultsA_lora_fold", ""),
            "desc_top1": ("vaultsA_fold_desc_top1", ""),
        },
    },
}


class CachedEmbeddings:
    """Read-only view of the candidate-recall embedding cache."""

    def __init__(self, path: Path):
        self.cache: dict[str, list[float]] = json.loads(path.read_text())

    def __call__(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self.cache]
        if missing:
            raise KeyError(f"{len(missing)} texts not in embedding cache; refusing to call the API")
        a = np.asarray([self.cache[t] for t in texts], dtype=np.float32)
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


def knn_variants(train: list[dict], val: list[dict], emb: CachedEmbeddings) -> dict[str, list[list[str]]]:
    tv, vv = emb([t["text"] for t in train]), emb([v["text"] for v in val])
    sim = vv @ tv.T
    order = np.argsort(-sim, axis=1)
    keys = [path_key(t["gold_path"]) for t in train]

    def weighted(k: int) -> list[list[str]]:
        out = []
        for i in range(len(val)):
            votes: dict[str, float] = collections.defaultdict(float)
            for j in order[i][:k]:
                votes[keys[j]] += float(sim[i][j])
            out.append([s for s in max(votes.items(), key=lambda x: x[1])[0].split("/") if s])
        return out

    # nearest folder centroid: each folder is one point regardless of size
    folders = sorted(set(keys))
    cent = np.stack([tv[[j for j, k in enumerate(keys) if k == f]].mean(axis=0) for f in folders])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9
    nearest = np.argmax(vv @ cent.T, axis=1)
    centroid = [[s for s in folders[c].split("/") if s] for c in nearest]

    out = {f"kNN_k{k}": weighted(k) for k in K_GRID}
    out["kNN"] = out["kNN_k5"]
    out["centroid"] = centroid
    return out


def load_arm(run: str, vault: str, suffix: str) -> dict[str, dict] | None:
    f = RUNS / run / f"{vault}{suffix}" / "holdout_results.jsonl"
    if not f.exists():
        return None
    return {r["id"]: r for r in load_jsonl(f)}


def verdict(gold: list[str], pred: list[str] | None) -> tuple[bool, bool]:
    if pred is None:
        return False, False
    s = soft_score(gold, pred)
    return s["exact"], s["soft_hit"]


def item_rows(corpus: str, emb: CachedEmbeddings) -> list[dict]:
    cfg = CORPORA[corpus]
    rows = []
    for d in vault_dirs(cfg["build"]):
        vault = d.name
        train, val = load_jsonl(d / "train.jsonl"), load_jsonl(d / "val.jsonl")
        support = collections.Counter(path_key(t["gold_path"]) for t in train)
        majority = [s for s in support.most_common(1)[0][0].split("/") if s]
        preds = {"majority": [majority] * len(val), "BM25": bm25_preds(train, val, 5)}
        preds.update(knn_variants(train, val, emb))
        stored = {a: load_arm(run, vault, sfx) for a, (run, sfx) in cfg["item_arms"].items()}
        for i, v in enumerate(val):
            row = {"vault": vault, "bucket": bucket(support[path_key(v["gold_path"])])}
            for arm, p in preds.items():
                row[arm], row[arm + "~soft"] = verdict(v["gold_path"], p[i])
            for arm, recs in stored.items():
                if recs is None or v["id"] not in recs:
                    row[arm] = row[arm + "~soft"] = None
                else:
                    row[arm], row[arm + "~soft"] = verdict(v["gold_path"], recs[v["id"]]["pred_path"])
            rows.append(row)
    return rows


def fold_rows(corpus: str, emb: CachedEmbeddings) -> list[dict]:
    cfg = CORPORA[corpus]
    rows = []
    for d in vault_dirs(cfg["build"]):
        if not (d / "fold_val.jsonl").exists():
            continue
        vault = d.name
        val = load_jsonl(d / "fold_val.jsonl")
        con = sqlite3.connect(d / "hierstore.sqlite")
        dirs = [json.loads(r[0]) for r in con.execute("select path_json from dirs")]
        con.close()
        vv, dv = emb([v["text"] for v in val]), emb([path_text(p) for p in dirs])
        top1 = np.argmax(vv @ dv.T, axis=1)
        stored = {a: load_arm(run, vault, sfx) for a, (run, sfx) in cfg["fold_arms"].items()}
        for i, v in enumerate(val):
            row = {"vault": vault, "bucket": "0"}
            row["path_top1"], row["path_top1~soft"] = verdict(v["gold_path"], dirs[top1[i]])
            for arm, recs in stored.items():
                if recs is None or v["id"] not in recs:
                    row[arm] = row[arm + "~soft"] = None
                else:
                    row[arm], row[arm + "~soft"] = verdict(v["gold_path"], recs[v["id"]]["pred_path"])
            rows.append(row)
    return rows


def per_vault(rows: list[dict], arm: str) -> dict[str, tuple[int, int]]:
    """vault -> (hits, n) over rows where the arm has a verdict."""
    out: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get(arm) is not None:
            out[r["vault"]][0] += int(r[arm])
            out[r["vault"]][1] += 1
    return {v: (h, n) for v, (h, n) in out.items()}


PAIRS = [("cascade", "kNN"), ("cascade", "centroid"), ("LoRA", "kNN"), ("cascade", "LoRA"), ("cascade", "path_top1"),
         ("cascade", "flat"), ("flat", "LoRA"), ("desc_top1", "path_top1"),
         ("cascade_member", "cascade"), ("cascade_member", "kNN"), ("cascade_member", "LoRA"),
         ("LoRA_pervault_e1", "LoRA_pooled_reserved"), ("LoRA_pervault_e3", "LoRA_pooled_reserved"),
         ("LoRA_pervault_e3", "cascade"), ("LoRA_pooled_reserved", "LoRA")]


def cluster_bootstrap(rows: list[dict], arms: list[str], reps: int, rng: np.random.Generator,
                      suffix: str = "") -> dict:
    """Pooled accuracy per arm with 95% percentile intervals from resampling vaults.

    Differences are paired: each replicate uses the same vault draw for both arms,
    restricted to items both arms scored.
    """
    vaults = sorted({r["vault"] for r in rows})
    idx = rng.integers(0, len(vaults), size=(reps, len(vaults)))

    def stat(table: dict[str, tuple[int, int]]) -> dict:
        h = np.array([table.get(v, (0, 0))[0] for v in vaults], dtype=float)
        n = np.array([table.get(v, (0, 0))[1] for v in vaults], dtype=float)
        if n.sum() == 0:
            return {}
        boot = h[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
        return {"acc": round(h.sum() / n.sum(), 4),
                "ci": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
                "n": int(n.sum())}

    out = {a: stat(per_vault(rows, a)) for a in arms}
    for a, b in ((a + suffix, b + suffix) for a, b in PAIRS):
        both = [r for r in rows if r.get(a) is not None and r.get(b) is not None]
        if not both:
            continue
        ta, tb = per_vault(both, a), per_vault(both, b)
        ha = np.array([ta.get(v, (0, 0))[0] for v in vaults], dtype=float)
        hb = np.array([tb.get(v, (0, 0))[0] for v in vaults], dtype=float)
        n = np.array([ta.get(v, (0, 0))[1] for v in vaults], dtype=float)
        boot = (ha[idx].sum(1) - hb[idx].sum(1)) / np.maximum(n[idx].sum(1), 1)
        out[f"{a}-{b}"] = {"diff": round((ha.sum() - hb.sum()) / n.sum(), 4),
                           "ci": [round(float(np.percentile(boot, 2.5)), 4),
                                  round(float(np.percentile(boot, 97.5)), 4)],
                           "n_vaults": int((n > 0).sum())}
    return out


def by_bucket(rows: list[dict], arms: list[str], reps: int, seed: int) -> dict:
    suffix = "~soft" if arms and arms[0].endswith("~soft") else ""
    out = {}
    for b in BUCKETS:
        rs = rows if b == "all" else [r for r in rows if r["bucket"] == b]
        if rs:
            out[b] = cluster_bootstrap(rs, arms, reps, np.random.default_rng(seed), suffix)
    return out


def crossing_per_vault(rows: list[dict], a: str = "cascade", b: str = "kNN") -> dict:
    """How many vaults show the crossing themselves: a beats b sparse, loses dense."""
    per: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for bk in ("1-2", "10+"):
        rs = [r for r in rows if r["bucket"] == bk and r.get(a) is not None and r.get(b) is not None]
        ta, tb = per_vault(rs, a), per_vault(rs, b)
        for v in ta:
            per[v][bk] = (ta[v][0] - tb[v][0]) / ta[v][1]
    both = {v: d for v, d in per.items() if "1-2" in d and "10+" in d}
    return {
        "vaults_with_both_buckets": len(both),
        "sparse_gap_positive": sum(d["1-2"] > 0 for d in both.values()),
        "dense_gap_negative": sum(d["10+"] < 0 for d in both.values()),
        "full_crossing": sum(d["1-2"] > 0 and d["10+"] < 0 for d in both.values()),
        "reverse_crossing": sum(d["1-2"] < 0 and d["10+"] > 0 for d in both.values()),
        "per_vault": {v: {k: round(x, 3) for k, x in d.items()} for v, d in sorted(both.items())},
    }


def nested_k(rows: list[dict]) -> dict:
    """Leave-one-vault-out choice of k: pick k on the other vaults, score the held-out one.

    Removes the selection-on-reported-data problem for kNN without new data. The
    pooled result is an honest estimate of a kNN whose k is tuned by the same
    procedure on unseen vaults.
    """
    vaults = sorted({r["vault"] for r in rows})
    hits = {k: per_vault(rows, f"kNN_k{k}") for k in K_GRID}
    chosen, nested_rows = {}, []
    for v in vaults:
        score = {k: sum(h for u, (h, _) in hits[k].items() if u != v) for k in K_GRID}
        best = max(K_GRID, key=lambda k: (score[k], -abs(k - 5)))
        chosen[v] = best
        nested_rows += [{**r, "kNN_nested": r[f"kNN_k{best}"]} for r in rows if r["vault"] == v]
    by_b = {}
    for b in BUCKETS:
        rs = nested_rows if b == "all" else [r for r in nested_rows if r["bucket"] == b]
        if rs:
            by_b[b] = {"n": len(rs), "kNN_nested": round(sum(r["kNN_nested"] for r in rs) / len(rs), 4),
                       "kNN_k5": round(sum(r["kNN_k5"] for r in rs) / len(rs), 4),
                       "best_fixed_k": max(K_GRID, key=lambda k: sum(r[f"kNN_k{k}"] for r in rs))}
    return {"k_chosen_per_heldout_vault": collections.Counter(chosen.values()), "by_bucket": by_b,
            "grid_pooled": {k: round(sum(h for h, _ in hits[k].values()) / len(rows), 4) for k in K_GRID}}


def corpus_a_table() -> dict:
    """Corpus A (one private tree) item split, every arm by occupancy bucket.

    One tree, so there is no vault to cluster on; these are item-level rates.
    kNN is recomputed at k=5 because the stored run kept only its first k.
    """
    d = ROOT / "data" / "user_dir_snap_v2"
    train, val = load_jsonl(d / "train.jsonl"), load_jsonl(d / "val.jsonl")
    support = collections.Counter(path_key(t["gold_path"]) for t in train)
    majority = [s for s in support.most_common(1)[0][0].split("/") if s]
    preds = {"majority": [majority] * len(val), "BM25": bm25_preds(train, val, 5)}
    stored = {"flat": "v2_val_gpt4o", "kNN": "v2_val_knn", "cascade": "v2_val_cascade_note20", "LoRA": "v2_val_lora"}
    recs = {a: {r["id"]: r for r in load_jsonl(RUNS / run / "holdout_results.jsonl")} for a, run in stored.items()}
    rows = []
    for i, v in enumerate(val):
        row = {"bucket": bucket(support[path_key(v["gold_path"])])}
        for a, p in preds.items():
            row[a] = v["gold_path"] == p[i]
        for a, r in recs.items():
            row[a] = bool(r[v["id"]]["exact"]) if v["id"] in r else None
        rows.append(row)
    arms = ["majority", "flat", "BM25", "kNN", "cascade", "LoRA"]
    out = {}
    for b in BUCKETS:
        rs = rows if b == "all" else [r for r in rows if r["bucket"] == b]
        out[b] = {"n": len(rs), **{a: round(sum(bool(r[a]) for r in rs) / len(rs), 3) for a in arms}}
    return out


def spearman_perm(x: list[float], y: list[float], reps: int, rng: np.random.Generator) -> tuple[float, float]:
    rank = lambda a: np.argsort(np.argsort(a)).astype(float)  # noqa: E731
    rx, ry = rank(np.array(x)), rank(np.array(y))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(reps)])
    return round(rho, 3), round(float((np.abs(null) >= abs(rho)).mean()), 4)


def structure_correlates(rows: list[dict], build: Path, reps: int, seed: int) -> dict:
    """Do vault structure descriptors predict the per-vault method margins?

    The unit is the vault, as in the retracted tree-size analysis. Descriptors
    come from each vault's training notes and store, not from the outcomes.
    """
    desc = {}
    for d in vault_dirs(build):
        train = load_jsonl(d / "train.jsonl")
        sizes = np.array(sorted(collections.Counter(path_key(t["gold_path"]) for t in train).values()))
        con = sqlite3.connect(d / "hierstore.sqlite")
        dirs = [json.loads(r[0]) for r in con.execute("select path_json from dirs")]
        con.close()
        gini = float((2 * np.arange(1, len(sizes) + 1) - len(sizes) - 1) @ sizes / (len(sizes) * sizes.sum()))
        desc[d.name] = {"log_folders": float(np.log(len(dirs))), "max_depth": max(len(p) for p in dirs),
                        "notes_per_folder": float(sizes.mean()), "folder_size_gini": gini,
                        "top_folder_share": float(sizes.max() / sizes.sum())}
    margins = {}
    for a, b in (("cascade", "kNN"), ("LoRA", "cascade"), ("LoRA", "kNN")):
        both = [r for r in rows if r.get(a) is not None and r.get(b) is not None]
        ta, tb = per_vault(both, a), per_vault(both, b)
        margins[f"{a}-{b}"] = {v: (ta[v][0] - tb[v][0]) / ta[v][1] for v in ta}
    rng = np.random.default_rng(seed)
    out = {}
    for m, mv in margins.items():
        vs = sorted(set(mv) & set(desc))
        out[m] = {k: spearman_perm([desc[v][k] for v in vs], [mv[v] for v in vs], reps, rng)
                  for k in next(iter(desc.values()))}
        out[m]["n_vaults"] = len(vs)
    return out


def capped_vaults(build: Path) -> list[str]:
    out = []
    for d in vault_dirs(build):
        meta = json.loads((d / "META.json").read_text())
        if meta.get("skip_reasons", {}).get("max_files_per_root"):
            out.append(d.name)
    return out


def fmt(title: str, tab: dict, arms: list[str]) -> str:
    lines = [title]
    for b, row in tab.items():
        cells = []
        for a in arms:
            s = row.get(a)
            if s:
                cells.append(f"{a} {s['acc']:.3f} [{s['ci'][0]:.3f},{s['ci'][1]:.3f}]")
        lines.append(f"  {b:<5} n={next(iter(row.values())).get('n', '?'):<5} " + "  ".join(cells))
        diffs = [f"{k} {s['diff']:+.3f} [{s['ci'][0]:+.3f},{s['ci'][1]:+.3f}]"
                 for k, s in row.items() if "-" in k]
        if diffs:
            lines.append("        " + "  ".join(diffs))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=RUNS / "vault_level_robustness")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("loading embedding cache ...", flush=True)
    emb = CachedEmbeddings(CACHE)
    report: dict = {"reps": args.reps, "seed": args.seed}

    exact_arms = ["majority", "flat", "BM25", "kNN", "kNN_k1", "centroid", "cascade", "cascade_llama", "LoRA",
                  "cascade_member", "LoRA_pervault_e1", "LoRA_pervault_e3", "LoRA_pooled_reserved"]

    for corpus in CORPORA:
        rows = item_rows(corpus, emb)
        arms = [a for a in exact_arms if any(r.get(a) is not None for r in rows)]
        rep = {
            "n_items": len(rows),
            "n_vaults": len({r["vault"] for r in rows}),
            "exact": by_bucket(rows, arms, args.reps, args.seed),
            "soft": by_bucket(rows, [a + "~soft" for a in arms], args.reps, args.seed),
            "crossing_cascade_vs_kNN": crossing_per_vault(rows, "cascade", "kNN"),
            "crossing_cascade_vs_centroid": crossing_per_vault(rows, "cascade", "centroid"),
            "crossing_member_vs_kNN": crossing_per_vault(rows, "cascade_member", "kNN"),
            "nested_k": nested_k(rows),
        }
        rep["structure_correlates"] = structure_correlates(rows, CORPORA[corpus]["build"], args.reps, args.seed)
        for m, x in rep["structure_correlates"].items():
            print(f"structure vs {m} (n={x['n_vaults']}): " +
                  "  ".join(f"{k} rho={v[0]:+.2f} p={v[1]:.3f}" for k, v in x.items() if k != "n_vaults"))
        nk = rep["nested_k"]
        print(f"\nnested k ({corpus}): grid {nk['grid_pooled']}  chosen {dict(nk['k_chosen_per_heldout_vault'])}")
        for b, x in nk["by_bucket"].items():
            print(f"  {b:<5} n={x['n']:<5} nested {x['kNN_nested']:.3f}  k=5 {x['kNN_k5']:.3f}")
        print(fmt(f"\n=== corpus {corpus}, item split, exact (vault-clustered 95% CI) ===", rep["exact"], arms))
        print(fmt(f"\n=== corpus {corpus}, item split, soft ===", rep["soft"], [a + "~soft" for a in arms]))
        c = rep["crossing_cascade_vs_kNN"]
        print(f"\nper-vault crossing (cascade vs kNN): {c['full_crossing']}/{c['vaults_with_both_buckets']} vaults "
              f"show it, {c['reverse_crossing']} reverse; sparse gap > 0 in {c['sparse_gap_positive']}, "
              f"dense gap < 0 in {c['dense_gap_negative']}")
        c = rep["crossing_cascade_vs_centroid"]
        print(f"per-vault crossing (cascade vs centroid): {c['full_crossing']}/{c['vaults_with_both_buckets']}")

        if corpus == "B":
            capped = capped_vaults(CORPORA[corpus]["build"])
            uncapped = [r for r in rows if r["vault"] not in capped]
            no_outl = [r for r in uncapped if not any(o in r["vault"] for o in OUTLIERS)]
            rep["capped_vaults"] = capped
            rep["exact_excluding_capped"] = by_bucket(uncapped, arms, args.reps, args.seed)
            rep["exact_excluding_capped_and_outliers"] = by_bucket(no_outl, arms, args.reps, args.seed)
            print(fmt(f"\n=== corpus B excluding {len(capped)} capped vaults ===", rep["exact_excluding_capped"], arms))

        if CORPORA[corpus]["fold_arms"]:
            frows = fold_rows(corpus, emb)
            farms = ["path_top1", *CORPORA[corpus]["fold_arms"]]
            rep["folder_disjoint"] = {
                "exact": by_bucket(frows, farms, args.reps, args.seed)["all"],
                "soft": by_bucket(frows, [a + "~soft" for a in farms], args.reps, args.seed)["all"],
            }
            print(fmt(f"\n=== corpus {corpus} folder-disjoint (occupancy 0) ===",
                      {"0": rep["folder_disjoint"]["exact"]}, farms))
        report[corpus] = rep

    report["A"] = corpus_a_table()
    print("\n=== corpus A (one tree), item split, item-level ===")
    for b, x in report["A"].items():
        print(f"  {b:<5} " + "  ".join(f"{k} {v}" for k, v in x.items()))

    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nwrote", args.out / "summary.json")


if __name__ == "__main__":
    main()
