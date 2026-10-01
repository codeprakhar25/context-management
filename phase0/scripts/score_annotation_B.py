#!/usr/bin/env python3
"""Score the corpus-B annotation study (H1 gold audit, H2 alternative folders, H3 filter audit).

Reads the two returned annotators' CSV sheets (same filenames/columns as the ones
build_annotation_sample_B.py / build_filter_audit.py wrote) plus the keys, and reports
what the corpus-A write-up reported:
  H1: per-annotator verdict counts, Cohen's kappa + exact agreement on shared items,
      agreement and "correct" share by occupancy bucket, share of sensible gold paths
      (correct, and correct+ambiguous).
  H2: share of alternatives judged sensible (yes; yes+partly) per annotator and by the
      method that made them (from the key), and the resulting lenient accuracy per method:
      lenient = exact + (1 - exact) * share_sensible, exact taken over the full B item split,
      the sensible share bucket-reweighted by that method's wrong-count per bucket (falls
      back to the method's pooled share where a bucket cell is empty).
  H3: per-class error rate vs the classifier with Wilson 95% intervals, per annotator and
      on consensus (items where both annotators agree; disagreements counted separately).
Blank cells are reported as missing and skipped. No API calls.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANNOTATORS = ["annotator_1", "annotator_2"]
VERDICTS = ["correct", "ambiguous", "unclear", "wrong"]
H2_VALS = ["yes", "partly", "no"]
BUCKETS = ["0", "1-2", "3-9", "10+"]
METHODS = {
    "cascade": ("vaultB_cascade", "__item_note20"),
    "kNN": ("vaultB_knn_k5only", "__item"),
    "LoRA": ("vaultB_lora_item", ""),
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def read_col(path: Path, col: str, allowed: dict[str, str]) -> dict[str, str]:
    """item -> normalised value; blanks skipped, unknown values raise (a typo must not vanish)."""
    out, blank = {}, 0
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            v = (r[col] or "").strip().lower()
            if not v:
                blank += 1
                continue
            if v not in allowed:
                raise SystemExit(f"{path.name}: item {r['item']}: unrecognised value {v!r} in {col}")
            out[r["item"]] = allowed[v]
    if blank:
        print(f"  [{path.parent.name}/{path.name}] {blank} blank rows skipped")
    return out


def pct(k: int, n: int) -> str:
    return f"{k}/{n} ({k / n:.3f})" if n else "0/0"


def h1(fd: Path, sample: Path) -> None:
    print("\n== H1 gold-label audit ==")
    items = {i["item"]: i for i in json.loads(sample.read_text())["items"]}
    allowed = {v: v for v in VERDICTS}
    lab = {a: read_col(fd / a / "h1_gold_label.csv", "verdict", allowed) for a in ANNOTATORS}
    for a in ANNOTATORS:
        c = Counter(lab[a].values())
        n = len(lab[a])
        print(f"{a}: " + "  ".join(f"{v} {c[v]}/{n}" for v in VERDICTS))
    a1, a2 = (lab[a] for a in ANNOTATORS)
    shared = sorted(set(a1) & set(a2))
    x, y = [a1[i] for i in shared], [a2[i] for i in shared]
    agree = sum(p == q for p, q in zip(x, y))
    usable = lambda v: v in ("correct", "ambiguous")  # noqa: E731
    bin_agree = sum(usable(p) == usable(q) for p, q in zip(x, y))
    cross = sum({p, q} == {"correct", "wrong"} for p, q in zip(x, y))
    print(f"shared n={len(shared)}: exact agreement {pct(agree, len(shared))}, "
          f"Cohen's kappa {kappa(x, y):.3f}, usable-vs-not agreement {pct(bin_agree, len(shared))}, "
          f"correct<->wrong crossings {cross}")
    print("by occupancy bucket (shared items):  n | exact agreement | correct share a1 / a2")
    for b in BUCKETS:
        ids = [i for i in shared if items[i]["occupancy_bucket"] == b]
        if not ids:
            continue
        ag = sum(a1[i] == a2[i] for i in ids)
        c1 = sum(a1[i] == "correct" for i in ids)
        c2 = sum(a2[i] == "correct" for i in ids)
        print(f"  {b:>4}: {len(ids):3d} | {pct(ag, len(ids))} | {c1 / len(ids):.3f} / {c2 / len(ids):.3f}")
    for a in ANNOTATORS:
        n = len(lab[a])
        c = Counter(lab[a].values())
        lo, hi = wilson(c["correct"] + c["ambiguous"], n)
        print(f"{a}: sensible gold (correct) {c['correct'] / n:.3f}; "
              f"(correct+ambiguous) {(c['correct'] + c['ambiguous']) / n:.3f} [Wilson {lo:.3f}, {hi:.3f}]")


def exact_by_method() -> dict[str, tuple[int, int]]:
    out = {}
    for m, (run, suf) in METHODS.items():
        k = n = 0
        for vd in sorted((ROOT / "runs" / run).glob(f"*{suf}" if suf else "*")):
            f = vd / "holdout_results.jsonl"
            if not f.exists():
                continue
            for l in f.read_text().splitlines():
                if l.strip():
                    n += 1
                    k += bool(json.loads(l)["exact"])
        out[m] = (k, n)
    return out


def h2(fd: Path, key_path: Path) -> None:
    print("\n== H2 alternative-folder audit ==")
    key = json.loads(key_path.read_text())
    lab = {a: read_col(fd / a / "h2_alternative.csv", "also_sensible", {v: v for v in H2_VALS}) for a in ANNOTATORS}
    score = {"yes": {"yes": 1.0, "partly": 0.0, "no": 0.0},
             "yes+partly": {"yes": 1.0, "partly": 1.0, "no": 0.0}}
    for a in ANNOTATORS:
        c = Counter(lab[a].values())
        n = len(lab[a])
        print(f"{a}: " + "  ".join(f"{v} {c[v]}/{n}" for v in H2_VALS))
    shared = sorted(set(lab[ANNOTATORS[0]]) & set(lab[ANNOTATORS[1]]))
    x, y = [lab[ANNOTATORS[0]][i] for i in shared], [lab[ANNOTATORS[1]][i] for i in shared]
    print(f"shared n={len(shared)}: exact agreement {pct(sum(p == q for p, q in zip(x, y)), len(shared))}, "
          f"Cohen's kappa {kappa(x, y):.3f}")
    ex = exact_by_method()
    # sensible share per (method, bucket), averaged over annotators
    for rule, w in score.items():
        print(f"-- sensible = {rule}")
        cell: dict[tuple[str, str], list[float]] = defaultdict(list)
        for a in ANNOTATORS:
            for i, v in lab[a].items():
                cell[(key[i]["method"], key[i]["occupancy_bucket"])].append(w[v])
        allv = [s for v in cell.values() for s in v]
        print(f"overall alternatives judged sensible: {sum(allv) / len(allv):.3f} (n judgments {len(allv)})")
        for m in METHODS:
            vals = [s for (mm, _), v in cell.items() if mm == m for s in v]
            n_items = len({i for a in ANNOTATORS for i in lab[a] if key[i]["method"] == m})
            if not vals:
                print(f"  {m}: no judged alternatives")
                continue
            pooled = sum(vals) / len(vals)
            lo, hi = wilson(round(sum(vals)), len(vals))
            k, n = ex[m]
            # bucket weights: this method's wrong count per bucket over the full B item split
            wrong_b = _wrong_by_bucket(m)
            tot = sum(wrong_b.values())
            share = 0.0
            for b, c in wrong_b.items():
                v = cell.get((m, b))
                share += c / tot * (sum(v) / len(v) if v else pooled)
            lenient = k / n + (1 - k / n) * share
            print(f"  {m}: {n_items} items, sensible {pooled:.3f} (Wilson {lo:.3f}, {hi:.3f}); "
                  f"bucket-reweighted {share:.3f}; exact {k / n:.3f} -> lenient accuracy {lenient:.3f}")


_wrong_cache: dict[str, Counter] = {}


def _wrong_by_bucket(m: str) -> Counter:
    """Wrong-prediction count per occupancy bucket for method m on the B item split."""
    if m in _wrong_cache:
        return _wrong_cache[m]
    run, suf = METHODS[m]
    build = ROOT / "data" / "vaults_build"
    out: Counter = Counter()
    for vd in sorted(d for d in build.iterdir() if (d / "val.jsonl").exists()):
        cnt: Counter = Counter()
        for l in (vd / "train.jsonl").read_text().splitlines():
            if l.strip():
                cnt[tuple(json.loads(l)["gold_path"])] += 1
        f = ROOT / "runs" / run / (vd.name + suf) / "holdout_results.jsonl"
        if not f.exists():
            continue
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if not r["exact"]:
                    n = cnt[tuple(r["gold_path"])]
                    out["1-2" if n <= 2 else "3-9" if n <= 9 else "10+"] += 1
    _wrong_cache[m] = out
    return out


def h3(fd: Path, key_path: Path) -> None:
    print("\n== H3 vault-filter audit ==")
    key = json.loads(key_path.read_text())
    allowed = {"personal": "personal", "not": "not", "p": "personal", "n": "not"}
    lab = {a: read_col(fd / a / "h3_filter_audit.csv", "personal_note_collection (personal / not)", allowed)
           for a in ANNOTATORS}
    shared = sorted(set(lab[ANNOTATORS[0]]) & set(lab[ANNOTATORS[1]]))
    x, y = [lab[ANNOTATORS[0]][i] for i in shared], [lab[ANNOTATORS[1]][i] for i in shared]
    print(f"annotator agreement on {len(shared)} repos: {pct(sum(p == q for p, q in zip(x, y)), len(shared))}, "
          f"kappa {kappa(x, y):.3f}")
    # classifier is_software False = "personal"; error = annotator disagrees with it
    classes = {"classifier personal (is_software=false)": False, "classifier software (is_software=true)": True}
    views = {a: lab[a] for a in ANNOTATORS}
    views["consensus (both agree)"] = {i: lab[ANNOTATORS[0]][i] for i in shared
                                       if lab[ANNOTATORS[0]][i] == lab[ANNOTATORS[1]][i]}
    for name, v in views.items():
        print(f"-- {name}")
        for cname, sw in classes.items():
            ids = [i for i in v if key[i]["is_software"] == sw]
            err = sum((v[i] == "not") if not sw else (v[i] == "personal") for i in ids)
            lo, hi = wilson(err, len(ids))
            print(f"  {cname}: error {pct(err, len(ids))}, Wilson 95% [{lo:.3f}, {hi:.3f}]")
    dis = [i for i in shared if lab[ANNOTATORS[0]][i] != lab[ANNOTATORS[1]][i]]
    print(f"annotator disagreements (excluded from consensus): {len(dis)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filled", type=Path, default=ROOT / "data" / "annotation_B" / "filled",
                    help="dir holding annotator_1/ and annotator_2/ with the returned CSVs")
    ap.add_argument("--keys", type=Path, default=ROOT / "data" / "annotation_B")
    ap.add_argument("--sample", type=Path, default=ROOT / "data" / "annotation_sample_B.json")
    ap.add_argument("--only", choices=["h1", "h2", "h3"], default=None)
    args = ap.parse_args()
    if args.only in (None, "h1"):
        h1(args.filled, args.sample)
    if args.only in (None, "h2"):
        h2(args.filled, args.keys / "h2_key.json")
    if args.only in (None, "h3"):
        h3(args.filled, args.keys / "h3_key.json")


if __name__ == "__main__":
    main()
