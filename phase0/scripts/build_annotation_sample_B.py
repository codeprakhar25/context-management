#!/usr/bin/env python3
"""H1 gold-label sample + H2 alternative-folder sample from corpus B (27 vaults), plus
per-annotator offline CSV sheets. Pre-specified in PLACER_FINDINGS.md (H1, H2).

H1 mirrors build_annotation_sample.py (corpus A): 100 notes, quota 35 folder-disjoint
(occupancy 0) / 18 / 25 / 22 over buckets 1-2 / 3-9 / 10+, drawn from item-split val and
fold-split val, deduped by id (a note can sit in both val pools under different buckets).
Corpus B is 27 vaults of very different size, so a per-vault cap (--vault-cap, default 6,
~1.6x the 3.7 mean) keeps a few big vaults from dominating the sample.

H2: 50 item-split items (disjoint from H1) where cascade, kNN or LoRA is wrong; one wrong
prediction is picked at random and shown as the "alternative" folder. The method that made
it goes ONLY to the key file, so annotators cannot tell which method it was.

Annotators see: a GitHub permalink to the note at the vault's pinned commit (they read it on
GitHub; no third-party note text is copied into any sheet), the recorded folder, and the
vault's folder list (corpus A showed the existing dirs; here that is
data/annotation_B/vault_folders.txt). Bucket, split and method are never on a sheet. The
sample JSONs hold no note text either, only ids, paths and links.
Item order is shuffled per annotator (seeded) so position effects do not correlate.

Outputs are still gitignored (they name third-party files). No API calls, no network:
links are built from the manifest commit + source_file and checked against data/vaults_raw.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
from urllib.parse import quote
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "data" / "vaults_build"
RAW = ROOT / "data" / "vaults_raw"
RUNS = ROOT / "runs"
ANNOTATORS = ["annotator_1", "annotator_2"]

# method -> (run dir, per-vault subdir suffix); per-item records have id/gold_path/pred_path/exact
METHODS = {
    "cascade": ("vaultB_cascade", "__item_note20"),
    "kNN": ("vaultB_knn_k5only", "__item"),
    "LoRA": ("vaultB_lora_item", ""),
}


def store_dirs(db: Path) -> list[str]:
    """Folder list straight from the frozen store (probe_llm_placer would pull in the API client)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return ["/".join(json.loads(r[0])) for r in con.execute("SELECT path_json FROM dirs ORDER BY depth, path_key")]
    finally:
        con.close()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def permalink(vault: str, commit: dict[str, tuple[str, str]], source_file: str) -> str | None:
    """https://github.com/<owner>/<repo>/blob/<sha>/<rel path>, or None if the file is not in the raw clone."""
    try:
        rel = Path(source_file).relative_to(RAW / vault)
    except ValueError:
        return None
    if not (RAW / vault / rel).is_file():
        return None
    full_name, sha = commit[vault]
    return f"https://github.com/{full_name}/blob/{sha}/" + "/".join(quote(seg, safe="") for seg in rel.parts)


def bucket(n: int) -> str:
    return "1-2" if n <= 2 else "3-9" if n <= 9 else "10+"


def parse_plan(s: str) -> dict[str, int]:
    return {k: int(v) for k, v in (p.split(":") for p in s.split(","))}


def draw(by_bucket: dict[str, list[dict]], plan: dict[str, int], order: list[str],
         cap: int, rng: random.Random, used: set[str]) -> list[dict]:
    """Per bucket: shuffle, take greedily skipping vaults already at the cap, skip used ids."""
    per_vault: Counter = Counter()
    out: list[dict] = []
    for b in order:
        pool = [t for t in by_bucket[b] if t["id"] not in used]
        rng.shuffle(pool)
        take: list[dict] = []
        for t in pool:
            if len(take) == plan.get(b, 0):
                break
            if per_vault[t["vault"]] < cap:
                take.append(t)
                per_vault[t["vault"]] += 1
        if len(take) < plan.get(b, 0):
            print(f"WARNING: bucket {b} filled {len(take)}/{plan[b]} under cap {cap}")
        out.extend(take)
        used.update(t["id"] for t in take)
    return out


def write_sheet(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel opens non-ASCII paths correctly
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(header)
        w.writerows(rows)


def shuffled(items: list, seed: int, annotator: str, tag: str) -> list:
    r = random.Random(f"{seed}/{annotator}/{tag}")
    out = list(items)
    r.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "annotation_sample_B.json")
    ap.add_argument("--alt-out", type=Path, default=ROOT / "data" / "annotation_alternatives_B.json")
    ap.add_argument("--sheets-dir", type=Path, default=ROOT / "data" / "annotation_B")
    ap.add_argument("--plan", default="0:35,1-2:18,3-9:25,10+:22")
    ap.add_argument("--alt-plan", default="1-2:17,3-9:17,10+:16")
    ap.add_argument("--vault-cap", type=int, default=6, help="max H1 items per vault")
    ap.add_argument("--alt-vault-cap", type=int, default=4, help="max H2 items per vault")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    plan, alt_plan = parse_plan(args.plan), parse_plan(args.alt_plan)

    manifest = json.loads((ROOT / "data" / "vaults_manifest.json").read_text())["vaults"]
    commit = {r["full_name"].replace("/", "__"): (r["full_name"], r["commit"]) for r in manifest}
    unresolved: list[tuple[str, str]] = []

    vaults = sorted(d for d in BUILD.iterdir() if d.is_dir() and (d / "val.jsonl").exists())
    h1_pool: dict[str, list[dict]] = defaultdict(list)
    h2_pool: dict[str, list[dict]] = defaultdict(list)
    vault_dirs: dict[str, list[str]] = {}
    id_vault: dict[str, str] = {}
    for vd in vaults:
        v = vd.name
        cnt: Counter = Counter()
        for t in load_jsonl(vd / "train.jsonl"):
            cnt[tuple(t["gold_path"])] += 1
        preds = {m: {r["id"]: r for r in load_jsonl(RUNS / run / (v + suf) / "holdout_results.jsonl")}
                 for m, (run, suf) in METHODS.items()}
        for t in load_jsonl(vd / "val.jsonl"):
            b = bucket(cnt[tuple(t["gold_path"])])
            url = permalink(v, commit, t["source_file"])
            if url is None:
                unresolved.append((v, t["id"]))
                continue
            rec = {**t, "vault": v, "split": "item", "occupancy_bucket": b, "url": url}
            h1_pool[b].append(rec)
            wrong = {m: p[t["id"]]["pred_path"] for m, p in preds.items()
                     if t["id"] in p and p[t["id"]]["pred_path"] != t["gold_path"]}
            if wrong:
                h2_pool[b].append({**rec, "wrong": wrong})
        for t in load_jsonl(vd / "fold_val.jsonl"):
            url = permalink(v, commit, t["source_file"])
            if url is None:
                unresolved.append((v, t["id"]))
                continue
            h1_pool["0"].append({**t, "vault": v, "split": "folder", "occupancy_bucket": "0", "url": url})
        vault_dirs[v] = store_dirs(vd / "hierstore.sqlite")
    for b in h1_pool:
        for t in h1_pool[b]:
            assert id_vault.setdefault(t["id"], t["vault"]) == t["vault"], "id collides across vaults"

    rng = random.Random(args.seed)
    used: set[str] = set()
    h1 = draw(h1_pool, plan, ["1-2", "3-9", "10+", "0"], args.vault_cap, rng, used)
    rng.shuffle(h1)
    assert len({t["id"] for t in h1}) == len(h1), "duplicate id survived sampling"

    h2 = draw(h2_pool, alt_plan, ["1-2", "3-9", "10+"], args.alt_vault_cap, rng, used)
    rng.shuffle(h2)
    assert len({t["id"] for t in h2}) == len(h2) and not {t["id"] for t in h2} & {t["id"] for t in h1}

    h1_items, h2_items, h2_key = [], [], {}
    for i, t in enumerate(h1):
        h1_items.append({"item": f"H1-{i + 1:03d}", "id": t["id"], "vault": t["vault"], "url": t["url"],
                         "gold_path": t["gold_path"], "occupancy_bucket": t["occupancy_bucket"],
                         "split": t["split"]})
    for i, t in enumerate(h2):
        m = rng.choice(sorted(t["wrong"]))  # random one of the wrong predictions
        code = f"H2-{i + 1:03d}"
        h2_items.append({"item": code, "id": t["id"], "vault": t["vault"], "url": t["url"],
                         "gold_path": t["gold_path"], "alt_path": t["wrong"][m],
                         "occupancy_bucket": t["occupancy_bucket"]})
        h2_key[code] = {"method": m, "id": t["id"], "occupancy_bucket": t["occupancy_bucket"],
                        "n_wrong_methods": len(t["wrong"])}

    args.out.write_text(json.dumps({
        "corpus": "B", "n_items": len(h1_items), "bucket_plan": plan, "vault_cap": args.vault_cap,
        "seed": args.seed, "items": h1_items}, indent=2, ensure_ascii=False))
    args.alt_out.write_text(json.dumps({
        "corpus": "B", "n_items": len(h2_items), "bucket_plan": alt_plan, "vault_cap": args.alt_vault_cap,
        "seed": args.seed, "items": h2_items}, indent=2, ensure_ascii=False))
    sd = args.sheets_dir
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "h2_key.json").write_text(json.dumps(h2_key, indent=2))

    used_vaults = sorted({i["vault"] for i in h1_items + h2_items})
    (sd / "vault_folders.txt").write_text("".join(
        f"=== {v}\n" + "\n".join(sorted(vault_dirs[v])) + "\n\n" for v in used_vaults))

    fp = lambda p: "/".join(p)  # noqa: E731
    for a in ANNOTATORS:
        write_sheet(sd / a / "h1_gold_label.csv",
                    ["item", "vault", "note_link", "recorded_folder", "verdict", "comment"],
                    [[i["item"], i["vault"], i["url"], fp(i["gold_path"]), "", ""]
                     for i in shuffled(h1_items, args.seed, a, "h1")])
        write_sheet(sd / a / "h2_alternative.csv",
                    ["item", "vault", "note_link", "recorded_folder", "alternative_folder",
                     "also_sensible", "comment"],
                    [[i["item"], i["vault"], i["url"], fp(i["gold_path"]), fp(i["alt_path"]), "", ""]
                     for i in shuffled(h2_items, args.seed, a, "h2")])

    print(f"unresolved source_file (excluded from pools): {len(unresolved)} {unresolved[:10]}")
    print(f"H1: {len(h1_items)} items -> {args.out}")
    print(" bucket", dict(Counter(i["occupancy_bucket"] for i in h1_items)))
    print(" vaults", len({i['vault'] for i in h1_items}), "max/vault",
          max(Counter(i["vault"] for i in h1_items).values()))
    print(f"H2: {len(h2_items)} items -> {args.alt_out}")
    print(" bucket", dict(Counter(i["occupancy_bucket"] for i in h2_items)))
    print(" vaults", len({i['vault'] for i in h2_items}), "max/vault",
          max(Counter(i["vault"] for i in h2_items).values()))
    print(" shown method (key only)", dict(Counter(k["method"] for k in h2_key.values())))
    print(f"sheets -> {sd}/{{{','.join(ANNOTATORS)}}}")


if __name__ == "__main__":
    main()
