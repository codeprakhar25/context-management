#!/usr/bin/env python3
"""Build one pooled chat-SFT dataset from all 27 corpus-B vaults, one split.

Each vault keeps its own train/val split (already built by
build_vault_corpus.py) and its own existing_dirs / MAX_DEPTH -- a vault's
LoRA example must only ever see that vault's folder tree, same as every
corpus-B eval script. Pooling happens after rows are rendered, not before.

--per-vault writes one train/val pair per vault under out/<vault>/ instead of
one pooled file (for one-adapter-per-vault LoRA training); rows are the same
bytes either way, only the grouping differs.

Row format and system prompt are byte-identical to build_sft_placer.py
(corpus A), so train/eval prompt parity holds across corpora.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.store import max_depth_from_store  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


probe = _load("probe_llm_placer", "scripts/probe_llm_placer.py")
sft = _load("build_sft_placer", "scripts/build_sft_placer.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--split", choices=["item", "folder"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-vault", action="store_true",
                    help="write out/<vault>/{train,val,val_meta}.jsonl per vault, no pooled file")
    args = ap.parse_args()

    pre = "fold_" if args.split == "folder" else ""
    snaps = sorted(d for d in args.build.iterdir()
                   if d.is_dir() and (d / "hierstore.sqlite").exists())

    train_rows, val_rows = [], []
    per_vault: dict[str, tuple[list, list]] = {}
    n_vaults = 0
    for snap in snaps:
        train_f, val_f = snap / f"{pre}train.jsonl", snap / f"{pre}val.jsonl"
        store = snap / "hierstore.sqlite"
        if not (train_f.exists() and val_f.exists()):
            continue
        train_t = [json.loads(l) for l in train_f.read_text().splitlines() if l.strip()]
        val_t = [json.loads(l) for l in val_f.read_text().splitlines() if l.strip()]
        if not (train_t or val_t):
            continue
        roots = (train_t or val_t)[0].get("roots") or ["vault"]
        existing = probe.dirs_from_store(store, roots)
        max_depth = max_depth_from_store(store)
        v_train, v_val = [], []
        for t in train_t:
            r = sft.to_openai_row(t, existing, max_depth, probe.PLACER_SYSTEM)
            r["meta"]["vault"] = snap.name
            v_train.append(r)
        for t in val_t:
            r = sft.to_openai_row(t, existing, max_depth, probe.PLACER_SYSTEM)
            r["meta"]["vault"] = snap.name
            v_val.append(r)
        per_vault[snap.name] = (v_train, v_val)
        train_rows += v_train
        val_rows += v_val
        n_vaults += 1

    args.out.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "\n".join(json.dumps({"messages": r["messages"]}, ensure_ascii=False) for r in rows)
            + ("\n" if rows else "")
        )

    def write_set(d: Path, tr: list[dict], va: list[dict]) -> None:
        d.mkdir(parents=True, exist_ok=True)
        write_jsonl(d / "train.jsonl", tr)
        write_jsonl(d / "val.jsonl", va)
        (d / "val_meta.jsonl").write_text(
            "\n".join(json.dumps(r["meta"]) for r in va) + ("\n" if va else "")
        )

    if args.per_vault:
        counts = {}
        for name, (tr, va) in per_vault.items():
            write_set(args.out / name, tr, va)
            counts[name] = {"n_train": len(tr), "n_val": len(va)}
        meta = {"split": args.split, "per_vault": True, "n_vaults": n_vaults,
                "n_train": len(train_rows), "n_val": len(val_rows), "vaults": counts}
        (args.out / "META.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps({k: v for k, v in meta.items() if k != "vaults"}, indent=2))
        return

    write_set(args.out, train_rows, val_rows)

    meta = {
        "split": args.split,
        "n_vaults": n_vaults,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
    }
    (args.out / "META.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
