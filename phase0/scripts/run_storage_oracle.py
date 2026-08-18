#!/usr/bin/env python3
"""Replay scripted storage oracle cases (CRUD integrity + path fidelity).

No LLM. Proves hard-tree executor: MKDIR / MOVE / depth / roots / subtree.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.store import HierStore, Op  # noqa: E402

DEFAULT_CASES = ROOT / "data" / "storage_oracle" / "cases.jsonl"


def load_cases(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def run_case(case: dict) -> dict:
    expect = case.get("expect") or {}
    strict = bool(case.get("strict", False))
    roots = case.get("roots") or ["work", "personal", "inbox"]
    errors: list[str] = []
    raised: str | None = None

    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "oracle.sqlite"
        store = HierStore(
            db,
            roots=roots,
            strict_dirs=strict or expect.get("raises") is not None
            and "does not exist" in str(expect.get("raises")),
        )
        # depth/root reject cases always strict
        if expect.get("raises"):
            store.strict_dirs = True

        ops = [Op(**o) if isinstance(o, dict) else o for o in case["ops"]]
        try:
            store.apply_ops(ops, manager="oracle")
        except Exception as e:
            raised = f"{type(e).__name__}: {e}"
            if not expect.get("raises"):
                errors.append(f"unexpected raise: {raised}")
            else:
                needle = str(expect["raises"]).lower()
                if needle not in raised.lower():
                    errors.append(
                        f"raise mismatch: want substring {needle!r}, got {raised!r}"
                    )
            store.close()
            return {
                "id": case["id"],
                "ok": len(errors) == 0,
                "errors": errors,
                "raised": raised,
            }

        if expect.get("raises"):
            errors.append(f"expected raise containing {expect['raises']!r}, none raised")

        snap = store.snapshot()
        by_id = {f["id"]: f for f in snap["facts"]}

        for fid, path in (expect.get("fact_paths") or {}).items():
            f = by_id.get(fid)
            if f is None:
                errors.append(f"missing fact {fid}")
            elif f["path"] != path:
                errors.append(f"{fid} path {f['path']} != gold {path}")

        for fid, text in (expect.get("fact_texts") or {}).items():
            f = by_id.get(fid)
            if f is None:
                errors.append(f"missing fact {fid} for text check")
            elif f["text"] != text:
                errors.append(f"{fid} text {f['text']!r} != {text!r}")

        for fid in expect.get("valid_ids") or []:
            f = by_id.get(fid)
            if f is None or not f["valid"]:
                errors.append(f"expected valid {fid}")

        for fid in expect.get("invalid_ids") or []:
            f = by_id.get(fid)
            if f is None:
                errors.append(f"expected soft-deleted {fid} present")
            elif f["valid"]:
                errors.append(f"expected invalid {fid}")

        for fid in expect.get("missing_ids") or []:
            if fid in by_id:
                errors.append(f"expected hard-deleted missing {fid}")

        if "valid_count" in expect and snap["valid_count"] != expect["valid_count"]:
            errors.append(
                f"valid_count {snap['valid_count']} != {expect['valid_count']}"
            )

        have_dirs = {json.dumps(d) for d in snap["dirs"]}
        for d in expect.get("must_dirs") or []:
            if json.dumps(d) not in have_dirs:
                errors.append(f"missing dir {d}")

        sub = expect.get("subtree")
        if sub:
            got = {f["id"] for f in store.read_subtree(sub["active"])}
            want = set(sub["ids"])
            if got != want:
                errors.append(f"subtree {sub['active']}: got {sorted(got)} want {sorted(want)}")

        # unexplained missing: every valid_id / fact_paths key accounted
        store.close()

    return {
        "id": case["id"],
        "ok": len(errors) == 0,
        "errors": errors,
        "raised": raised,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "storage_oracle.json")
    args = ap.parse_args()

    cases = load_cases(args.cases)
    results = [run_case(c) for c in cases]
    n_ok = sum(1 for r in results if r["ok"])
    report = {
        "n": len(results),
        "n_ok": n_ok,
        "n_fail": len(results) - n_ok,
        "all_pass": n_ok == len(results),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("n", "n_ok", "n_fail", "all_pass")}, indent=2))
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"  {status} {r['id']}")
        for e in r.get("errors") or []:
            print(f"    - {e}")
    if not report["all_pass"]:
        sys.exit(1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
