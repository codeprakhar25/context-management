#!/usr/bin/env python3
"""Clone the corpus-B vaults at their pinned commits.

Shallow-fetches exactly the commit recorded in the manifest, so a rebuild months
later reproduces the same corpus even after the upstream repos move on. Clones
land outside git (see .gitignore) -- we redistribute the manifest and the build
scripts, never other people's note text.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(args: list[str], cwd: Path | None = None, timeout: int = 180) -> tuple[int, str]:
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s: {' '.join(args)}"
    return p.returncode, (p.stderr or p.stdout).strip()


def fetch_one(v: dict, dest: Path, force: bool) -> dict:
    name = v["full_name"]
    slug = name.replace("/", "__")
    d = dest / slug
    if d.exists() and not force:
        code, head = run(["git", "-C", str(d), "rev-parse", "HEAD"])
        if code == 0 and head.strip() == v["commit"]:
            return {"repo": name, "status": "cached", "path": str(d)}
        shutil.rmtree(d)

    d.mkdir(parents=True, exist_ok=True)
    for args in (
        ["git", "init", "-q"],
        ["git", "remote", "add", "origin", v["clone_url"]],
        # pinned commit only, no history, no other branches
        ["git", "fetch", "-q", "--depth", "1", "origin", v["commit"]],
        ["git", "checkout", "-q", "FETCH_HEAD"],
    ):
        code, err = run(args, cwd=d)
        if code != 0:
            shutil.rmtree(d, ignore_errors=True)
            return {"repo": name, "status": "failed", "error": err[:300]}

    n_md = sum(1 for p in d.rglob("*") if p.suffix.lower() in (".md", ".markdown"))
    return {"repo": name, "status": "fetched", "path": str(d), "n_md_on_disk": n_md}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text())
    args.dest.mkdir(parents=True, exist_ok=True)

    results = []
    for i, v in enumerate(man["vaults"], 1):
        r = fetch_one(v, args.dest, args.force)
        results.append(r)
        print(f"[{i}/{len(man['vaults'])}] {r['status']:<8} {r['repo']}"
              + (f"  ({r.get('n_md_on_disk')} md)" if r.get("n_md_on_disk") else "")
              + (f"  {r.get('error','')}" if r["status"] == "failed" else ""),
              flush=True)

    ok = [r for r in results if r["status"] in ("fetched", "cached")]
    (args.dest / "fetch_report.json").write_text(
        json.dumps({"n_ok": len(ok), "n_failed": len(results) - len(ok),
                    "results": results}, indent=2) + "\n"
    )
    print(f"\n{len(ok)}/{len(results)} vaults on disk -> {args.dest}")
    if len(ok) < len(results):
        print("failed: " + ", ".join(r["repo"] for r in results if r["status"] == "failed"),
              file=sys.stderr)


if __name__ == "__main__":
    main()
