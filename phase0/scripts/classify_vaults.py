#!/usr/bin/env python3
"""Separate personal note vaults from software repos that happen to ship docs.

Corpus B only means something if its hierarchies are folksonomies -- folder trees
a person invented for their own notes. A software project's markdown lives in a
folder layout dictated by the codebase, which is the same thing corpus A already
measures. Letting those through would make corpus B a duplicate of corpus A under
a different name.

Structural size filters do not catch this: `obsidian-mcp-server` and
`Langchain-Chatchat` pass every note/folder/depth threshold. Nor does the
markdown-to-file ratio, because vaults carry attachments -- a real vault
(AlexiaChen/my-notes, 0.41) can score below a real software repo
(awslabs/generative-ai-cdk-constructs, 0.51).

What does separate them is build tooling and source files. A personal vault has
no package manifest and essentially no code.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CODE_EXT = {
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py", ".rs", ".go", ".java",
    ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt",
    ".scala", ".vue", ".svelte", ".dart", ".ex", ".exs", ".zig",
}
MANIFESTS = {
    "package.json", "cargo.toml", "pyproject.toml", "setup.py", "go.mod",
    "pom.xml", "build.gradle", "build.gradle.kts", "gemfile", "composer.json",
    "pubspec.yaml", "mix.exs", "cmakelists.txt", "makefile", "build.zig",
}


def sh(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, timeout=120).stdout


def classify(full_name: str) -> dict | None:
    out = sh(["gh", "api", f"repos/{full_name}/git/trees/HEAD?recursive=1",
              "--jq", ".tree[]? | select(.type==\"blob\") | .path"])
    paths = [p for p in out.splitlines() if p]
    if not paths:
        return None

    n_code = sum(1 for p in paths if Path(p).suffix.lower() in CODE_EXT)
    manifests = sorted({
        Path(p).name.lower() for p in paths
        if Path(p).name.lower() in MANIFESTS
        # a manifest nested deep is a sample/fixture, not the project's own
        and p.count("/") <= 1
    })
    md = [p for p in paths if p.lower().endswith((".md", ".markdown"))]
    # software repos concentrate markdown at the root or under docs/
    in_docs = sum(1 for p in md if p.split("/")[0].lower() in
                  ("docs", "doc", "documentation", "website", "site"))

    return {
        "n_code": n_code,
        "manifests": manifests,
        "md_in_docs_frac": round(in_docs / max(len(md), 1), 3),
        "is_software": bool(manifests) or n_code > 30,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    probe = json.loads(args.probe.read_text())
    rows = [r for r in probe["rows"] if r["usable"]]
    print(f"classifying {len(rows)} usable repos", file=sys.stderr)

    for i, r in enumerate(rows, 1):
        try:
            c = classify(r["full_name"])
        except Exception as e:  # noqa: BLE001
            print(f"  {r['full_name']}: {e}", file=sys.stderr)
            c = None
        r.update(c or {"n_code": -1, "manifests": [], "is_software": True,
                       "md_in_docs_frac": 1.0})
        if i % 20 == 0:
            print(f"  [{i}/{len(rows)}] vaults so far: "
                  f"{sum(1 for x in rows[:i] if not x['is_software'])}", file=sys.stderr)

    vaults = [r for r in rows if not r["is_software"]]
    args.out.write_text(json.dumps({
        "n_usable": len(rows),
        "n_personal_vault": len(vaults),
        "n_software": len(rows) - len(vaults),
        "code_ext": sorted(CODE_EXT),
        "manifests": sorted(MANIFESTS),
        "rows": rows,
    }, indent=2) + "\n")

    print(f"\npersonal vaults {len(vaults)} / software {len(rows)-len(vaults)}")
    print(f"{'repo':<48}{'md':>6}{'dirs':>6}{'code':>6}  lic")
    for r in sorted(vaults, key=lambda x: -x["n_md_in_dirs"])[:45]:
        print(f"{r['full_name']:<48}{r['n_md_in_dirs']:>6}{r['n_dirs']:>6}"
              f"{r['n_code']:>6}  {r['license']}")


if __name__ == "__main__":
    main()
