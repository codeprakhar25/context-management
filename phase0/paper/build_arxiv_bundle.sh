#!/usr/bin/env bash
# Assemble the arXiv submission tarball from the paper sources.
#
# arXiv runs LaTeX but not BibTeX, so main.bbl must ship pre-built. main.bib is
# included too: arXiv tolerates it, and it lets the bundle be rebuilt from a
# clean checkout without the surrounding repo.
#
# Usage:  ./build_arxiv_bundle.sh
# Output: arxiv_submission/ and arxiv_submission.tar.gz (both git-ignored)

set -euo pipefail
cd "$(dirname "$0")"

TECTONIC="${TECTONIC:-$HOME/.local/bin/tectonic}"
OUT=arxiv_submission

command -v "$TECTONIC" >/dev/null 2>&1 || { echo "tectonic not found at $TECTONIC" >&2; exit 1; }

echo "==> compiling to generate main.bbl"
LOG=$(mktemp)
"$TECTONIC" --keep-intermediates main.tex >"$LOG" 2>&1 \
  || { echo "compile FAILED:" >&2; cat "$LOG" >&2; exit 1; }

echo "==> assembling $OUT/"
rm -rf "$OUT" "$OUT.tar.gz"
mkdir -p "$OUT/figures"
cp main.tex main.bib main.bbl acl.sty acl_natbib.bst "$OUT/"
cp figures/fig1_coverage_curve.pdf "$OUT/figures/"

echo "==> verifying the bundle builds standalone"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP" "$LOG"' EXIT
cp -r "$OUT"/. "$TMP"/
( cd "$TMP" && "$TECTONIC" main.tex >"$LOG" 2>&1 ) \
  || { echo "standalone build FAILED:" >&2; cat "$LOG" >&2; exit 1; }

tar czf "$OUT.tar.gz" -C "$OUT" .

# drop build intermediates the compile above left in the source dir
rm -f main.aux main.out main.log main.blg main.fls main.fdb_latexmk

echo "==> ok"
tar tzf "$OUT.tar.gz" | sort | sed 's/^/    /'
echo "    $(du -h "$OUT.tar.gz" | cut -f1)  $OUT.tar.gz"
