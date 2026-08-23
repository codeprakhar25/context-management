#!/usr/bin/env bash
# Build a NeurIPS 2026 workshop submission PDF from the single anonymized source.
#
# main_neurips.tex is venue-neutral; the only per-venue difference is the
# \workshoptitle string that the neurips_2026 style prints in the footer.
#
# Usage:  ./build_workshop.sh tae
#         ./build_workshop.sh palm
# Output: submission_<venue>.pdf

set -euo pipefail
cd "$(dirname "$0")"

TECTONIC="${TECTONIC:-$HOME/.local/bin/tectonic}"
VENUE="${1:-}"

case "$VENUE" in
  tae)  TITLE="TAE: Can We Trust AI Evaluation? Robustness, Causality, and Risk in Modern AI Assessment" ;;
  palm) TITLE="PALM: Personalized, Aligned, Long-Term Memory for AI Systems" ;;
  *)    echo "usage: $0 {tae|palm}" >&2; exit 1 ;;
esac

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

cp main_neurips.tex main.bib neurips_2026.sty "$WORK/"
mkdir -p "$WORK/figures"
cp figures/fig1_coverage_curve.pdf "$WORK/figures/"

# substitute the workshop title (python, so the string needs no sed escaping)
VENUE_TITLE="$TITLE" python3 - "$WORK/main_neurips.tex" <<'PY'
import os, sys
p = sys.argv[1]
s = open(p).read()
s = s.replace(r"\workshoptitle{WORKSHOP TITLE}",
              "\\workshoptitle{%s}" % os.environ["VENUE_TITLE"])
open(p, "w").write(s)
PY

LOG=$(mktemp)
( cd "$WORK" && "$TECTONIC" main_neurips.tex >"$LOG" 2>&1 ) \
  || { echo "compile FAILED:" >&2; cat "$LOG" >&2; rm -f "$LOG"; exit 1; }
rm -f "$LOG"

cp "$WORK/main_neurips.pdf" "submission_${VENUE}.pdf"
echo "==> submission_${VENUE}.pdf  ($(du -h "submission_${VENUE}.pdf" | cut -f1))"
