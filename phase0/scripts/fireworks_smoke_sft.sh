#!/usr/bin/env bash
# Launch Fireworks LoRA SFT smoke (Llama-3.1-8B) on multitree placer data.
# Datasets already uploaded; this only creates the job.
#
# Prereq: payment method on https://fireworks.ai (credits alone not enough for SFT).
# Usage (from phase0/):  bash scripts/fireworks_smoke_sft.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
set -a; source .env; set +a
KEY="${FIREWORKS_API_KEY:-${FIREWORKS_API:-}}"
ACCT="${FIREWORKS_ACCOUNT_ID:-prakharkhatri123-edp}"
BASE="https://api.fireworks.ai/v1/accounts/${ACCT}"
JOB_ID="placer-smoke-llama31-8b-$(date +%Y%m%d-%H%M)"

if [[ -z "$KEY" ]]; then
  echo "missing FIREWORKS_API / FIREWORKS_API_KEY in .env" >&2
  exit 1
fi

echo "creating SFT job $JOB_ID ..."
curl -sS -X POST "$BASE/supervisedFineTuningJobs?supervisedFineTuningJobId=${JOB_ID}" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"displayName\": \"placer multitree smoke llama3.1-8b\",
    \"dataset\": \"accounts/${ACCT}/datasets/placer-multitree-smoke-train\",
    \"evaluationDataset\": \"accounts/${ACCT}/datasets/placer-multitree-smoke-holdout\",
    \"baseModel\": \"accounts/fireworks/models/llama-v3p1-8b-instruct\",
    \"outputModel\": \"accounts/${ACCT}/models/placer-smoke-llama31-8b\",
    \"epochs\": 2,
    \"learningRate\": 0.0001,
    \"loraRank\": 16
  }" | tee "runs/fireworks_${JOB_ID}_create.json" | python3 -m json.tool

echo
echo "poll: curl -sS -H \"Authorization: Bearer \$KEY\" $BASE/supervisedFineTuningJobs/${JOB_ID} | python3 -m json.tool"
echo "or:   firectl -a $ACCT sftj get $JOB_ID"
