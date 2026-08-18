#!/usr/bin/env bash
# Morning: wait for mid SFT (if needed), live-merge deploy, holdout eval, teardown.
# Usage: bash scripts/fireworks_mid_finish.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
set -a; source .env; set +a
KEY="${FIREWORKS_API_KEY:-${FIREWORKS_API:-}}"
ACCT="${FIREWORKS_ACCOUNT_ID:-prakharkhatri123-edp}"
BASE="https://api.fireworks.ai/v1/accounts/${ACCT}"
JOB_ID="$(cat runs/fireworks_mid_job_id.txt)"
MODEL="accounts/${ACCT}/models/placer-mid-llama31-8b"
DEP_ID="placer-mid-llama31-8b-live"

echo "job=$JOB_ID"
while true; do
  st=$(curl -sS -H "Authorization: Bearer $KEY" "$BASE/supervisedFineTuningJobs/${JOB_ID}")
  state=$(python3 -c 'import sys,json; print(json.load(sys.stdin).get("state"))' <<<"$st")
  pct=$(python3 -c 'import sys,json; print((json.load(sys.stdin).get("jobProgress") or {}).get("percent"))' <<<"$st")
  echo "$(date +%H:%M:%S) $state pct=$pct"
  case "$state" in
    JOB_STATE_COMPLETED) break ;;
    JOB_STATE_FAILED|JOB_STATE_CANCELLED|JOB_STATE_DELETED)
      echo "$st" | python3 -m json.tool | head -40
      exit 1
      ;;
  esac
  sleep 30
done

echo "deploy live-merge $DEP_ID ..."
curl -sS -X POST "${BASE}/deployments?deploymentId=${DEP_ID}" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{
    \"baseModel\": \"${MODEL}\",
    \"displayName\": \"placer mid llama31-8b live-merge\",
    \"acceleratorType\": \"NVIDIA_H200_141GB\",
    \"acceleratorCount\": 1,
    \"minReplicaCount\": 1,
    \"maxReplicaCount\": 1,
    \"precision\": \"BF16\",
    \"deploymentShape\": \"accounts/fireworks/deploymentShapes/rft-llama-v3p1-8b-instruct\"
  }" | python3 -c 'import sys,json;d=json.load(sys.stdin); print(d.get("state") or d.get("message"), d.get("name"))'

while true; do
  st=$(curl -sS -H "Authorization: Bearer $KEY" "$BASE/deployments/${DEP_ID}")
  state=$(python3 -c 'import sys,json; print(json.load(sys.stdin).get("state"))' <<<"$st")
  msg=$(python3 -c 'import sys,json; print((json.load(sys.stdin).get("status") or {}).get("message",""))' <<<"$st")
  echo "$(date +%H:%M:%S) deploy $state $msg"
  [[ "$state" == "READY" ]] && break
  [[ "$state" == "FAILED" ]] && exit 1
  sleep 15
done

ROUTE="${MODEL}#accounts/${ACCT}/deployments/${DEP_ID}"
python3 scripts/eval_fireworks_placer.py \
  --tasks data/multitree_synth_mid/place_holdout.jsonl \
  --out runs/fireworks_placer_mid \
  --model "$ROUTE"

echo "teardown deployment..."
curl -sS -X DELETE -H "Authorization: Bearer $KEY" \
  "${BASE}/deployments/${DEP_ID}?ignoreChecks=true" >/dev/null
echo "done → runs/fireworks_placer_mid/summary.json"
cat runs/fireworks_placer_mid/summary.json
