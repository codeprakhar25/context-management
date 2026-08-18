#!/usr/bin/env bash
# Deploy each corpus-B LoRA (live-merge, dedicated H200), eval all 27 vaults on
# its split, teardown, repeat for the other split. One deployment up at a time.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
set -a; source .env; set +a
KEY="${FIREWORKS_API_KEY:-${FIREWORKS_API:-}}"
ACCT="${FIREWORKS_ACCOUNT_ID:-prakharkhatri123-edp}"
BASE="https://api.fireworks.ai/v1/accounts/${ACCT}"

deploy_wait() {
  local model="$1" dep_id="$2" display="$3"
  echo "deploy live-merge $dep_id ..."
  curl -sS -X POST "${BASE}/deployments?deploymentId=${dep_id}" \
    -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d "{
      \"baseModel\": \"${model}\",
      \"displayName\": \"${display}\",
      \"acceleratorType\": \"NVIDIA_H200_141GB\",
      \"acceleratorCount\": 1,
      \"minReplicaCount\": 1,
      \"maxReplicaCount\": 1,
      \"precision\": \"BF16\",
      \"deploymentShape\": \"accounts/fireworks/deploymentShapes/rft-llama-v3p1-8b-instruct\"
    }" | python3 -c 'import sys,json;d=json.load(sys.stdin); print(d.get("state") or d.get("message"), d.get("name"))'

  while true; do
    st=$(curl -sS -H "Authorization: Bearer $KEY" "$BASE/deployments/${dep_id}")
    state=$(python3 -c 'import sys,json; print(json.load(sys.stdin).get("state"))' <<<"$st")
    msg=$(python3 -c 'import sys,json; print((json.load(sys.stdin).get("status") or {}).get("message",""))' <<<"$st")
    echo "$(date +%H:%M:%S) deploy $state $msg"
    [[ "$state" == "READY" ]] && break
    [[ "$state" == "FAILED" ]] && exit 1
    sleep 15
  done
}

teardown() {
  local dep_id="$1"
  echo "teardown $dep_id ..."
  curl -sS -X DELETE -H "Authorization: Bearer $KEY" \
    "${BASE}/deployments/${dep_id}?ignoreChecks=true" >/dev/null
}

# --- item split ---
ITEM_MODEL="accounts/${ACCT}/models/placer-vaultb-item-llama31-8b"
ITEM_DEP="placer-vaultb-item-live"
deploy_wait "$ITEM_MODEL" "$ITEM_DEP" "placer vaultB item llama31-8b live-merge"
python3 scripts/vault_lora_gate.py --build data/vaults_build --out runs/vaultB_lora_item \
  --split item --model "${ITEM_MODEL}#accounts/${ACCT}/deployments/${ITEM_DEP}" --workers 4
teardown "$ITEM_DEP"

# --- folder split ---
FOLD_MODEL="accounts/${ACCT}/models/placer-vaultb-fold-llama31-8b"
FOLD_DEP="placer-vaultb-fold-live"
deploy_wait "$FOLD_MODEL" "$FOLD_DEP" "placer vaultB fold llama31-8b live-merge"
python3 scripts/vault_lora_gate.py --build data/vaults_build --out runs/vaultB_lora_fold \
  --split folder --model "${FOLD_MODEL}#accounts/${ACCT}/deployments/${FOLD_DEP}" --workers 4
teardown "$FOLD_DEP"

echo "-- final deployment list (should be empty of placer-vaultb-*) --"
curl -sS -H "Authorization: Bearer $KEY" "$BASE/deployments" \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);[print(x["name"],x["state"]) for x in d.get("deployments",[]) if "vaultb" in x["name"]]'
echo "done"
cat runs/vaultB_lora_item/pooled.json
cat runs/vaultB_lora_fold/pooled.json
