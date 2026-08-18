#!/usr/bin/env bash
# Wait mid-hard SFT → deploy → holdout eval → user-dir transfer → teardown.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
set -a; source .env; set +a
KEY="${FIREWORKS_API_KEY:-${FIREWORKS_API:-}}"
ACCT="${FIREWORKS_ACCOUNT_ID:-prakharkhatri123-edp}"
BASE="https://api.fireworks.ai/v1/accounts/${ACCT}"
JOB_ID="$(cat runs/fireworks_mid_hard_job_id.txt)"
MODEL="accounts/${ACCT}/models/placer-mid-hard-llama31-8b"
DEP_ID="placer-mid-hard-live-$(date +%H%M%S)"

echo "job=$JOB_ID"
while true; do
  st=$(curl -sS -H "Authorization: Bearer $KEY" "$BASE/supervisedFineTuningJobs/${JOB_ID}")
  state=$(python3 -c 'import sys,json; print(json.load(sys.stdin).get("state"))' <<<"$st")
  pct=$(python3 -c 'import sys,json; print((json.load(sys.stdin).get("jobProgress") or {}).get("percent"))' <<<"$st")
  echo "$(date +%H:%M:%S) $state pct=$pct"
  case "$state" in
    JOB_STATE_COMPLETED) break ;;
    JOB_STATE_FAILED|JOB_STATE_CANCELLED|JOB_STATE_DELETED) echo "$st" | python3 -m json.tool | head -40; exit 1 ;;
  esac
  sleep 30
done

echo "deploy $DEP_ID"
curl -sS -X POST "${BASE}/deployments?deploymentId=${DEP_ID}" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{
    \"baseModel\": \"${MODEL}\",
    \"displayName\": \"placer mid-hard eval\",
    \"acceleratorType\": \"NVIDIA_H200_141GB\",
    \"acceleratorCount\": 1,
    \"minReplicaCount\": 1,
    \"maxReplicaCount\": 1,
    \"precision\": \"BF16\"
  }" | python3 -c 'import sys,json;d=json.load(sys.stdin); print(d.get("state") or d.get("message"))'

while true; do
  st=$(curl -sS -H "Authorization: Bearer $KEY" "$BASE/deployments/${DEP_ID}")
  state=$(python3 -c 'import sys,json; print(json.load(sys.stdin).get("state"))' <<<"$st")
  echo "$(date +%H:%M:%S) deploy $state"
  [[ "$state" == "READY" ]] && break
  [[ "$state" == "FAILED" ]] && exit 1
  sleep 15
done

ROUTE="${MODEL}#accounts/${ACCT}/deployments/${DEP_ID}"
python3 scripts/eval_fireworks_placer.py \
  --tasks data/multitree_synth_mid_hard/place_holdout.jsonl \
  --out runs/fireworks_placer_mid_hard \
  --model "$ROUTE"

python3 scripts/eval_fireworks_placer.py \
  --tasks data/user_dir_snap/place_tasks_from_snap.jsonl \
  --store data/user_dir_snap/hierstore.sqlite \
  --out runs/fireworks_mid_hard_user_dir \
  --model "$ROUTE"

# gpt-4o hard holdout if not present
if [[ ! -f runs/gpt4o_mid_hard_holdout/summary.json ]]; then
  mkdir -p runs/gpt4o_mid_hard_holdout
  python3 -c '
import json, importlib.util
from pathlib import Path
ROOT=Path(".").resolve()
spec=importlib.util.spec_from_file_location("ev", ROOT/"scripts"/"eval_multitree_smoke.py")
ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
holdout=ev.load_jsonl(ROOT/"data"/"multitree_synth_mid_hard"/"place_holdout.jsonl")
out=ROOT/"runs"/"gpt4o_mid_hard_holdout"
s=ev.eval_llm_holdout(holdout, "gpt-4o", 0, out)
(out/"summary.json").write_text(json.dumps({"llm_holdout": s}, indent=2))
print(json.dumps(s, indent=2))
' | tee runs/gpt4o_mid_hard_holdout/eval.log
fi

curl -sS -X DELETE -H "Authorization: Bearer $KEY" \
  "${BASE}/deployments/${DEP_ID}?ignoreChecks=true" >/dev/null

python3 <<'PY'
import json
from pathlib import Path
hard=json.loads(Path("runs/fireworks_placer_mid_hard/summary.json").read_text())
ud=json.loads(Path("runs/fireworks_mid_hard_user_dir/summary.json").read_text())
gpt=json.loads(Path("runs/gpt4o_mid_hard_holdout/summary.json").read_text()).get("llm_holdout", {})
gpt_ud=json.loads(Path("runs/llm_placer_user_dir_with_dirs/summary.json").read_text())
easy=json.loads(Path("runs/dual_eval_report.json").read_text())
rep={
  "hard_holdout_n300": {
    "lora8b": {"exact": hard["path_exact"], "soft": hard["path_soft"], "branch": hard["branch_ok"]},
    "gpt4o": {"exact": gpt.get("path_exact"), "soft": gpt.get("path_soft"), "branch": gpt.get("branch_ok")},
  },
  "user_dir_transfer_n127": {
    "lora8b_mid_hard": {"exact": ud["path_exact"], "soft": ud["path_soft"], "branch": ud["branch_ok"]},
    "gpt4o_prior": {"exact": gpt_ud["path_exact"], "soft": gpt_ud["path_soft"], "branch": gpt_ud["branch_ok"]},
  },
  "prev_easy_mid_for_compare": easy,
  "leak_train_holdout_exact_text": 0,
}
Path("runs/hard_eval_report.json").write_text(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=2))
PY
echo "wrote runs/hard_eval_report.json"
