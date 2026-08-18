#!/usr/bin/env bash
# Upload a chat-JSONL file as a Fireworks dataset.
#   bash scripts/fireworks_upload_dataset.sh <dataset-id> <path/to/file.jsonl>
# Create -> signed URL -> PUT -> validate. Idempotent only in the sense that a
# duplicate dataset-id fails loudly rather than silently overwriting.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
set -a; source .env; set +a
KEY="${FIREWORKS_API_KEY:-${FIREWORKS_API:-}}"
ACCT="${FIREWORKS_ACCOUNT_ID:-prakharkhatri123-edp}"
BASE="https://api.fireworks.ai/v1/accounts/${ACCT}"

DS_ID="$1"
FILE="$2"
[[ -f "$FILE" ]] || { echo "no such file: $FILE" >&2; exit 1; }
ROWS=$(wc -l < "$FILE")
BYTES=$(stat -c %s "$FILE")
echo "dataset=$DS_ID rows=$ROWS bytes=$BYTES"

echo "-- create"
curl -sS -X POST "${BASE}/datasets" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"datasetId\":\"${DS_ID}\",\"dataset\":{\"displayName\":\"${DS_ID}\",\"exampleCount\":\"${ROWS}\",\"userUploaded\":{}}}" \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("state") or d.get("message") or d)'

echo "-- signed url"
URL=$(curl -sS -X POST "${BASE}/datasets/${DS_ID}:getUploadEndpoint" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"filenameToSize\":{\"$(basename "$FILE")\":${BYTES}}}" \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);print(list(d["filenameToSignedUrls"].values())[0])')

echo "-- upload"
curl -sS -X PUT -H "Content-Type: application/octet-stream" \
  -H "x-goog-content-length-range: ${BYTES},${BYTES}" \
  --data-binary "@${FILE}" "$URL"

echo "-- validate"
curl -sS -X POST "${BASE}/datasets/${DS_ID}:validateUpload" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{}' \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("state") or d)'

for _ in $(seq 30); do
  ST=$(curl -sS -H "Authorization: Bearer $KEY" "${BASE}/datasets/${DS_ID}" \
       | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("state"),d.get("exampleCount"))')
  echo "  $ST"
  [[ "$ST" == READY* ]] && exit 0
  sleep 3
done
echo "dataset did not reach READY" >&2; exit 1
