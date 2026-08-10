#!/usr/bin/env bash
# host-check.sh — validate that an Ollama host is reachable and can serve
# inference. Defaults to the configured lab server.
#
# Usage:
#   scripts/host-check.sh [BASE_URL] [model]
#   scripts/host-check.sh                          # uses http://192.168.10.108:11434
#   scripts/host-check.sh http://127.0.0.1:11434   # check a different host
#   scripts/host-check.sh '' llama3.2:latest       # only run a chat probe
set -euo pipefail

BASE_URL="${1:-http://192.168.10.108:11434}"
MODEL="${2:-llama3.2:latest}"
BASE_URL="${BASE_URL%/}"

echo "==> Ollama host: $BASE_URL"

echo -n "  version  : "
curl -fsS --max-time 10 "$BASE_URL/api/version" | sed -E 's/.*"version":"([^"]+)".*/\1/' || { echo "UNREACHABLE"; exit 1; }
echo "OK"

echo "==> Loaded models (GET /api/ps):"
curl -fsS --max-time 10 "$BASE_URL/api/ps" \
  | grep -oE '"name":"[^"]+"' | sed 's/"name":"//; s/"$/  (loaded)/' || echo "  (none loaded)"
echo "  (if nothing is listed above, no model is resident in memory yet)"

echo -n "==> Chat probe with '$MODEL' (first call may wait for model load): "
curl -fsS --max-time 300 "$BASE_URL/api/chat" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: OK\"}],\"options\":{\"temperature\":0,\"num_predict\":8}}" \
  | grep -oE '"content":"[^"]*"' | sed 's/"content":"/  reply: /; s/"$//' \
  || { echo "FAILED (did the model time out?)"; exit 1; }

echo "==> Host is ready."