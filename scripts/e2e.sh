#!/usr/bin/env bash
# Destructive only to its uniquely named temporary Qdrant collection/containers.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${1:-localhost/sage-image-search-edge-app:production-test}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
E2E_PORT="${E2E_PORT:-18099}"
COLLECTION="image_search_e2e_$(date +%s)_$$"
WORK_DIR="$(mktemp -d /tmp/sage-image-search-e2e.XXXXXX)"
API_CONTAINER="image-search-api-e2e-$$"

cleanup() {
  docker rm -f "$API_CONTAINER" >/dev/null 2>&1 || true
  curl -fsS -X DELETE "$QDRANT_URL/collections/$COLLECTION" >/dev/null 2>&1 || true
  rm -r "$WORK_DIR"
}
trap cleanup EXIT

mkdir -p "$WORK_DIR/images" "$WORK_DIR/data"
cp "$REPO_DIR/e2e/images/testset/firebench_34182404715_6454a87316_o.jpg" \
  "$WORK_DIR/images/fire.jpg"
cp "$REPO_DIR/e2e/images/testset/commonobjectsbench_000000001238.jpg" \
  "$WORK_DIR/images/elephant.jpg"

common_args=(
  --device nvidia.com/gpu=all
  --network host
  -v "$MODEL_ROOT:/model:ro"
  -e HF_HOME=/model/hf-cache
  -e EMBED_MODEL_DIR=/model/jina-clip-v2
  -e QDRANT_URL="$QDRANT_URL"
  -e COLLECTION="$COLLECTION"
  -e LOG_FORMAT=text
)

docker run --rm \
  "${common_args[@]}" \
  -v "$WORK_DIR/images:/data/images:ro" \
  -v "$WORK_DIR/data:/data" \
  -e CAPTURE_SOURCE=directory \
  -e IMAGE_DIR=/data/images \
  -e EXIT_WHEN_DRAINED=true \
  -e PUBLISH_TO_BEEHIVE=false \
  -e NODE_ID=E2E \
  -e CAMERA_ID=test-camera \
  -e OLLAMA_URL="$OLLAMA_URL" \
  -e CAPTION_TARGET_WORDS=40 \
  "$IMAGE"

points_before="$(
  curl -fsS "$QDRANT_URL/collections/$COLLECTION" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["points_count"])'
)"
[ "$points_before" = 2 ]

# Restart against the same spool: deterministic identity and completed records
# must prevent duplicate work.
docker run --rm \
  "${common_args[@]}" \
  -v "$WORK_DIR/images:/data/images:ro" \
  -v "$WORK_DIR/data:/data" \
  -e CAPTURE_SOURCE=directory \
  -e IMAGE_DIR=/data/images \
  -e EXIT_WHEN_DRAINED=true \
  -e PUBLISH_TO_BEEHIVE=false \
  -e NODE_ID=E2E \
  -e CAMERA_ID=test-camera \
  -e OLLAMA_URL="$OLLAMA_URL" \
  "$IMAGE"

docker run -d --name "$API_CONTAINER" \
  "${common_args[@]}" \
  -e SEARCH_PORT="$E2E_PORT" \
  -e SEARCH_API_KEY=e2e-key \
  "$IMAGE" search_api.py >/dev/null

for _ in $(seq 1 60); do
  curl -fsS --max-time 2 "http://127.0.0.1:$E2E_PORT/readyz" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS "http://127.0.0.1:$E2E_PORT/readyz" >/dev/null

SEARCH_API_KEY=e2e-key python3 "$REPO_DIR/search_cli.py" \
  "wildfire smoke and flames" \
  --url "http://127.0.0.1:$E2E_PORT" \
  --top-k 2 \
  --json > "$WORK_DIR/result.json"

python3 - "$WORK_DIR/result.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
assert result["returned"] == 2
assert result["results"][0]["image_id"] == "fire.jpg"
assert result["legs_queried"] == ["bm25", "caption", "image"]
print("E2E PASS: ingest, restart idempotence, authenticated API, and ranking")
PY
