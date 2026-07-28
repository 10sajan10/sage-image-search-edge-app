#!/usr/bin/env bash
# Build the two independently deployable application images.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INGEST_IMAGE="${INGEST_IMAGE:-10.31.81.1:5000/local/image-search-ingest:0.3.3}"
SEARCH_IMAGE="${SEARCH_IMAGE:-10.31.81.1:5000/local/image-search-api:0.3.3}"
PUSH="${PUSH:-true}"

docker build \
  -t "$INGEST_IMAGE" \
  "$REPO_DIR/apps/ingest"

docker build \
  -t "$SEARCH_IMAGE" \
  "$REPO_DIR/apps/search"

if [ "$PUSH" = "true" ]; then
  docker push "$INGEST_IMAGE"
  docker push "$SEARCH_IMAGE"
fi

printf 'ingest=%s\nsearch=%s\n' "$INGEST_IMAGE" "$SEARCH_IMAGE"
