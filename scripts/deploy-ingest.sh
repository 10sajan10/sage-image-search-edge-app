#!/usr/bin/env bash
# Deploy the already-built ingestion image through Sage pluginctl.
#
# Usage:
#   ENV_FILE=/secure/path/ingest.env \
#   scripts/deploy-ingest.sh REGISTRY/IMAGE:TAG
#
# This script deliberately does not build images, start infrastructure, remove
# an existing workload, or invent service addresses.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_REF="${1:-}"
ENV_FILE="${ENV_FILE:-}"
MODEL_ROOT="${MODEL_ROOT:-/opt/sage/image-search-models}"
DATA_ROOT="${DATA_ROOT:-/var/lib/sage-image-search}"

fail() {
  echo "error: $*" >&2
  exit 1
}

[ -n "$IMAGE_REF" ] || fail "usage: ENV_FILE=/secure/ingest.env $0 REGISTRY/IMAGE:TAG"
[ -n "$ENV_FILE" ] || fail "ENV_FILE is required"
[ -f "$ENV_FILE" ] || fail "environment file does not exist: $ENV_FILE"
[ -d "$MODEL_ROOT/jina-clip-v2" ] || fail "missing $MODEL_ROOT/jina-clip-v2"
[ -d "$MODEL_ROOT/hf-cache" ] || fail "missing $MODEL_ROOT/hf-cache"
command -v pluginctl >/dev/null || fail "pluginctl is not installed"
command -v kubectl >/dev/null || fail "kubectl is not installed"

# pluginctl requires strict KEY=VALUE lines: no comments or blank lines.
if awk 'NF == 0 || /^#/ || index($0, "=") == 0 {bad=1} END {exit bad}' "$ENV_FILE"; then
  :
else
  fail "$ENV_FILE must contain only non-empty KEY=VALUE lines"
fi

sudo pluginctl deploy \
  --name image-search-ingest \
  --type deployment \
  --selector resource.gpu=true \
  --resource request.memory=8Gi,limit.memory=16Gi \
  --env-from "$ENV_FILE" \
  -v "$MODEL_ROOT:/model" \
  -v "$DATA_ROOT:/data" \
  "$IMAGE_REF"

sudo kubectl patch deployment image-search-ingest \
  --type strategic \
  --patch-file "$REPO_DIR/deploy/sage/ingest-production-patch.yaml"

sudo kubectl rollout status deployment/image-search-ingest --timeout=10m
