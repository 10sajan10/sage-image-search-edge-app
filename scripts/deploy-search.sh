#!/usr/bin/env bash
# Deploy the always-running search image as a Deployment + private Service.
#
# Usage:
#   CONFIG_FILE=/secure/search-config.yaml \
#   SECRET_FILE=/secure/search-secret.yaml \
#   MODEL_ROOT=/opt/sage/image-search-models \
#   DATA_ROOT=/var/lib/sage-image-search \
#   scripts/deploy-search.sh REGISTRY/IMAGE:TAG
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_REF="${1:-}"
CONFIG_FILE="${CONFIG_FILE:-}"
SECRET_FILE="${SECRET_FILE:-}"
MODEL_ROOT="${MODEL_ROOT:-/opt/sage/image-search-models}"
DATA_ROOT="${DATA_ROOT:-/var/lib/sage-image-search}"

fail() {
  echo "error: $*" >&2
  exit 1
}

[ -n "$IMAGE_REF" ] || fail "an immutable search image reference is required"
[ -f "$CONFIG_FILE" ] || fail "CONFIG_FILE does not exist: $CONFIG_FILE"
if [ -n "$SECRET_FILE" ]; then
  [ -f "$SECRET_FILE" ] || fail "SECRET_FILE does not exist: $SECRET_FILE"
else
  sudo kubectl get secret image-search-secret >/dev/null 2>&1 \
    || fail "SECRET_FILE is required when image-search-secret does not exist"
fi
[ -d "$MODEL_ROOT/jina-clip-v2" ] || fail "missing $MODEL_ROOT/jina-clip-v2"
[ -d "$MODEL_ROOT/hf-cache" ] || fail "missing $MODEL_ROOT/hf-cache"
[ -d "$DATA_ROOT" ] || fail "missing persistent image directory: $DATA_ROOT"
[[ "$IMAGE_REF" =~ ^[A-Za-z0-9._/@:-]+$ ]] || fail "invalid image reference"
[[ "$MODEL_ROOT" =~ ^/[A-Za-z0-9._/-]+$ ]] || fail "invalid MODEL_ROOT"
[[ "$DATA_ROOT" =~ ^/[A-Za-z0-9._/-]+$ ]] || fail "invalid DATA_ROOT"

sudo kubectl apply -f "$CONFIG_FILE"
if [ -n "$SECRET_FILE" ]; then
  sudo kubectl apply -f "$SECRET_FILE"
fi

sed \
  -e "s|registry.sagecontinuum.org/USERNAME/sage-image-search-api:0.3.4|$IMAGE_REF|" \
  -e "s|path: /opt/sage/image-search-models|path: $MODEL_ROOT|" \
  -e "s|path: /var/lib/sage-image-search|path: $DATA_ROOT|" \
  "$REPO_DIR/deploy/sage/search-deployment.yaml" \
  | sudo kubectl apply -f -

sudo kubectl scale deployment/image-search-api --replicas=1
sudo kubectl rollout status deployment/image-search-api --timeout=10m
