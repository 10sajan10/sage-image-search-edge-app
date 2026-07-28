#!/usr/bin/env bash
# Render and submit a scheduled one-shot Sage ingestion job.
#
# Required:
#   INGEST_IMAGE=registry/path/image:tag
#   QDRANT_URL=http://private-qdrant:6333
#   OLLAMA_URL=http://private-ollama:11434
#
# Optional values, including COLLECTION and SCHEDULE, are documented in
# deploy/sage/jobs/README.md. Set DRY_RUN=true to print instead of submit.
set -euo pipefail

INGEST_IMAGE="${INGEST_IMAGE:-}"
NODE_ID="${NODE_ID:-H01E}"
QDRANT_URL="${QDRANT_URL:-}"
OLLAMA_URL="${OLLAMA_URL:-}"
COLLECTION="${COLLECTION:-edge_v4_live}"
SCHEDULE="${SCHEDULE:-*/5 * * * *}"
JOB_NAME="${JOB_NAME:-image-search-ingest-${NODE_ID,,}}"
PLUGIN_NAME="${PLUGIN_NAME:-image-search-ingest-once}"
CAMERA_SECRET_NAME="${CAMERA_SECRET_NAME:-image-search-ingest}"
CAMERA_SECRET_KEY="${CAMERA_SECRET_KEY:-CAMERA}"
MODEL_ROOT="${MODEL_ROOT:-/opt/sage/image-search-models}"
DATA_ROOT="${DATA_ROOT:-/var/lib/sage-image-search}"
JOB_ENV_FILE="${JOB_ENV_FILE:-}"
DRY_RUN="${DRY_RUN:-false}"

fail() {
  echo "error: $*" >&2
  exit 1
}

[ -n "$INGEST_IMAGE" ] || fail "INGEST_IMAGE is required"
[ -n "$QDRANT_URL" ] || fail "QDRANT_URL is required"
[ -n "$OLLAMA_URL" ] || fail "OLLAMA_URL is required"
[[ "$JOB_NAME" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]] \
  || fail "JOB_NAME must contain only lowercase letters, digits, and hyphens"
[[ "$PLUGIN_NAME" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]] \
  || fail "PLUGIN_NAME must contain only lowercase letters, digits, and hyphens"
[[ "$CAMERA_SECRET_NAME" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]] \
  || fail "CAMERA_SECRET_NAME must contain lowercase letters, digits, and hyphens"
[[ "$CAMERA_SECRET_KEY" =~ ^[A-Za-z0-9]+$ ]] \
  || fail "CAMERA_SECRET_KEY must be alphanumeric"
[[ "$MODEL_ROOT" =~ ^/[A-Za-z0-9._/-]+$ ]] || fail "MODEL_ROOT must be absolute"
[[ "$DATA_ROOT" =~ ^/[A-Za-z0-9._/-]+$ ]] || fail "DATA_ROOT must be absolute"
if [ -n "$JOB_ENV_FILE" ]; then
  [ -f "$JOB_ENV_FILE" ] || fail "JOB_ENV_FILE does not exist: $JOB_ENV_FILE"
fi
if [ "$DRY_RUN" != "true" ]; then
  command -v sesctl >/dev/null || fail "sesctl is not installed"
fi
command -v python3 >/dev/null || fail "python3 is not installed"

job_file="$(mktemp /tmp/image-search-ingest-job.XXXXXX.json)"
cleanup() {
  rm -f "$job_file"
}
trap cleanup EXIT

export INGEST_IMAGE NODE_ID QDRANT_URL OLLAMA_URL COLLECTION SCHEDULE
export JOB_NAME PLUGIN_NAME CAMERA_SECRET_NAME CAMERA_SECRET_KEY
export MODEL_ROOT DATA_ROOT JOB_ENV_FILE

python3 - "$job_file" <<'PY'
import json
import os
import re
import sys


def value(name: str) -> str:
    return os.environ[name]


extra_env: dict[str, str] = {}
env_file = value("JOB_ENV_FILE")
if env_file:
    with open(env_file, encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise SystemExit(
                    f"{env_file}:{line_number}: expected a KEY=VALUE line"
                )
            key, item_value = line.split("=", 1)
            key = key.strip()
            item_value = item_value.strip()
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
                raise SystemExit(
                    f"{env_file}:{line_number}: invalid environment key {key!r}"
                )
            if key in {"CAMERA", "RUN_MODE"}:
                raise SystemExit(
                    f"{env_file}:{line_number}: {key} is controlled by the "
                    "submission helper"
                )
            extra_env[key] = item_value

plugin_env = {
    "CAPTURE_SOURCE": "camera",
    "CAMERA": (
        f"{{secret.{value('CAMERA_SECRET_NAME')}."
        f"{value('CAMERA_SECRET_KEY')}}}"
    ),
    "CAMERA_ID": "rtsp-main",
    "NODE_ID": value("NODE_ID"),
    "QDRANT_URL": value("QDRANT_URL"),
    "COLLECTION": value("COLLECTION"),
    "OLLAMA_URL": value("OLLAMA_URL"),
    "CAPTION_MODEL": "gemma4:e2b",
    "EMBED_MODEL_DIR": "/model/jina-clip-v2",
    "HF_HOME": "/model/hf-cache",
    "REQUIRE_GPU": "true",
    "FRAME_DIR": "/data/frames",
    "SPOOL_DIR": "/data/spool",
    "PUBLISH_TO_BEEHIVE": "true",
    "LOG_FORMAT": "json",
}
plugin_env.update(extra_env)
plugin_env["RUN_MODE"] = "oneshot"

plugin_name = value("PLUGIN_NAME")
job = {
    "name": value("JOB_NAME"),
    "plugins": [
        {
            "name": plugin_name,
            "pluginSpec": {
                "image": value("INGEST_IMAGE"),
                "selector": {"resource.gpu": "true"},
                "resource": {
                    "request.memory": "8Gi",
                    "limit.memory": "16Gi",
                },
                "volume": {
                    value("MODEL_ROOT"): "/model",
                    value("DATA_ROOT"): "/data",
                },
                "env": plugin_env,
            },
        }
    ],
    "nodes": {value("NODE_ID"): True},
    "scienceRules": [
        f'schedule({plugin_name}): cronjob("{plugin_name}", '
        f'"{value("SCHEDULE")}")'
    ],
    "successCriteria": ["WallClock(1d)"],
}

with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(job, stream, indent=2)
    stream.write("\n")
PY

if [ "$DRY_RUN" = "true" ]; then
  cat "$job_file"
  exit 0
fi

printf 'Submitting %s on %s: collection=%s schedule=%s\n' \
  "$JOB_NAME" "$NODE_ID" "$COLLECTION" "$SCHEDULE"
sesctl submit --file-path "$job_file"
