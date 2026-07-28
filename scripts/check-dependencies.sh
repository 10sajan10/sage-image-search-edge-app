#!/usr/bin/env bash
# Read-only preflight for externally managed Qdrant and Ollama endpoints.
set -euo pipefail

QDRANT_URL="${QDRANT_URL:?QDRANT_URL is required}"
OLLAMA_URL="${OLLAMA_URL:?OLLAMA_URL is required}"
COLLECTION="${COLLECTION:-edge_v4_live}"
CAPTION_MODEL="${CAPTION_MODEL:-gemma4:e2b}"

python3 - "$QDRANT_URL" "$OLLAMA_URL" "$COLLECTION" "$CAPTION_MODEL" <<'PY'
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

qdrant, ollama, collection, model = (value.rstrip("/") for value in sys.argv[1:])

def request_json(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)

errors = []
try:
    with urllib.request.urlopen(f"{qdrant}/healthz", timeout=10) as response:
        health = response.read().decode("utf-8", "replace").strip()
    print(f"Qdrant reachable: {health}")
except Exception as error:
    errors.append(f"Qdrant unreachable: {error}")

try:
    encoded = urllib.parse.quote(collection, safe="")
    info = request_json(f"{qdrant}/collections/{encoded}")
    print(f"Collection {collection!r}: {info['result']['points_count']} points")
except urllib.error.HTTPError as error:
    if error.code == 404:
        print(f"Collection {collection!r}: absent (ingestion will create it)")
    else:
        errors.append(f"Collection check failed: HTTP {error.code}")
except Exception as error:
    errors.append(f"Collection check failed: {error}")

try:
    tags = request_json(f"{ollama}/api/tags")
    names = {item["name"] for item in tags.get("models", [])}
    if model not in names:
        errors.append(f"Ollama is reachable but {model!r} is not installed")
    else:
        print(f"Ollama model available: {model}")
except Exception as error:
    errors.append(f"Ollama unreachable: {error}")

if errors:
    print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
    raise SystemExit(1)
PY
