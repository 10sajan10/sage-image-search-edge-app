#!/usr/bin/env bash
# Thin terminal wrapper around the versioned Python API client.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEARCH_API_URL="${SEARCH_API_URL:-http://127.0.0.1:8099}"

exec python3 "$REPO_DIR/search_cli.py" "$@" --url "$SEARCH_API_URL"
