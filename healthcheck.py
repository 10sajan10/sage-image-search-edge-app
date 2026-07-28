"""Container health check for the persistent ingestion role."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    path = Path(os.environ.get("HEARTBEAT_PATH", "/data/spool/heartbeat.json"))
    max_age = float(os.environ.get("HEARTBEAT_MAX_AGE_SECONDS", "30"))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        age = (time.time_ns() - int(data["timestamp_ns"])) / 1e9
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        print(f"unhealthy: heartbeat unavailable: {error}", file=sys.stderr)
        return 1
    if age > max_age:
        print(
            f"unhealthy: heartbeat is {age:.1f}s old (maximum {max_age:.1f}s)",
            file=sys.stderr,
        )
        return 1
    print(f"healthy: heartbeat age={age:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
