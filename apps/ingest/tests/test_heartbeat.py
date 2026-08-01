"""The liveness heartbeat must not depend on ingestion progress."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from image_search.config import IngestConfig
from image_search.spool import DurableSpool

import main


def build_config(monkeypatch, tmp_path: Path) -> IngestConfig:
    monkeypatch.setenv("CAPTURE_SOURCE", "directory")
    monkeypatch.setenv("SPOOL_DIR", str(tmp_path / "spool"))
    monkeypatch.setenv("HEARTBEAT_INTERVAL_SECONDS", "0.5")
    return IngestConfig.from_env()


def test_heartbeat_refreshes_while_ingestion_is_blocked(
    monkeypatch, tmp_path: Path
) -> None:
    """Captioning blocks the work loop for far longer than the probe budget.

    A single Ollama call is bounded by CAPTION_TIMEOUT (300s) and retried with
    backoff. If the heartbeat were written from the work loop it would go stale
    during healthy work and the liveness probe would restart the pod mid-frame.
    """
    config = build_config(monkeypatch, tmp_path)
    spool = DurableSpool(config.spool_dir)
    stop_event = threading.Event()

    thread = threading.Thread(
        target=main.run_heartbeat,
        args=(config, spool, stop_event, {}, {}),
        daemon=True,
    )
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while not config.heartbeat_path.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        first = json.loads(config.heartbeat_path.read_text(encoding="utf-8"))

        # Simulate the work loop being blocked inside a slow caption.
        time.sleep(1.5)
        second = json.loads(config.heartbeat_path.read_text(encoding="utf-8"))
    finally:
        stop_event.set()
        thread.join(timeout=5)

    assert second["timestamp_ns"] > first["timestamp_ns"]
    assert second["role"] == "ingest"
    assert second["spool"] == {"pending": 0, "ingested": 0, "failed": 0}


def test_heartbeat_thread_stops_on_shutdown(monkeypatch, tmp_path: Path) -> None:
    config = build_config(monkeypatch, tmp_path)
    spool = DurableSpool(config.spool_dir)
    stop_event = threading.Event()

    thread = threading.Thread(
        target=main.run_heartbeat,
        args=(config, spool, stop_event, {}, {}),
        daemon=True,
    )
    thread.start()
    stop_event.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
