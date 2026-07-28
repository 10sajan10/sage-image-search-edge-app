"""Persistent Sage capture and database-ingestion worker.

Capture/discovery and database processing are intentionally decoupled by a
durable filesystem spool:

    scheduled capture -> JPEG + pending JSON -> caption/embed -> Qdrant

If Ollama, Qdrant, or the GPU is temporarily unavailable, the pending record
survives process and pod restarts and is retried with exponential backoff.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import asdict
from typing import Any

from app_config import IngestConfig
from logging_setup import configure_logging
from pipeline import Captioner, Embedder, FrameIndexer, QdrantStore
from sources import run_source
from spool import DurableSpool, atomic_json

LOGGER = logging.getLogger("image_search.ingest")


def open_plugin(enabled: bool):
    if not enabled:
        return None
    try:
        from waggle.plugin import Plugin

        plugin = Plugin()
        plugin.__enter__()
        return plugin
    except Exception as error:
        LOGGER.warning(
            "PyWaggle telemetry is unavailable; ingestion will continue",
            extra={"error": str(error)},
        )
        return None


def publish(plugin, name: str, value: Any, timestamp: int, meta: dict) -> None:
    if plugin is None:
        return
    with contextlib.suppress(Exception):
        plugin.publish(name, value, timestamp=timestamp, meta=meta)


def write_heartbeat(
    config: IngestConfig,
    spool: DurableSpool,
    source_status: dict,
    runtime: dict,
) -> None:
    atomic_json(
        config.heartbeat_path,
        {
            "timestamp_ns": time.time_ns(),
            "pid": os.getpid(),
            "role": "ingest",
            "collection": config.common.collection,
            "node_id": config.common.node_id,
            "camera_id": config.common.camera_id,
            "capture_source": config.capture_source,
            "spool": spool.counts(),
            "source": dict(source_status),
            "runtime": dict(runtime),
        },
    )


def run() -> int:
    config = IngestConfig.from_env()
    LOGGER.info("resolved ingest configuration", extra={"config": config.public_dict()})

    stop_event = threading.Event()
    wake_event = threading.Event()

    def request_stop(signum, _frame) -> None:
        LOGGER.info("shutdown requested", extra={"signal": signum})
        stop_event.set()
        wake_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    spool = DurableSpool(config.spool_dir)
    store = QdrantStore(config.common)
    embedder = Embedder(config.common)
    captioner = Captioner(config.common, config.caption)
    indexer = FrameIndexer(store, captioner, embedder)

    store.ensure_collection()
    embedder.load()

    plugin = open_plugin(config.publish_to_beehive)
    source_status: dict[str, Any] = {
        "finished": False,
        "fatal_error": None,
    }
    runtime: dict[str, Any] = {
        "indexed": 0,
        "retrying": 0,
        "dead_lettered": 0,
        "last_indexed_at_ns": None,
        "last_error": None,
    }

    source_thread = threading.Thread(
        name="capture-source",
        target=run_source,
        args=(config, spool, stop_event, wake_event, source_status),
        daemon=True,
    )
    source_thread.start()

    exit_code = 0
    try:
        while not stop_event.is_set():
            records = spool.ready(limit=1)
            if records:
                record = records[0]
                try:
                    result = indexer.index(record.to_frame())
                except Exception as error:
                    dead, delay = spool.fail(
                        record,
                        error,
                        config.max_attempts,
                        config.retry_base_seconds,
                        config.retry_max_seconds,
                    )
                    runtime["last_error"] = (
                        f"{type(error).__name__}: {error}"
                    )
                    if dead:
                        runtime["dead_lettered"] += 1
                        LOGGER.error(
                            "frame moved to dead letter",
                            extra={
                                "point_id": record.point_id,
                                "image_id": record.image_id,
                                "attempts": record.attempts,
                                "error": str(error),
                            },
                        )
                    else:
                        runtime["retrying"] += 1
                        LOGGER.warning(
                            "frame ingestion failed; scheduled retry",
                            extra={
                                "point_id": record.point_id,
                                "image_id": record.image_id,
                                "attempts": record.attempts,
                                "retry_in_seconds": delay,
                                "error": str(error),
                            },
                        )
                    publish(
                        plugin,
                        "imagesearch.ingest.error",
                        1,
                        record.timestamp,
                        {
                            "node_id": record.node_id,
                            "camera_id": record.camera_id,
                            "attempt": record.attempts,
                            "dead_letter": dead,
                        },
                    )
                    write_heartbeat(config, spool, source_status, runtime)
                    continue

                spool.complete(record, asdict(result))
                runtime["indexed"] += 1
                runtime["last_indexed_at_ns"] = time.time_ns()
                runtime["last_error"] = None
                LOGGER.info(
                    "indexed frame",
                    extra={
                        "point_id": result.point_id,
                        "image_id": result.image_id,
                        "caption_words": result.caption_words,
                        "elapsed_seconds": round(result.elapsed_seconds, 3),
                    },
                )
                telemetry_meta = {
                    "node_id": record.node_id,
                    "camera_id": record.camera_id,
                    "source": record.source,
                }
                publish(
                    plugin,
                    "imagesearch.frame.indexed",
                    1,
                    record.timestamp,
                    telemetry_meta,
                )
                publish(
                    plugin,
                    "imagesearch.caption.words",
                    result.caption_words,
                    record.timestamp,
                    telemetry_meta,
                )
                publish(
                    plugin,
                    "imagesearch.ingest.seconds",
                    result.elapsed_seconds,
                    record.timestamp,
                    telemetry_meta,
                )
                write_heartbeat(config, spool, source_status, runtime)
                continue

            write_heartbeat(config, spool, source_status, runtime)
            if source_status.get("fatal_error"):
                raise RuntimeError(
                    f"capture source failed: {source_status['fatal_error']}"
                )
            if (
                config.exit_when_drained
                and source_status.get("finished")
                and spool.counts()["pending"] == 0
            ):
                LOGGER.info("source finished and spool drained; exiting")
                break

            delay = max(0.05, min(1.0, spool.next_retry_delay(1.0)))
            wake_event.wait(delay)
            wake_event.clear()
    except Exception:
        exit_code = 1
        LOGGER.exception("ingestion worker terminated")
    finally:
        stop_event.set()
        wake_event.set()
        source_thread.join(timeout=10)
        if source_thread.is_alive():
            LOGGER.error("capture source did not stop within 10 seconds")
            exit_code = 1
        if plugin is not None:
            with contextlib.suppress(Exception):
                plugin.__exit__(None, None, None)
        write_heartbeat(config, spool, source_status, runtime)

    LOGGER.info(
        "ingestion worker stopped",
        extra={"exit_code": exit_code, "spool": spool.counts()},
    )
    return exit_code


if __name__ == "__main__":
    configure_logging()
    try:
        sys.exit(run())
    except ValueError as error:
        LOGGER.error("invalid configuration", extra={"error": str(error)})
        sys.exit(2)
