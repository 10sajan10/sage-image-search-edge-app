"""Scheduled camera and directory producers for the durable ingest spool."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path

from PIL import Image

from app_config import IngestConfig
from spool import DurableSpool, SpoolRecord

LOGGER = logging.getLogger("image_search.sources")
POINT_NAMESPACE = uuid.UUID("6f9d5c1e-3a7b-4e2f-9c88-0f5a1b2c3d4e")


def point_id(config: IngestConfig, source: str, image_id: str) -> str:
    identity = (
        f"{config.common.node_id}/{config.common.camera_id}/{source}/{image_id}"
    )
    return str(uuid.uuid5(POINT_NAMESPACE, identity))


def safe_path_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return component or "unknown"


def make_record(
    config: IngestConfig,
    image_path: Path,
    image_id: str,
    source: str,
    timestamp: int,
) -> SpoolRecord:
    return SpoolRecord(
        point_id=point_id(config, source, image_id),
        image_id=image_id,
        image_path=str(image_path),
        source=source,
        node_id=config.common.node_id,
        camera_id=config.common.camera_id,
        timestamp=timestamp,
        created_at_ns=time.time_ns(),
    )


def discover_directory(
    config: IngestConfig, spool: DurableSpool
) -> tuple[int, int]:
    if not config.image_dir.is_dir():
        raise FileNotFoundError(f"IMAGE_DIR does not exist: {config.image_dir}")
    discovered = queued = 0
    for image_path in sorted(config.image_dir.glob(config.image_glob)):
        if not image_path.is_file():
            continue
        discovered += 1
        relative = image_path.relative_to(config.image_dir)
        timestamp = image_path.stat().st_mtime_ns
        record = make_record(
            config,
            image_path,
            str(relative),
            "directory",
            timestamp,
        )
        if spool.enqueue(record):
            queued += 1
            if queued >= config.max_discover_per_cycle:
                break
    return discovered, queued


def save_camera_frame(
    config: IngestConfig, snapshot, timestamp: int
) -> Path:
    captured_at = time.gmtime(timestamp / 1_000_000_000)
    directory = (
        config.frame_dir
        / safe_path_component(config.common.node_id)
        / safe_path_component(config.common.camera_id)
        / time.strftime("%Y/%m/%d", captured_at)
    )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{timestamp}.jpg"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    image = Image.fromarray(snapshot.data).convert("RGB")
    with temporary.open("wb") as handle:
        image.save(handle, format="JPEG", quality=90, optimize=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def run_camera(
    config: IngestConfig,
    spool: DurableSpool,
    stop_event: threading.Event,
    wake_event: threading.Event,
    status: dict,
) -> None:
    from waggle.data.vision import Camera

    interval = float(config.capture_interval_seconds)
    next_tick = time.monotonic()
    captures = 0
    LOGGER.info(
        "opening camera",
        extra={"camera": config.camera, "interval_seconds": interval},
    )
    with Camera(config.camera) as camera:
        while not stop_event.is_set():
            wait = next_tick - time.monotonic()
            if wait > 0 and stop_event.wait(wait):
                break
            snapshot = camera.snapshot()
            timestamp = int(
                getattr(snapshot, "timestamp", None) or time.time_ns()
            )
            image_path = save_camera_frame(config, snapshot, timestamp)
            image_id = str(image_path.relative_to(config.frame_dir))
            queued = spool.enqueue(
                make_record(
                    config, image_path, image_id, "camera", timestamp
                )
            )
            captures += 1
            status["captures"] = captures
            status["last_capture_ns"] = timestamp
            LOGGER.info(
                "captured frame",
                extra={
                    "image_id": image_id,
                    "queued": queued,
                    "captures": captures,
                },
            )
            wake_event.set()
            if config.max_captures and captures >= config.max_captures:
                break

            next_tick += interval
            now = time.monotonic()
            if next_tick <= now:
                missed = int((now - next_tick) // interval) + 1
                next_tick += missed * interval
                status["missed_capture_intervals"] = (
                    status.get("missed_capture_intervals", 0) + missed
                )
                LOGGER.warning(
                    "capture schedule overran; skipped intervals",
                    extra={"missed_intervals": missed},
                )


def run_directory(
    config: IngestConfig,
    spool: DurableSpool,
    stop_event: threading.Event,
    wake_event: threading.Event,
    status: dict,
) -> None:
    cycles = 0
    while not stop_event.is_set():
        discovered, queued = discover_directory(config, spool)
        cycles += 1
        status["discovery_cycles"] = cycles
        status["last_discovery_ns"] = time.time_ns()
        status["last_discovered"] = discovered
        status["last_queued"] = queued
        LOGGER.info(
            "directory discovery complete",
            extra={"discovered": discovered, "queued": queued, "cycle": cycles},
        )
        if queued:
            wake_event.set()
        if config.exit_when_drained:
            return
        if stop_event.wait(config.capture_interval_seconds):
            return


def run_source(
    config: IngestConfig,
    spool: DurableSpool,
    stop_event: threading.Event,
    wake_event: threading.Event,
    status: dict,
) -> None:
    try:
        if config.capture_source == "camera":
            run_camera(config, spool, stop_event, wake_event, status)
        else:
            run_directory(config, spool, stop_event, wake_event, status)
    except Exception as error:  # surfaced to and handled by the main process
        status["fatal_error"] = f"{type(error).__name__}: {error}"
        LOGGER.exception("capture/discovery producer failed")
    finally:
        status["finished"] = True
        wake_event.set()
