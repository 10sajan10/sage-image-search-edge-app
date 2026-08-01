from __future__ import annotations

from pathlib import Path

from PIL import Image

from image_search.config import CaptionConfig, CommonConfig, IngestConfig
from image_search.sources import discover_directory, safe_path_component
from image_search.spool import DurableSpool


def config(tmp_path: Path) -> IngestConfig:
    common = CommonConfig(
        qdrant_url="http://qdrant:6333",
        collection="test",
        vector_dim=3,
        bm25_model="Qdrant/bm25",
        embed_model_dir=tmp_path,
        node_id="H01E",
        camera_id="bottom",
        require_gpu=False,
        http_retries=0,
        http_backoff_seconds=0,
    )
    caption = CaptionConfig(
        ollama_url="http://ollama:11434",
        model="test",
        timeout_seconds=10,
        think=False,
        temperature=0,
        target_words=10,
        num_ctx=256,
        num_predict=20,
        max_image_edge=128,
        prompt="caption",
    )
    return IngestConfig(
        common=common,
        caption=caption,
        run_mode="daemon",
        capture_source="directory",
        camera="",
        capture_interval_seconds=10,
        image_dir=tmp_path / "images",
        image_glob="**/*.jpg",
        frame_dir=tmp_path / "frames",
        spool_dir=tmp_path / "spool",
        max_discover_per_cycle=100,
        max_attempts=3,
        retry_base_seconds=1,
        retry_max_seconds=10,
        max_captures=0,
        exit_when_drained=True,
        publish_to_beehive=False,
        heartbeat_path=tmp_path / "heartbeat.json",
        heartbeat_interval_seconds=10.0,
    )


def test_directory_discovery_is_durable_and_idempotent(tmp_path: Path) -> None:
    settings = config(tmp_path)
    settings.image_dir.mkdir()
    Image.new("RGB", (8, 8), "red").save(settings.image_dir / "one.jpg")
    Image.new("RGB", (8, 8), "blue").save(settings.image_dir / "two.jpg")
    spool = DurableSpool(settings.spool_dir)

    assert discover_directory(settings, spool) == (2, 2)
    assert discover_directory(settings, spool) == (2, 0)
    records = spool.ready(limit=10)
    assert len(records) == 2
    assert {item.node_id for item in records} == {"H01E"}
    assert {item.camera_id for item in records} == {"bottom"}


def test_completed_first_batch_does_not_starve_later_files(tmp_path: Path) -> None:
    settings = config(tmp_path)
    object.__setattr__(settings, "max_discover_per_cycle", 1)
    settings.image_dir.mkdir()
    Image.new("RGB", (8, 8), "red").save(settings.image_dir / "a.jpg")
    Image.new("RGB", (8, 8), "blue").save(settings.image_dir / "b.jpg")
    spool = DurableSpool(settings.spool_dir)

    assert discover_directory(settings, spool) == (1, 1)
    first = spool.ready()[0]
    spool.mark_ingested(first, {})
    discovered, queued = discover_directory(settings, spool)
    assert discovered == 2
    assert queued == 1
    assert spool.ready()[0].image_id == "b.jpg"


def test_camera_path_components_are_sanitized() -> None:
    assert safe_path_component("../../node name") == "node_name"
