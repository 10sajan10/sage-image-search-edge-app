from __future__ import annotations

import json
from pathlib import Path

from image_search.spool import DurableSpool, SpoolRecord


def record(image_path: Path) -> SpoolRecord:
    return SpoolRecord(
        point_id="00000000-0000-4000-8000-000000000001",
        image_id="image.jpg",
        image_path=str(image_path),
        source="camera",
        node_id="H01E",
        camera_id="bottom",
        timestamp=123,
        created_at_ns=123,
    )


def test_spool_is_idempotent_and_marks_ingested(tmp_path: Path) -> None:
    spool = DurableSpool(tmp_path / "spool")
    item = record(tmp_path / "image.jpg")

    assert spool.enqueue(item)
    assert not spool.enqueue(item)
    assert spool.counts() == {"pending": 1, "ingested": 0, "failed": 0}
    assert spool.ready() == [item]

    spool.mark_ingested(item, {"caption_words": 42})
    assert spool.counts() == {"pending": 0, "ingested": 1, "failed": 0}
    ingested = json.loads(
        (spool.ingested_dir / f"{item.point_id}.json").read_text()
    )
    assert ingested["caption_words"] == 42
    assert not spool.enqueue(item)


def test_failure_persists_retry_then_dead_letters(tmp_path: Path) -> None:
    spool = DurableSpool(tmp_path / "spool")
    item = record(tmp_path / "image.jpg")
    spool.enqueue(item)

    dead, delay = spool.fail(item, RuntimeError("offline"), 2, 0.1, 1)
    assert not dead
    assert delay == 0.1
    pending = json.loads(
        (spool.pending_dir / f"{item.point_id}.json").read_text()
    )
    assert pending["attempts"] == 1
    assert "offline" in pending["last_error"]

    pending["next_attempt_ns"] = 0
    (spool.pending_dir / f"{item.point_id}.json").write_text(
        json.dumps(pending), encoding="utf-8"
    )
    retried = spool.ready()[0]
    dead, delay = spool.fail(retried, RuntimeError("still offline"), 2, 0.1, 1)
    assert dead
    assert delay == 0
    assert spool.counts() == {"pending": 0, "ingested": 0, "failed": 1}


def test_ingested_record_wins_after_interrupted_cleanup(tmp_path: Path) -> None:
    spool = DurableSpool(tmp_path / "spool")
    item = record(tmp_path / "image.jpg")
    spool.enqueue(item)
    ingested = spool.ingested_dir / f"{item.point_id}.json"
    ingested.write_text('{"ingested": true}\n', encoding="utf-8")

    assert spool.ready() == []
    assert spool.counts() == {"pending": 0, "ingested": 1, "failed": 0}


def test_legacy_completed_records_are_migrated(tmp_path: Path) -> None:
    root = tmp_path / "spool"
    completed = root / "completed"
    completed.mkdir(parents=True)
    legacy = completed / "legacy-point.json"
    legacy.write_text('{"completed": true}\n', encoding="utf-8")

    spool = DurableSpool(root)

    assert not legacy.exists()
    assert (spool.ingested_dir / legacy.name).exists()
    assert spool.counts() == {"pending": 0, "ingested": 1, "failed": 0}
