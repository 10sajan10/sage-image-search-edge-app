"""Durable filesystem spool between capture/discovery and database ingestion."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from image_search.pipeline import Frame


@dataclass
class SpoolRecord:
    point_id: str
    image_id: str
    image_path: str
    source: str
    node_id: str
    camera_id: str
    timestamp: int
    created_at_ns: int
    attempts: int = 0
    next_attempt_ns: int = 0
    last_error: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "SpoolRecord":
        return cls(**data)

    def to_frame(self) -> Frame:
        return Frame(
            point_id=self.point_id,
            image_id=self.image_id,
            image_path=Path(self.image_path),
            source=self.source,
            node_id=self.node_id,
            camera_id=self.camera_id,
            timestamp=self.timestamp,
        )


def atomic_json(path: Path, value: dict) -> None:
    """Write and fsync JSON before atomically replacing the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class DurableSpool:
    def __init__(self, root: Path):
        self.root = root
        self.pending_dir = root / "pending"
        self.ingested_dir = root / "ingested"
        self.failed_dir = root / "failed"
        for directory in (self.pending_dir, self.ingested_dir, self.failed_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._migrate_completed_records()

    def _migrate_completed_records(self) -> None:
        """Move records written by releases which named the state completed."""
        completed_dir = self.root / "completed"
        if not completed_dir.is_dir():
            return
        for source in completed_dir.glob("*.json"):
            destination = self.ingested_dir / source.name
            if not destination.exists():
                os.replace(source, destination)

    def _path(self, directory: Path, point_id: str) -> Path:
        return directory / f"{point_id}.json"

    def contains(self, point_id: str) -> bool:
        return any(
            self._path(directory, point_id).exists()
            for directory in (
                self.pending_dir,
                self.ingested_dir,
                self.failed_dir,
            )
        )

    def enqueue(self, record: SpoolRecord) -> bool:
        if self.contains(record.point_id):
            return False
        atomic_json(self._path(self.pending_dir, record.point_id), asdict(record))
        return True

    def ready(self, limit: int = 1) -> list[SpoolRecord]:
        now = time.time_ns()
        records = []
        for path in sorted(self.pending_dir.glob("*.json")):
            # mark_ingested() writes the durable result before unlinking pending.
            # A crash between those operations is safe: discard the stale
            # pending marker instead of repeating captioning and embedding.
            if (
                self._path(self.ingested_dir, path.stem).exists()
                or self._path(self.failed_dir, path.stem).exists()
            ):
                path.unlink()
                continue
            try:
                record = SpoolRecord.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                corrupt = self.failed_dir / f"{path.stem}.corrupt.json"
                os.replace(path, corrupt)
                atomic_json(
                    corrupt.with_suffix(".error.json"),
                    {
                        "failed_at_ns": now,
                        "error": f"invalid spool record: {error}",
                        "record_path": str(corrupt),
                    },
                )
                continue
            if record.next_attempt_ns <= now:
                records.append(record)
                if len(records) >= limit:
                    break
        return records

    def mark_ingested(self, record: SpoolRecord, result: dict) -> None:
        source = self._path(self.pending_dir, record.point_id)
        destination = self._path(self.ingested_dir, record.point_id)
        ingested = {**asdict(record), "ingested_at_ns": time.time_ns(), **result}
        atomic_json(destination, ingested)
        if source.exists():
            source.unlink()

    def fail(
        self,
        record: SpoolRecord,
        error: Exception,
        max_attempts: int,
        retry_base_seconds: float,
        retry_max_seconds: float,
    ) -> tuple[bool, float]:
        record.attempts += 1
        record.last_error = f"{type(error).__name__}: {error}"
        if record.attempts >= max_attempts:
            source = self._path(self.pending_dir, record.point_id)
            destination = self._path(self.failed_dir, record.point_id)
            atomic_json(
                destination,
                {**asdict(record), "failed_at_ns": time.time_ns()},
            )
            if source.exists():
                source.unlink()
            return True, 0.0

        delay = min(
            retry_max_seconds,
            retry_base_seconds * (2 ** max(0, record.attempts - 1)),
        )
        record.next_attempt_ns = time.time_ns() + int(delay * 1_000_000_000)
        atomic_json(
            self._path(self.pending_dir, record.point_id), asdict(record)
        )
        return False, delay

    def counts(self) -> dict[str, int]:
        return {
            "pending": sum(1 for _ in self.pending_dir.glob("*.json")),
            "ingested": sum(1 for _ in self.ingested_dir.glob("*.json")),
            "failed": sum(1 for _ in self.failed_dir.glob("*.json")),
        }

    def next_retry_delay(self, default: float = 1.0) -> float:
        now = time.time_ns()
        waits: list[float] = []
        for path in self.pending_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                waits.append(max(0.0, (int(data["next_attempt_ns"]) - now) / 1e9))
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                return 0.0
        return min(waits, default=default)
