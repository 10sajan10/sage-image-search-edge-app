"""Self-contained helpers for the NRP benchmark-search notebook."""

from __future__ import annotations

import io
import csv
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import numpy as np


DATASETS = {
    "cloudbench": {
        "repo_id": "sagecontinuum/CloudBench",
        "revision": "85eb9925499efde29b982780991d96f9321d1faf",
        "id_column": "image_id",
    },
    "commonobjectsbench": {
        "repo_id": "sagecontinuum/CommonObjectsBench",
        "revision": "445ae940b5676b3362b900a1a0e8a2a05636cbfb",
        "id_column": "image_id",
    },
    "firebench": {
        "repo_id": "sagecontinuum/FireBench",
        "revision": "be2f646c88843ce271410234aa885b375bf3cdbf",
        "id_column": "image_id",
    },
    "inquire-small": {
        "repo_id": "sagecontinuum/INQUIRE-Benchmark-small",
        "revision": "0ca35458fc1c68f38fa5ad62c98b913ac0446cbd",
        "id_column": "inat24_image_id",
    },
    "sagebench": {
        "repo_id": "sagecontinuum/SageBench",
        "revision": "7428a2d887f48c4a217eee82c93a518d78734fd2",
        "id_column": "image_id",
    },
}

BENCHMARK_NAMES = {
    "firebench": "Firebench",
    "cloudbench": "Cloudbench",
    "inquire-small": "INQUIRE",
    "commonobjectsbench": "Commonobjectsbench",
    "sagebench": "Sagebench",
}

REFERENCE_RESULTS_REVISION = "049f6384d7e80c11666701bb320a09727a7d8133"
REFERENCE_VERSIONS = ("baseline", "v10", "v11", "v12")

EVALUATION_COLUMNS = {
    "cloudbench": ("query_id", "query_text", "image_id", "relevance_label"),
    "commonobjectsbench": ("query_id", "query_text", "image_id", "relevance_label"),
    "firebench": ("query_id", "query_text", "image_id", "relevance_label"),
    "inquire-small": ("query_id", "query", "inat24_image_id", "relevant"),
    "sagebench": ("query_id", "query_text", "image_id", "relevance_label"),
}

PORTABLE_INDEX_VERSION = 1


@dataclass(frozen=True)
class CaptionRecord:
    dataset: str
    image_id: str
    caption: str
    image_path: str


@dataclass
class PortableIndex:
    model_id: str
    records: list[CaptionRecord]
    image_vectors: np.ndarray
    caption_vectors: np.ndarray | None = None
    source: str = "unknown"

    def validate(self) -> None:
        count = len(self.records)
        if self.image_vectors.ndim != 2 or self.image_vectors.shape[0] != count:
            raise ValueError("image_vectors and records have different lengths")
        if self.caption_vectors is not None:
            if self.caption_vectors.ndim != 2 or self.caption_vectors.shape[0] != count:
                raise ValueError("caption_vectors and records have different lengths")
            if self.caption_vectors.shape[1] != self.image_vectors.shape[1]:
                raise ValueError("image and caption vector dimensions differ")


@dataclass
class BenchmarkQuery:
    query_id: str
    text: str
    relevant: set[str]


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def safe_image_path(image_id: str) -> Path:
    relative = PurePosixPath(image_id)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe image_id: {image_id!r}")
    return Path(*relative.with_suffix(".jpg").parts)


def download_benchmarks(root: Path, token: str | None = None) -> None:
    """Download all five datasets at the revisions used by edge_v1."""
    from huggingface_hub import snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    for local_name, spec in DATASETS.items():
        destination = root / local_name
        print(f"{local_name}: {spec['repo_id']}@{spec['revision']}")
        snapshot_download(
            repo_id=str(spec["repo_id"]),
            repo_type="dataset",
            revision=str(spec["revision"]),
            local_dir=destination,
            token=token or None,
            max_workers=4,
        )


def _image_bytes(value: dict[str, Any], parquet_path: Path) -> bytes:
    if value.get("bytes"):
        return value["bytes"]
    stored = value.get("path")
    if stored:
        for candidate in (
            parquet_path.parent / stored,
            parquet_path.parent.parent / stored,
        ):
            if candidate.is_file():
                return candidate.read_bytes()
    raise FileNotFoundError(f"Cannot resolve image in {parquet_path}")


def extract_benchmark_images(dataset_root: Path, image_root: Path) -> dict[str, int]:
    """Materialize one stable JPEG per unique benchmark image."""
    import pyarrow.parquet as pq
    from PIL import Image

    counts: dict[str, int] = {}
    for dataset, spec in DATASETS.items():
        seen: set[str] = set()
        written = existing = 0
        shards = sorted((dataset_root / dataset / "data").glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No Parquet shards in {dataset_root / dataset / 'data'}")
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            for batch in parquet.iter_batches(
                batch_size=64,
                columns=[str(spec["id_column"]), "image"],
            ):
                ids = batch.column(str(spec["id_column"])).to_pylist()
                images = batch.column("image").to_pylist()
                for raw_id, image_value in zip(ids, images):
                    image_id = str(raw_id)
                    if image_id in seen:
                        continue
                    seen.add(image_id)
                    output = image_root / dataset / safe_image_path(image_id)
                    if output.is_file():
                        existing += 1
                        continue
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(io.BytesIO(_image_bytes(image_value, shard))) as image:
                        image.convert("RGB").save(output, format="JPEG", quality=95)
                    written += 1
        counts[dataset] = len(seen)
        print(f"{dataset}: images={len(seen)} written={written} existing={existing}")
    return counts


def load_and_resolve_captions(
    captions_file: Path,
    image_root: Path,
    resolved_file: Path,
) -> list[CaptionRecord]:
    """Validate captions and write paths relative to the workspace image root."""
    if not captions_file.is_file():
        raise FileNotFoundError(
            f"Gemma caption asset not found: {captions_file}. "
            "Restore the bundled caption file or choose an NPZ backup."
        )

    records: dict[tuple[str, str], CaptionRecord] = {}
    with captions_file.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            dataset = str(row["dataset"])
            image_id = str(row["image_id"])
            caption = " ".join(str(row["caption"]).split())
            if dataset not in DATASETS:
                raise ValueError(f"{captions_file}:{line_number}: unknown dataset {dataset!r}")
            relative = Path(dataset) / safe_image_path(image_id)
            if not (image_root / relative).is_file():
                raise FileNotFoundError(f"Caption image is missing: {image_root / relative}")
            records[(dataset, image_id)] = CaptionRecord(
                dataset=dataset,
                image_id=image_id,
                caption=caption,
                image_path=relative.as_posix(),
            )

    ordered = [records[key] for key in sorted(records)]
    missing = sorted(set(DATASETS) - {record.dataset for record in ordered})
    if missing:
        raise ValueError(f"Caption export does not cover datasets: {missing}")

    resolved_file.parent.mkdir(parents=True, exist_ok=True)
    with resolved_file.open("w", encoding="utf-8") as output:
        for record in ordered:
            output.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
    print(f"Wrote {len(ordered):,} resolved captions to {resolved_file}")
    return ordered


def _metadata_bytes(index: PortableIndex) -> np.ndarray:
    metadata = {
        "format_version": PORTABLE_INDEX_VERSION,
        "model_id": index.model_id,
        "source": index.source,
        "records": [record.__dict__ for record in index.records],
    }
    return np.frombuffer(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    )


def save_portable_index(index: PortableIndex, path: Path) -> None:
    index.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    caption_vectors = (
        index.caption_vectors.astype(np.float32, copy=False)
        if index.caption_vectors is not None
        else np.empty((0, 0), dtype=np.float32)
    )
    np.savez(
        path,
        metadata=_metadata_bytes(index),
        image_vectors=index.image_vectors.astype(np.float32, copy=False),
        caption_vectors=caption_vectors,
    )
    print(f"Saved {len(index.records):,} records to {path} ({path.stat().st_size / 2**20:.1f} MiB)")


def load_portable_index(path: Path) -> PortableIndex:
    if not path.is_file():
        raise FileNotFoundError(f"Portable vector index not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
        if metadata.get("format_version") != PORTABLE_INDEX_VERSION:
            raise ValueError(f"Unsupported index format: {metadata.get('format_version')}")
        captions = archive["caption_vectors"]
        index = PortableIndex(
            model_id=str(metadata["model_id"]),
            source=str(metadata.get("source", "unknown")),
            records=[CaptionRecord(**record) for record in metadata["records"]],
            image_vectors=np.asarray(archive["image_vectors"], dtype=np.float32),
            caption_vectors=(
                np.asarray(captions, dtype=np.float32) if captions.size else None
            ),
        )
    index.validate()
    print(
        f"Loaded {len(index.records):,} records from {path}; "
        f"model={index.model_id}, source={index.source}"
    )
    return index


def save_sqlite_vector_database(index: PortableIndex, path: Path) -> None:
    """Persist the portable index in an embedded SQLite database (no server)."""
    index.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE images (
                position INTEGER PRIMARY KEY,
                dataset TEXT NOT NULL,
                image_id TEXT NOT NULL,
                caption TEXT NOT NULL,
                image_path TEXT NOT NULL,
                image_vector BLOB NOT NULL,
                caption_vector BLOB
            );
            CREATE UNIQUE INDEX images_dataset_id ON images(dataset, image_id);
            CREATE INDEX images_dataset ON images(dataset);
        """)
        metadata = {
            "format_version": str(PORTABLE_INDEX_VERSION),
            "model_id": index.model_id,
            "source": index.source,
            "vector_dtype": "float32",
            "image_dimension": str(index.image_vectors.shape[1]),
            "caption_dimension": str(
                index.caption_vectors.shape[1] if index.caption_vectors is not None else 0
            ),
            "record_count": str(len(index.records)),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        def database_rows():
            for position, record in enumerate(index.records):
                caption_blob = (
                    index.caption_vectors[position].astype(np.float32, copy=False).tobytes()
                    if index.caption_vectors is not None
                    else None
                )
                yield (
                    position,
                    record.dataset,
                    record.image_id,
                    record.caption,
                    record.image_path,
                    index.image_vectors[position].astype(np.float32, copy=False).tobytes(),
                    caption_blob,
                )
        connection.executemany(
            """INSERT INTO images(
                position, dataset, image_id, caption, image_path,
                image_vector, caption_vector
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            database_rows(),
        )
        connection.commit()
    finally:
        connection.close()
    temporary.replace(path)
    print(f"Saved {len(index.records):,} records to SQLite database {path}")


def load_sqlite_vector_database(path: Path) -> PortableIndex:
    """Load an embedded SQLite vector database into the exact-search engine."""
    if not path.is_file():
        raise FileNotFoundError(f"SQLite vector database not found: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        rows = connection.execute(
            """SELECT dataset, image_id, caption, image_path,
                      image_vector, caption_vector
               FROM images ORDER BY position"""
        ).fetchall()
    finally:
        connection.close()
    image_dimension = int(metadata["image_dimension"])
    caption_dimension = int(metadata["caption_dimension"])
    records = [CaptionRecord(*row[:4]) for row in rows]
    image_vectors = np.stack([
        np.frombuffer(row[4], dtype=np.float32, count=image_dimension).copy()
        for row in rows
    ])
    caption_vectors = (
        np.stack([
            np.frombuffer(row[5], dtype=np.float32, count=caption_dimension).copy()
            for row in rows
        ])
        if caption_dimension
        else None
    )
    index = PortableIndex(
        model_id=metadata["model_id"],
        records=records,
        image_vectors=image_vectors,
        caption_vectors=caption_vectors,
        source=metadata.get("source", "sqlite"),
    )
    index.validate()
    print(f"Loaded {len(records):,} records from SQLite database {path}")
    return index


class TextImageEncoder:
    """Use OpenCLIP for MobileCLIP2 and Transformers for the edge_v1 DFN model."""

    def __init__(self, model_id: str, device: str = "auto", token: str | None = None):
        import torch
        from huggingface_hub import snapshot_download

        self.torch = torch
        self.model_id = model_id
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )
        model_path = snapshot_download(repo_id=model_id, token=token or None)

        if "MobileCLIP" in model_id:
            import open_clip

            self.kind = "open_clip"
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                f"hf-hub:{model_id}"
            )
            self.tokenizer = open_clip.get_tokenizer(f"hf-hub:{model_id}")
        else:
            from transformers import AutoProcessor, CLIPModel

            self.kind = "transformers"
            self.processor = AutoProcessor.from_pretrained(model_path, token=token or None)
            self.model = CLIPModel.from_pretrained(model_path, token=token or None)
            self.preprocess = None
            self.tokenizer = None
        self.model = self.model.eval().to(self.device)
        print(f"Loaded {model_id} on {self.device}")

    def _normalize(self, tensor: Any) -> np.ndarray:
        tensor = tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return tensor.float().cpu().numpy()

    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            with self.torch.inference_mode():
                if self.kind == "open_clip":
                    tokens = self.tokenizer(batch).to(self.device)
                    features = self.model.encode_text(tokens)
                else:
                    inputs = self.processor(
                        text=batch,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    features = self.model.get_text_features(**inputs)
                    # transformers 5 returns BaseModelOutputWithPooling here;
                    # transformers 4 returned the projected tensor directly.
                    features = getattr(features, "pooler_output", features)
            chunks.append(self._normalize(features))
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    def encode_images(self, paths: Sequence[Path], batch_size: int = 64) -> np.ndarray:
        from PIL import Image

        chunks = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())
            with self.torch.inference_mode():
                if self.kind == "open_clip":
                    pixels = self.torch.stack([self.preprocess(image) for image in images])
                    features = self.model.encode_image(pixels.to(self.device))
                else:
                    inputs = self.processor(images=images, return_tensors="pt")
                    pixels = inputs["pixel_values"].to(self.device)
                    features = self.model.get_image_features(pixel_values=pixels)
                    features = getattr(features, "pooler_output", features)
            chunks.append(self._normalize(features))
            done = min(start + batch_size, len(paths))
            if done % (batch_size * 10) == 0 or done == len(paths):
                print(f"Embedded images: {done:,}/{len(paths):,}")
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def build_mobileclip_index(
    records: list[CaptionRecord],
    image_root: Path,
    output: Path,
    model_id: str,
    device: str,
    batch_size: int,
    token: str | None = None,
) -> PortableIndex:
    encoder = TextImageEncoder(model_id, device=device, token=token)
    image_vectors = encoder.encode_images(
        [image_root / record.image_path for record in records],
        batch_size=batch_size,
    )
    caption_vectors = encoder.encode_texts(
        [record.caption for record in records],
        batch_size=batch_size,
    )
    index = PortableIndex(
        model_id=model_id,
        records=records,
        image_vectors=image_vectors,
        caption_vectors=caption_vectors,
        source="mobileclip2",
    )
    save_portable_index(index, output)
    return index


def _relative_score(values: np.ndarray) -> np.ndarray:
    low = float(values.min())
    high = float(values.max())
    if math.isclose(low, high):
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _relative_topk_fusion(
    score_legs: Sequence[np.ndarray],
    weights: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply relative-score fusion over each retrieval leg's top-K candidates."""
    fused = np.zeros(len(score_legs[0]), dtype=np.float64)
    candidate_sets: list[np.ndarray] = []
    count = min(top_k, len(fused))
    for scores, weight in zip(score_legs, weights):
        if weight <= 0:
            continue
        candidates = np.argpartition(scores, -count)[-count:]
        candidate_sets.append(candidates)
        fused[candidates] += weight * _relative_score(scores[candidates])
    union = np.unique(np.concatenate(candidate_sets))
    ordered = union[np.argsort(fused[union])[::-1]][:count]
    return fused, ordered


def search_index(
    index: PortableIndex,
    encoder: TextImageEncoder,
    query: str,
    top_k: int = 25,
    image_weight: float = 0.70,
    caption_weight: float = 0.20,
    bm25_weight: float = 0.10,
) -> list[dict[str, Any]]:
    from rank_bm25 import BM25Okapi

    if encoder.model_id != index.model_id:
        raise ValueError(
            f"Index uses {index.model_id}, but query encoder is {encoder.model_id}"
        )
    weights = np.array([image_weight, caption_weight, bm25_weight], dtype=np.float64)
    if np.any(weights < 0) or not weights.sum():
        raise ValueError("Search weights must be non-negative and not all zero")
    if index.caption_vectors is None:
        weights[1] = 0
    weights /= weights.sum()

    query_vector = encoder.encode_texts([query])[0]
    image_scores = _relative_score(index.image_vectors @ query_vector)
    caption_scores = (
        _relative_score(index.caption_vectors @ query_vector)
        if index.caption_vectors is not None
        else np.zeros(len(index.records), dtype=np.float32)
    )
    tokenize = lambda text: re.findall(r"[a-z0-9]+", text.lower())
    bm25 = BM25Okapi([tokenize(record.caption) for record in index.records])
    bm25_scores = _relative_score(np.asarray(bm25.get_scores(tokenize(query))))
    count = min(top_k, len(index.records))
    fused, ordered = _relative_topk_fusion(
        [image_scores, caption_scores, bm25_scores], weights, count
    )
    return [
        {
            "rank": rank,
            "score": float(fused[position]),
            "image_score": float(image_scores[position]),
            "caption_score": float(caption_scores[position]),
            "bm25_score": float(bm25_scores[position]),
            **index.records[position].__dict__,
        }
        for rank, position in enumerate(ordered, 1)
    ]


def load_ground_truth(dataset_root: Path, dataset: str) -> list[BenchmarkQuery]:
    """Load the public relevance labels without using any prior edge results."""
    import pyarrow.parquet as pq

    query_column, text_column, image_column, relevance_column = EVALUATION_COLUMNS[dataset]
    grouped: dict[str, BenchmarkQuery] = {}
    shards = sorted((dataset_root / dataset / "data").glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No benchmark Parquet shards in {dataset_root / dataset / 'data'}")
    wanted = [query_column, text_column, image_column, relevance_column]
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=4096, columns=wanted):
            columns = [batch.column(name).to_pylist() for name in wanted]
            for raw_query_id, raw_text, raw_image_id, raw_relevance in zip(*columns):
                query_id = str(raw_query_id)
                query = grouped.setdefault(
                    query_id,
                    BenchmarkQuery(query_id=query_id, text=str(raw_text), relevant=set()),
                )
                if int(raw_relevance):
                    query.relevant.add(str(raw_image_id))
    return [grouped[key] for key in sorted(grouped)]


def _benchmark_scores(
    index: PortableIndex,
    encoder: TextImageEncoder,
    queries: Sequence[BenchmarkQuery],
    positions: np.ndarray,
    top_k: int,
    image_weight: float,
    caption_weight: float,
    bm25_weight: float,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Run exact, file-local retrieval and return per-query Success@K and RR."""
    from rank_bm25 import BM25Okapi

    records = [index.records[int(position)] for position in positions]
    image_vectors = index.image_vectors[positions]
    caption_vectors = (
        index.caption_vectors[positions] if index.caption_vectors is not None else None
    )
    tokenize = lambda value: re.findall(r"[a-z0-9]+", value.lower())
    bm25 = BM25Okapi([tokenize(record.caption) for record in records])
    query_vectors = encoder.encode_texts([query.text for query in queries], batch_size=batch_size)

    weights = np.asarray([image_weight, caption_weight, bm25_weight], dtype=np.float64)
    if np.any(weights < 0) or not weights.sum():
        raise ValueError("Benchmark weights must be non-negative and not all zero")
    if caption_vectors is None:
        weights[1] = 0
    weights /= weights.sum()

    rows: list[dict[str, Any]] = []
    count = min(top_k, len(records))
    for query, query_vector in zip(queries, query_vectors):
        image_scores = _relative_score(image_vectors @ query_vector)
        caption_scores = (
            _relative_score(caption_vectors @ query_vector)
            if caption_vectors is not None
            else np.zeros(len(records), dtype=np.float32)
        )
        bm25_scores = _relative_score(
            np.asarray(bm25.get_scores(tokenize(query.text)), dtype=np.float32)
        )
        _fused, ordered = _relative_topk_fusion(
            [image_scores, caption_scores, bm25_scores], weights, count
        )
        ranked_ids = [records[int(position)].image_id for position in ordered]
        ranked_vectors = image_vectors[ordered].astype(np.float64, copy=False)
        if len(ranked_vectors) > 1:
            norms = np.linalg.norm(ranked_vectors, axis=1, keepdims=True)
            ranked_vectors = ranked_vectors / np.maximum(norms, 1e-12)
            similarities = ranked_vectors @ ranked_vectors.T
            upper = np.triu_indices(len(ranked_vectors), k=1)
            diversity = 1.0 - float(np.mean(similarities[upper]))
        else:
            diversity = 0.0
        first_hit = next(
            (rank for rank, image_id in enumerate(ranked_ids, 1) if image_id in query.relevant),
            None,
        )
        rows.append({
            "query_id": query.query_id,
            "query": query.text,
            "hit": int(first_hit is not None),
            "reciprocal_rank": 1.0 / first_hit if first_hit else 0.0,
            "diversity": diversity,
            "first_relevant_rank": first_hit or "",
            "relevant_images": len(query.relevant),
            "returned_images": count,
        })
    return rows


def evaluate_benchmarks(
    index: PortableIndex,
    encoder: TextImageEncoder,
    dataset_root: Path,
    output_root: Path,
    top_k: int = 25,
    image_weight: float = 0.75,
    caption_weight: float = 0.0,
    bm25_weight: float = 0.25,
    batch_size: int = 64,
    system_version: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate a fresh benchmark run from the selected portable vector file."""
    if encoder.model_id != index.model_id:
        raise ValueError("The benchmark encoder does not match the portable index")
    output_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    all_query_rows: list[dict[str, Any]] = []
    for dataset, benchmark in BENCHMARK_NAMES.items():
        positions = np.asarray(
            [i for i, record in enumerate(index.records) if record.dataset == dataset],
            dtype=np.int64,
        )
        if not len(positions):
            raise ValueError(f"The portable index has no records for {dataset}")
        queries = load_ground_truth(dataset_root, dataset)
        rows = _benchmark_scores(
            index=index,
            encoder=encoder,
            queries=queries,
            positions=positions,
            top_k=top_k,
            image_weight=image_weight,
            caption_weight=caption_weight,
            bm25_weight=bm25_weight,
            batch_size=batch_size,
        )
        run_label = system_version or (
            f"edge_v1 (generated; {index.model_id.rsplit('/', 1)[-1]})"
        )
        for row in rows:
            row["benchmark"] = benchmark
            row["system_version"] = run_label
        all_query_rows.extend(rows)
        metrics_path = output_root / f"{dataset}_query_metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        mrr = float(np.mean([row["reciprocal_rank"] for row in rows]))
        success = float(np.mean([row["hit"] for row in rows]))
        diversity = float(np.mean([row["diversity"] for row in rows]))
        summary.append({
            "benchmark": benchmark,
            "system_version": run_label,
            "query_count": len(rows),
            "MRR": mrr,
            f"Success@{top_k}": success,
            f"Diversity@{top_k}": diversity,
            "Primary score": (mrr + success) / 2.0,
            "Primary + diversity score": (mrr + success + diversity) / 3.0,
        })
        print(
            f"{benchmark}: queries={len(rows)} MRR={mrr:.4f} "
            f"Success@{top_k}={success:.4f} Diversity@{top_k}={diversity:.4f}"
        )

    summary_path = output_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Fresh benchmark files written to {output_root}")
    return summary, all_query_rows


def load_reference_results(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load all upstream query rows and derive comparison metrics from them."""
    summaries: list[dict[str, Any]] = []
    all_query_rows: list[dict[str, Any]] = []
    for benchmark in BENCHMARK_NAMES.values():
        for version in REFERENCE_VERSIONS:
            path = root / benchmark / "results" / version / "query_eval_metrics.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Reference result missing: {path}")
            with path.open(encoding="utf-8") as source:
                raw_rows = list(csv.DictReader(source))
            normalized: list[dict[str, Any]] = []
            for raw in raw_rows:
                row = dict(raw)
                row.update({
                    "benchmark": benchmark,
                    "system_version": version,
                    "reciprocal_rank": (
                        float(raw["rerank_score_reciprocal_rank"])
                        if raw.get("rerank_score_reciprocal_rank", "").strip()
                        else np.nan
                    ),
                    "hit": float(raw["hit"]) if raw.get("hit", "").strip() else np.nan,
                    "diversity": (
                        float(raw["diversity"])
                        if raw.get("diversity", "").strip()
                        else np.nan
                    ),
                })
                normalized.append(row)
            all_query_rows.extend(normalized)
            mrr = float(np.nanmean([row["reciprocal_rank"] for row in normalized]))
            success = float(np.nanmean([row["hit"] for row in normalized]))
            diversity = float(np.nanmean([row["diversity"] for row in normalized]))
            summaries.append({
                "benchmark": benchmark,
                "system_version": version,
                "query_count": len(normalized),
                "MRR": mrr,
                "Success@25": success,
                "Diversity@25": diversity,
                "Primary score": (mrr + success) / 2.0,
                "Primary + diversity score": (mrr + success + diversity) / 3.0,
            })
    return summaries, all_query_rows


def benchmark_table(rows: Sequence[dict[str, Any]], benchmark: str):
    """Return one compact comparison table with both primary metrics and their mean."""
    import pandas as pd

    frame = pd.DataFrame(row for row in rows if row["benchmark"] == benchmark)
    frame["primary rank"] = frame["Primary score"].rank(method="dense", ascending=False).astype(int)
    frame["diversity rank"] = (
        frame["Primary + diversity score"].rank(method="dense", ascending=False).astype(int)
    )
    frame = frame.sort_values(
        ["Primary + diversity score", "system_version"], ascending=[False, True]
    )
    return frame[[
        "primary rank", "diversity rank", "system_version", "query_count",
        "MRR", "Success@25", "Diversity@25", "Primary score",
        "Primary + diversity score",
    ]]


def query_result_table(rows: Sequence[dict[str, Any]], benchmark: str):
    """Return every per-query row for one benchmark and every compared system."""
    import pandas as pd

    frame = pd.DataFrame(row for row in rows if row["benchmark"] == benchmark)
    columns = [
        "benchmark", "system_version", "query_id", "query", "hit",
        "reciprocal_rank", "diversity",
    ]
    return frame[columns].sort_values(["system_version", "query_id"]).reset_index(drop=True)


def overall_table(rows: Sequence[dict[str, Any]]):
    """Equal-weight the five benchmark summaries for each system."""
    import pandas as pd

    frame = pd.DataFrame(rows)
    overall = (
        frame.groupby("system_version", as_index=False)
        .agg(
            benchmark_count=("benchmark", "nunique"),
            MRR=("MRR", "mean"),
            **{
                "Success@25": ("Success@25", "mean"),
                "Diversity@25": ("Diversity@25", "mean"),
                "Primary score": ("Primary score", "mean"),
                "Primary + diversity score": ("Primary + diversity score", "mean"),
            },
        )
        .sort_values(
            ["Primary + diversity score", "system_version"], ascending=[False, True]
        )
        .reset_index(drop=True)
    )
    overall.insert(0, "rank", range(1, len(overall) + 1))
    return overall


def show_results(results: Sequence[dict[str, Any]], image_root: Path, width: int = 220) -> None:
    import html
    from IPython.display import HTML, display

    cards = []
    for row in results:
        image_path = image_root / row["image_path"]
        if image_path.is_file():
            import base64
            from PIL import Image

            buffer = io.BytesIO()
            with Image.open(image_path) as image:
                image.thumbnail((width * 2, 440))
                image.convert("RGB").save(buffer, format="JPEG", quality=82)
            source = f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"
            image_html = f'<img src="{source}" style="width:100%;max-height:220px;object-fit:contain">'
        else:
            image_html = "<em>Image file unavailable</em>"
        cards.append(
            f"""
            <article style="border:1px solid #ddd;border-radius:8px;padding:10px">
              <strong>#{row['rank']} · {html.escape(row['dataset'])}</strong>
              <div style="font-size:12px">
                score={row['score']:.4f} · image={row['image_score']:.4f} ·
                caption={row['caption_score']:.4f} · BM25={row['bm25_score']:.4f}
              </div>
              {image_html}
              <code style="font-size:11px">{html.escape(row['image_id'])}</code>
              <div style="font-size:11px;overflow-wrap:anywhere">
                path: {html.escape(row['image_path'])}
              </div>
              <p style="font-size:12px">{html.escape(row['caption'][:420])}</p>
            </article>
            """
        )
    display(HTML(
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax({width}px,1fr));'
        f'gap:12px">{"".join(cards)}</div>'
    ))
