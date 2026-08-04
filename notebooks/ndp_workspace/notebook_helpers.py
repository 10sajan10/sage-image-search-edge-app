"""Self-contained helpers for the NRP benchmark-search notebook."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

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
PORTABLE_INDEX_DOWNLOAD_BASE = (
    "https://media.githubusercontent.com/media/10sajan10/"
    "sage-image-search-edge-app/main/notebooks/ndp_workspace/data"
)
PORTABLE_INDEX_FILENAMES = {
    "edge_v1_benchmarks.npz",
    "edge_v2_benchmarks.npz",
    "edge_v3_benchmarks.npz",
}

MODEL_REVISIONS = {
    "apple/DFN5B-CLIP-ViT-H-14-378": "01b771ed0d1395ca5ffdd279897d665ebe00dfd2",
    "jinaai/jina-clip-v2": "e10d47f5691d0454a0fb5d13f46f2199b74cb436",
}
JINA_CODE_REVISION = "39e6a55ae971b59bea6e44675d237c99762e7ee2"


@dataclass(frozen=True)
class CaptionRecord:
    dataset: str
    image_id: str
    caption: str
    image_path: str


@dataclass
class PortableIndex:
    """Image vectors, caption text, and an optional dense caption leg.

    edge_v1 has no caption vectors: it and every bundled reference run
    (baseline/v10/v11/v12) use `clip_hybrid_query` against a single `clip`
    image vector, with the caption text searched lexically by BM25. So
    `caption_vectors` is None for edge_v1 and the caption leg is simply not
    used.

    edge_v2 and edge_v3 populate `caption_vectors`, so the leg stays first-class
    throughout. Edge v3 also records Jina's `retrieval.query` task. Requesting
    a caption weight against an index that has no caption vectors is an error
    rather than a silent reweighting --
    silently dropping the leg would make two runs look comparable when their
    fusion differed.
    """

    model_id: str
    records: list[CaptionRecord]
    image_vectors: np.ndarray
    caption_vectors: np.ndarray | None = None
    source: str = "unknown"
    query_task: str | None = None

    @property
    def has_caption_vectors(self) -> bool:
        return self.caption_vectors is not None

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
class MilvusVectorDatabase:
    """A local Milvus Lite file and its searchable collection."""

    path: Path
    collection_name: str
    model_id: str
    source: str
    record_count: int
    has_caption_vectors: bool
    query_task: str | None = None


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
        "query_task": index.query_task,
        "records": [record.__dict__ for record in index.records],
    }
    return np.frombuffer(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    )


def save_portable_index(index: PortableIndex, path: Path) -> None:
    index.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        metadata=_metadata_bytes(index),
        image_vectors=index.image_vectors.astype(np.float32, copy=False),
        # Empty for an image-only index such as edge_v1; populated in a
        # dual-vector export such as edge_v2 or edge_v3.
        caption_vectors=(
            index.caption_vectors.astype(np.float32, copy=False)
            if index.caption_vectors is not None
            else np.empty((0, 0), dtype=np.float32)
        ),
    )
    print(f"Saved {len(index.records):,} records to {path} ({path.stat().st_size / 2**20:.1f} MiB)")


def _lfs_pointer_metadata(path: Path) -> tuple[str, int] | None:
    """Return the SHA-256 and byte size stored in a Git LFS pointer."""
    with path.open("rb") as source:
        content = source.read(1024)
    if not content.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return None
    text = content.decode("ascii")
    oid = re.search(r"^oid sha256:([0-9a-f]{64})$", text, re.MULTILINE)
    size = re.search(r"^size ([0-9]+)$", text, re.MULTILINE)
    if oid is None or size is None:
        raise ValueError(f"Invalid Git LFS pointer: {path}")
    return oid.group(1), int(size.group(1))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 2**20):
            digest.update(chunk)
    return digest.hexdigest()


def _materialize_portable_index(path: Path) -> Path:
    """Return a verified NPZ, downloading one when given an LFS pointer."""
    metadata = _lfs_pointer_metadata(path)
    if metadata is None:
        return path
    if path.name not in PORTABLE_INDEX_FILENAMES:
        raise RuntimeError(
            f"{path} is a Git LFS pointer and has no automatic download source."
        )

    expected_sha256, expected_size = metadata
    url = f"{PORTABLE_INDEX_DOWNLOAD_BASE}/{path.name}"
    cache = path.parent / ".portable_index_cache" / path.name
    cache.parent.mkdir(parents=True, exist_ok=True)
    if (
        cache.is_file()
        and cache.stat().st_size == expected_size
        and _file_sha256(cache) == expected_sha256
    ):
        print(f"Using verified local cache for {path.name}")
        return cache

    temporary = cache.with_suffix(cache.suffix + ".download")
    digest = hashlib.sha256()
    received = 0
    print(
        f"{path.name}: this clone contains a small Git LFS pointer; "
        f"downloading the {expected_size / 2**20:.1f} MiB index automatically."
    )
    try:
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            temporary.open("wb") as output,
        ):
            while chunk := response.read(8 * 2**20):
                output.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                print(
                    f"\r{path.name}: {received / 2**20:.1f} / "
                    f"{expected_size / 2**20:.1f} MiB",
                    end="",
                    flush=True,
                )
        print()
        if received != expected_size:
            raise RuntimeError(
                f"Incomplete download for {path.name}: expected {expected_size:,} "
                f"bytes, received {received:,}."
            )
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {path.name}: expected {expected_sha256}, "
                f"received {actual_sha256}."
            )
        temporary.replace(cache)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Downloaded and verified {path.name}")
    return cache


def load_portable_index(path: Path) -> PortableIndex:
    if not path.is_file():
        raise FileNotFoundError(f"Portable vector index not found: {path}")
    resolved_path = _materialize_portable_index(path)
    with np.load(resolved_path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
        if metadata.get("format_version") != PORTABLE_INDEX_VERSION:
            raise ValueError(f"Unsupported index format: {metadata.get('format_version')}")
        captions = (
            archive["caption_vectors"]
            if "caption_vectors" in archive.files
            else np.empty((0, 0), dtype=np.float32)
        )
        index = PortableIndex(
            model_id=str(metadata["model_id"]),
            source=str(metadata.get("source", "unknown")),
            query_task=metadata.get("query_task"),
            records=[CaptionRecord(**record) for record in metadata["records"]],
            image_vectors=np.asarray(archive["image_vectors"], dtype=np.float32),
            caption_vectors=(
                np.asarray(captions, dtype=np.float32) if captions.size else None
            ),
        )
    index.validate()
    task_label = f", query_task={index.query_task}" if index.query_task else ""
    print(
        f"Loaded {len(index.records):,} records from {resolved_path}; "
        f"model={index.model_id}, source={index.source}{task_label}"
    )
    return index


def ensure_milvus_vector_database(
    index: PortableIndex,
    path: Path,
    collection_name: str = "images",
    batch_size: int = 512,
) -> MilvusVectorDatabase:
    """Create or reuse a file-backed Milvus Lite collection with native BM25."""
    from pymilvus import DataType, Function, FunctionType, MilvusClient

    index.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    client = MilvusClient(str(path))
    expected = {
        "model_id": index.model_id,
        "source": index.source,
        "record_count": len(index.records),
        "has_caption_vectors": index.has_caption_vectors,
        "query_task": index.query_task,
    }
    if client.has_collection(collection_name):
        client.load_collection(collection_name)
        rows = client.query(
            collection_name,
            filter="position == 0",
            output_fields=["model_id", "source", "has_caption_vector"],
            limit=1,
        )
        count = int(client.get_collection_stats(collection_name)["row_count"])
        reusable = bool(rows) and {
            "model_id": rows[0]["model_id"],
            "source": rows[0]["source"],
            "record_count": count,
            "has_caption_vectors": bool(rows[0]["has_caption_vector"]),
        } == {key: value for key, value in expected.items() if key != "query_task"}
        if reusable:
            client.close()
            print(f"Reusing {count:,} records from Milvus Lite database {path}")
            return MilvusVectorDatabase(path, collection_name, **expected)
        client.drop_collection(collection_name)

    dimension = int(index.image_vectors.shape[1])
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("position", DataType.INT64, is_primary=True)
    schema.add_field("dataset", DataType.VARCHAR, max_length=64)
    schema.add_field("image_id", DataType.VARCHAR, max_length=2048)
    schema.add_field(
        "caption", DataType.VARCHAR, max_length=65535, enable_analyzer=True
    )
    schema.add_field("image_path", DataType.VARCHAR, max_length=4096)
    schema.add_field("model_id", DataType.VARCHAR, max_length=512)
    schema.add_field("source", DataType.VARCHAR, max_length=64)
    schema.add_field("has_caption_vector", DataType.BOOL)
    schema.add_field("image_vector", DataType.FLOAT_VECTOR, dim=dimension)
    if index.has_caption_vectors:
        schema.add_field("caption_vector", DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field("caption_bm25", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(
        name="caption_bm25_function",
        input_field_names=["caption"],
        output_field_names=["caption_bm25"],
        function_type=FunctionType.BM25,
    ))

    index_params = client.prepare_index_params()
    index_params.add_index("image_vector", index_type="FLAT", metric_type="IP")
    if index.has_caption_vectors:
        index_params.add_index("caption_vector", index_type="FLAT", metric_type="IP")
    index_params.add_index(
        "caption_bm25",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )
    client.create_collection(
        collection_name,
        schema=schema,
        index_params=index_params,
        consistency_level="Strong",
    )
    for start in range(0, len(index.records), batch_size):
        rows = []
        for position in range(start, min(start + batch_size, len(index.records))):
            record = index.records[position]
            row = {
                "position": position,
                "dataset": record.dataset,
                "image_id": record.image_id,
                "caption": record.caption,
                "image_path": record.image_path,
                "model_id": index.model_id,
                "source": index.source,
                "has_caption_vector": index.has_caption_vectors,
                "image_vector": index.image_vectors[position].tolist(),
            }
            if index.has_caption_vectors:
                row["caption_vector"] = index.caption_vectors[position].tolist()
            rows.append(row)
        client.insert(collection_name, rows)
        print(
            f"\r{index.source}: populated {min(start + batch_size, len(index.records)):,}"
            f"/{len(index.records):,} Milvus records",
            end="",
            flush=True,
        )
    print()
    client.flush(collection_name)
    client.close()
    print(f"Saved {len(index.records):,} records to Milvus Lite database {path}")
    return MilvusVectorDatabase(path, collection_name, **expected)


class TextImageEncoder:
    """Load the model-specific text encoder recorded in a portable index."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        token: str | None = None,
        query_task: str | None = None,
    ):
        import torch
        from huggingface_hub import snapshot_download

        self.torch = torch
        self.model_id = model_id
        self.query_task = query_task
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )
        revision = MODEL_REVISIONS.get(model_id)
        model_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            token=token or None,
        )

        if model_id == "jinaai/jina-clip-v2":
            from transformers import AutoModel

            self.processor = None
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                code_revision=JINA_CODE_REVISION,
                use_text_flash_attn=False,
                use_vision_xformers=False,
                token=token or None,
            )
        else:
            from transformers import AutoProcessor, CLIPModel

            self.processor = AutoProcessor.from_pretrained(
                model_path,
                token=token or None,
                use_fast=False,
            )
            self.model = CLIPModel.from_pretrained(model_path, token=token or None)
        self.model = self.model.eval().to(self.device)
        task_label = f", query_task={query_task}" if query_task else ""
        print(f"Loaded {model_id} on {self.device}{task_label}")

    def _normalize(self, tensor: Any) -> np.ndarray:
        tensor = tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return tensor.float().cpu().numpy()

    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        if self.model_id == "jinaai/jina-clip-v2":
            with self.torch.inference_mode():
                features = self.model.encode_text(
                    list(texts),
                    task=self.query_task,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    truncate_dim=1024,
                )
            return np.asarray(features, dtype=np.float32)

        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            with self.torch.inference_mode():
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
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Apply relative-score fusion over each retrieval leg's top-K candidates.

    Also returns each leg's weighted contribution. Contributions are
    normalized over the candidate subset, exactly like the fused total, so a
    reported per-leg breakdown always sums to the reported score. Reporting a
    corpus-wide normalization beside a candidate-normalized total would not
    add up.
    """
    length = len(score_legs[0])
    count = min(top_k, length)
    if count < 1:
        raise ValueError("Cannot fuse an empty index")
    fused = np.zeros(length, dtype=np.float64)
    contributions = [np.zeros(length, dtype=np.float64) for _ in score_legs]
    candidate_sets: list[np.ndarray] = []
    for leg, (scores, weight) in enumerate(zip(score_legs, weights)):
        if weight <= 0:
            continue
        candidates = np.argpartition(scores, -count)[-count:]
        candidate_sets.append(candidates)
        contributions[leg][candidates] = weight * _relative_score(scores[candidates])
        fused[candidates] += contributions[leg][candidates]
    union = np.unique(np.concatenate(candidate_sets))
    ordered = union[np.argsort(fused[union])[::-1]][:count]
    return fused, ordered, contributions


def _relative_full_corpus_fusion(
    score_legs: Sequence[np.ndarray],
    weights: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Min-max normalize every leg over the complete corpus, then combine.

    This is edge_v2's benchmark method. Unlike edge_v1's native Weaviate
    hybrid candidate fusion, both dense matrices are available locally, so
    all candidates participate in each leg before the top-K is selected.
    """
    length = len(score_legs[0])
    count = min(top_k, length)
    if count < 1:
        raise ValueError("Cannot fuse an empty index")
    contributions = [
        weight * _relative_score(scores) if weight > 0 else np.zeros(length)
        for scores, weight in zip(score_legs, weights)
    ]
    fused = np.sum(contributions, axis=0)
    ordered = np.argsort(-fused, kind="stable")[:count]
    return fused, ordered, contributions


def resolve_weights(
    index: PortableIndex,
    image_weight: float,
    caption_weight: float,
    bm25_weight: float,
) -> np.ndarray:
    """Normalize the three leg weights, refusing an unsatisfiable caption leg.

    A missing caption leg is an error rather than a silent reweighting: quietly
    zeroing caption_weight would let an edge_v1 index and an edge_v2 index
    report the same configured weights while fusing differently, which makes
    their benchmark scores look comparable when they are not.
    """
    weights = np.asarray(
        [image_weight, caption_weight, bm25_weight], dtype=np.float64
    )
    if np.any(weights < 0) or not weights.sum():
        raise ValueError("Weights must be non-negative and not all zero")
    if caption_weight > 0 and not index.has_caption_vectors:
        raise ValueError(
            f"caption_weight={caption_weight} was requested, but the "
            f"{index.source!r} index has no caption vectors. edge_v1 searches "
            "captions with BM25 only. A dense caption leg requires an index "
            "export that already contains caption vectors."
        )
    return weights / weights.sum()


def search_index(
    index: PortableIndex,
    encoder: TextImageEncoder,
    query: str,
    top_k: int = 25,
    image_weight: float = 0.40,
    caption_weight: float = 0.0,
    bm25_weight: float = 0.60,
    fusion_mode: str = "topk",
) -> list[dict[str, Any]]:
    """Fuse the image-vector leg, the optional caption-vector leg, and BM25."""
    from rank_bm25 import BM25Okapi

    if encoder.model_id != index.model_id:
        raise ValueError(
            f"Index uses {index.model_id}, but query encoder is {encoder.model_id}"
        )
    if encoder.query_task != index.query_task:
        raise ValueError(
            f"Index query task is {index.query_task!r}, but encoder task is "
            f"{encoder.query_task!r}"
        )
    weights = resolve_weights(index, image_weight, caption_weight, bm25_weight)

    query_vector = encoder.encode_texts([query])[0]
    image_similarity = index.image_vectors @ query_vector
    caption_similarity = (
        index.caption_vectors @ query_vector
        if index.has_caption_vectors
        else np.zeros(len(index.records), dtype=np.float64)
    )
    tokenize = lambda text: re.findall(r"[a-z0-9]+", text.lower())
    bm25 = BM25Okapi([tokenize(record.caption) for record in index.records])
    bm25_raw = np.asarray(bm25.get_scores(tokenize(query)), dtype=np.float64)
    count = min(top_k, len(index.records))
    fusion = (
        _relative_full_corpus_fusion
        if fusion_mode == "full_corpus"
        else _relative_topk_fusion
    )
    if fusion_mode not in {"topk", "full_corpus"}:
        raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
    fused, ordered, contributions = fusion(
        [image_similarity, caption_similarity, bm25_raw], weights, count
    )
    return [
        {
            "rank": rank,
            # score == image_score + caption_score + bm25_score, by construction.
            "score": float(fused[position]),
            "image_score": float(contributions[0][position]),
            "caption_score": float(contributions[1][position]),
            "bm25_score": float(contributions[2][position]),
            "image_similarity": float(image_similarity[position]),
            "caption_similarity": float(caption_similarity[position]),
            "bm25_raw": float(bm25_raw[position]),
            **index.records[position].__dict__,
        }
        for rank, position in enumerate(ordered, 1)
    ]


def search_milvus_database(
    database: MilvusVectorDatabase,
    encoder: TextImageEncoder,
    query: str,
    top_k: int = 25,
    image_weight: float = 0.40,
    caption_weight: float = 0.0,
    bm25_weight: float = 0.60,
) -> list[dict[str, Any]]:
    """Search dense vectors and Milvus-native BM25, then fuse their candidates."""
    from pymilvus import MilvusClient

    if encoder.model_id != database.model_id:
        raise ValueError(
            f"Database uses {database.model_id}, but query encoder is {encoder.model_id}"
        )
    if encoder.query_task != database.query_task:
        raise ValueError(
            f"Database query task is {database.query_task!r}, but encoder task is "
            f"{encoder.query_task!r}"
        )
    weights = np.asarray(
        [image_weight, caption_weight, bm25_weight], dtype=np.float64
    )
    if np.any(weights < 0) or not weights.sum():
        raise ValueError("Weights must be non-negative and not all zero")
    if caption_weight > 0 and not database.has_caption_vectors:
        raise ValueError(
            f"caption_weight={caption_weight} was requested, but "
            f"{database.source!r} has no caption vectors"
        )
    weights /= weights.sum()

    query_vector = encoder.encode_texts([query])[0].tolist()
    output_fields = ["dataset", "image_id", "caption", "image_path"]
    client = MilvusClient(str(database.path))
    try:
        client.load_collection(database.collection_name)
        searches = [
            ("image_vector", [query_vector], float(weights[0])),
            ("caption_vector", [query_vector], float(weights[1])),
            ("caption_bm25", [query], float(weights[2])),
        ]
        leg_hits: list[list[dict[str, Any]]] = []
        entities: dict[int, dict[str, Any]] = {}
        for field, data, weight in searches:
            if not weight or (field == "caption_vector" and not database.has_caption_vectors):
                leg_hits.append([])
                continue
            hits = list(client.search(
                collection_name=database.collection_name,
                data=data,
                anns_field=field,
                limit=min(top_k, database.record_count),
                output_fields=output_fields,
            )[0])
            leg_hits.append(hits)
            for hit in hits:
                entities[int(hit["position"])] = dict(hit["entity"])
    finally:
        client.close()

    raw_by_leg: list[dict[int, float]] = []
    contribution_by_leg: list[dict[int, float]] = []
    for hits, weight in zip(leg_hits, weights):
        raw = {int(hit["position"]): float(hit["distance"]) for hit in hits}
        raw_by_leg.append(raw)
        if not raw or weight <= 0:
            contribution_by_leg.append({})
            continue
        positions = list(raw)
        values = np.asarray([raw[position] for position in positions], dtype=np.float64)
        normalized = _relative_score(values)
        contribution_by_leg.append({
            position: float(weight * score)
            for position, score in zip(positions, normalized)
        })

    fused = {
        position: sum(leg.get(position, 0.0) for leg in contribution_by_leg)
        for position in entities
    }
    ordered = sorted(fused, key=lambda position: (-fused[position], position))[:top_k]
    results = []
    for rank, position in enumerate(ordered, 1):
        entity = entities[position]
        results.append({
            "rank": rank,
            "score": fused[position],
            "image_score": contribution_by_leg[0].get(position, 0.0),
            "caption_score": contribution_by_leg[1].get(position, 0.0),
            "bm25_score": contribution_by_leg[2].get(position, 0.0),
            "image_similarity": raw_by_leg[0].get(position, 0.0),
            "caption_similarity": raw_by_leg[1].get(position, 0.0),
            "bm25_raw": raw_by_leg[2].get(position, 0.0),
            **entity,
        })
    return results


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
    fusion_mode: str,
) -> list[dict[str, Any]]:
    """Run exact, file-local retrieval and return per-query Success@K and RR."""
    from rank_bm25 import BM25Okapi

    records = [index.records[int(position)] for position in positions]
    image_vectors = index.image_vectors[positions]
    caption_vectors = (
        index.caption_vectors[positions] if index.has_caption_vectors else None
    )
    tokenize = lambda value: re.findall(r"[a-z0-9]+", value.lower())
    bm25 = BM25Okapi([tokenize(record.caption) for record in records])
    query_vectors = encoder.encode_texts([query.text for query in queries], batch_size=batch_size)

    weights = resolve_weights(index, image_weight, caption_weight, bm25_weight)

    rows: list[dict[str, Any]] = []
    count = min(top_k, len(records))
    zeros = np.zeros(len(records), dtype=np.float64)
    for query, query_vector in zip(queries, query_vectors):
        image_scores = _relative_score(image_vectors @ query_vector)
        caption_scores = (
            _relative_score(caption_vectors @ query_vector)
            if caption_vectors is not None
            else zeros
        )
        bm25_scores = _relative_score(
            np.asarray(bm25.get_scores(tokenize(query.text)), dtype=np.float32)
        )
        fusion = (
            _relative_full_corpus_fusion
            if fusion_mode == "full_corpus"
            else _relative_topk_fusion
        )
        if fusion_mode not in {"topk", "full_corpus"}:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
        _fused, ordered, _contributions = fusion(
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
    image_weight: float = 0.40,
    caption_weight: float = 0.0,
    bm25_weight: float = 0.60,
    batch_size: int = 64,
    system_version: str | None = None,
    fusion_mode: str = "topk",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate a fresh benchmark run from the selected portable vector file."""
    if encoder.model_id != index.model_id:
        raise ValueError("The benchmark encoder does not match the portable index")
    if encoder.query_task != index.query_task:
        raise ValueError("The benchmark encoder query task does not match the portable index")
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
            fusion_mode=fusion_mode,
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


def evaluate_edge_v1_alpha_sweep(
    index: PortableIndex,
    encoder: TextImageEncoder,
    dataset_root: Path,
    alphas: Sequence[float],
    top_k: int = 25,
    batch_size: int = 64,
):
    """Evaluate Edge v1 overall metrics as alpha moves from BM25 to image search.

    ``alpha`` is the image-vector weight and ``1 - alpha`` is the caption-BM25
    weight. Query embeddings, BM25 scores, and each leg's top-K candidates are
    computed once and reused for every alpha.
    """
    import pandas as pd
    from rank_bm25 import BM25Okapi

    if encoder.model_id != index.model_id:
        raise ValueError("The alpha-sweep encoder does not match the index")
    if index.has_caption_vectors:
        raise ValueError("The Edge v1 alpha sweep expects an image-only dense index")
    alpha_values = sorted({float(alpha) for alpha in alphas})
    if not alpha_values or any(alpha < 0 or alpha > 1 for alpha in alpha_values):
        raise ValueError("Alpha values must be between 0 and 1 inclusive")

    per_alpha: dict[float, list[dict[str, float]]] = {
        alpha: [] for alpha in alpha_values
    }
    tokenize = lambda value: re.findall(r"[a-z0-9]+", value.lower())
    for dataset, benchmark_name in BENCHMARK_NAMES.items():
        positions = np.asarray(
            [i for i, record in enumerate(index.records) if record.dataset == dataset],
            dtype=np.int64,
        )
        records = [index.records[int(position)] for position in positions]
        image_vectors = index.image_vectors[positions]
        queries = load_ground_truth(dataset_root, dataset)
        query_vectors = encoder.encode_texts(
            [query.text for query in queries], batch_size=batch_size
        )
        bm25 = BM25Okapi([tokenize(record.caption) for record in records])
        count = min(top_k, len(records))
        metrics = {
            alpha: {"rr": [], "hit": [], "diversity": []}
            for alpha in alpha_values
        }
        for query, query_vector in zip(queries, query_vectors):
            image_scores = _relative_score(image_vectors @ query_vector)
            bm25_scores = _relative_score(np.asarray(
                bm25.get_scores(tokenize(query.text)), dtype=np.float32
            ))
            image_candidates = np.argpartition(image_scores, -count)[-count:]
            bm25_candidates = np.argpartition(bm25_scores, -count)[-count:]
            normalized_image = np.zeros(len(records), dtype=np.float64)
            normalized_bm25 = np.zeros(len(records), dtype=np.float64)
            normalized_image[image_candidates] = _relative_score(
                image_scores[image_candidates]
            )
            normalized_bm25[bm25_candidates] = _relative_score(
                bm25_scores[bm25_candidates]
            )
            for alpha in alpha_values:
                active = []
                if alpha > 0:
                    active.append(image_candidates)
                if alpha < 1:
                    active.append(bm25_candidates)
                candidates = np.unique(np.concatenate(active))
                fused = (
                    alpha * normalized_image[candidates]
                    + (1.0 - alpha) * normalized_bm25[candidates]
                )
                ordered = candidates[np.argsort(fused, kind="stable")[::-1]][:count]
                ranked_ids = [records[int(position)].image_id for position in ordered]
                first_hit = next(
                    (
                        rank
                        for rank, image_id in enumerate(ranked_ids, 1)
                        if image_id in query.relevant
                    ),
                    None,
                )
                ranked_vectors = image_vectors[ordered].astype(np.float64, copy=False)
                if len(ranked_vectors) > 1:
                    norms = np.linalg.norm(ranked_vectors, axis=1, keepdims=True)
                    ranked_vectors = ranked_vectors / np.maximum(norms, 1e-12)
                    similarities = ranked_vectors @ ranked_vectors.T
                    upper = np.triu_indices(len(ranked_vectors), k=1)
                    diversity = 1.0 - float(np.mean(similarities[upper]))
                else:
                    diversity = 0.0
                metrics[alpha]["rr"].append(1.0 / first_hit if first_hit else 0.0)
                metrics[alpha]["hit"].append(float(first_hit is not None))
                metrics[alpha]["diversity"].append(diversity)

        for alpha in alpha_values:
            per_alpha[alpha].append({
                "MRR": float(np.mean(metrics[alpha]["rr"])),
                f"Success@{top_k}": float(np.mean(metrics[alpha]["hit"])),
                f"Diversity@{top_k}": float(np.mean(metrics[alpha]["diversity"])),
            })
        print(f"Alpha sweep prepared {benchmark_name}")

    rows = []
    for alpha in alpha_values:
        mrr = float(np.mean([row["MRR"] for row in per_alpha[alpha]]))
        success = float(np.mean([
            row[f"Success@{top_k}"] for row in per_alpha[alpha]
        ]))
        diversity = float(np.mean([
            row[f"Diversity@{top_k}"] for row in per_alpha[alpha]
        ]))
        rows.append({
            "alpha": alpha,
            "image_weight": alpha,
            "bm25_weight": 1.0 - alpha,
            "MRR": mrr,
            f"Success@{top_k}": success,
            f"Diversity@{top_k}": diversity,
            "Primary score": (mrr + success) / 2.0,
            "Primary + diversity score": (mrr + success + diversity) / 3.0,
        })
    return pd.DataFrame(rows)


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


def score_bar_charts(
    rows: Sequence[dict[str, Any]], benchmark: str | None = None
):
    """Plot Primary and Primary + Diversity scores side by side for all systems."""
    import matplotlib.pyplot as plt

    frame = overall_table(rows) if benchmark is None else benchmark_table(rows, benchmark)
    frame = frame.sort_values(
        ["Primary score", "system_version"], ascending=[False, True]
    ).reset_index(drop=True)
    labels = [
        str(value).split(" (generated;", 1)[0]
        for value in frame["system_version"]
    ]
    colors = [
        "#f28e2b" if label.startswith("edge_") else "#4e79a7"
        for label in labels
    ]
    positions = np.arange(len(frame))
    figure, axes = plt.subplots(
        1, 2, figsize=(14, max(4.0, 0.65 * len(frame))), sharey=True
    )
    title_prefix = benchmark or "Overall"
    for axis, column, title in (
        (axes[0], "Primary score", "Primary (50% MRR, 50% Success@25)"),
        (
            axes[1],
            "Primary + diversity score",
            "Primary + Diversity (equal thirds)",
        ),
    ):
        values = frame[column].to_numpy(dtype=float)
        bars = axis.barh(positions, values, color=colors)
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlim(0, 1)
        axis.set_xlabel("Composite score")
        axis.set_title(f"{title_prefix}\n{title}")
        axis.grid(axis="x", alpha=0.25)
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3)
    figure.tight_layout()
    plt.show()
    return figure, axes


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


def load_result_images(
    results: Sequence[dict[str, Any]],
    dataset_root: Path,
) -> dict[tuple[str, str], bytes]:
    """Read only result images from their Parquet row groups, without extraction."""
    import pyarrow.parquet as pq

    wanted: dict[str, set[str]] = {}
    for row in results:
        wanted.setdefault(str(row["dataset"]), set()).add(str(row["image_id"]))
    found: dict[tuple[str, str], bytes] = {}
    for dataset, pending in wanted.items():
        id_column = str(DATASETS[dataset]["id_column"])
        shards = sorted((dataset_root / dataset / "data").glob("*.parquet"))
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            for row_group in range(parquet.metadata.num_row_groups):
                ids = parquet.read_row_group(
                    row_group, columns=[id_column]
                ).column(id_column).to_pylist()
                matches = {
                    str(image_id): position
                    for position, image_id in enumerate(ids)
                    if str(image_id) in pending
                }
                if not matches:
                    continue
                images = parquet.read_row_group(
                    row_group, columns=["image"]
                ).column("image").to_pylist()
                for image_id, position in matches.items():
                    found[(dataset, image_id)] = _image_bytes(images[position], shard)
                    pending.remove(image_id)
                if not pending:
                    break
            if not pending:
                break
        if pending:
            print(f"Warning: {len(pending)} result images were not found in {dataset}")
    print(f"Loaded {len(found):,} requested images directly from Parquet")
    return found


def show_results(
    results: Sequence[dict[str, Any]],
    images: dict[tuple[str, str], bytes],
    width: int = 220,
) -> None:
    import html

    from IPython.display import HTML, display

    cards = []
    # Only name the caption-vector leg when it actually contributed, so an
    # edge_v1 card does not imply a leg the index does not have.
    show_caption_leg = any(row.get("caption_score", 0.0) for row in results)
    for row in results:
        image_bytes = images.get((str(row["dataset"]), str(row["image_id"])))
        if image_bytes is not None:
            import base64

            from PIL import Image

            buffer = io.BytesIO()
            with Image.open(io.BytesIO(image_bytes)) as image:
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
                score={row['score']:.4f} = image {row['image_score']:.4f}
                {f"+ caption {row['caption_score']:.4f} " if show_caption_leg else ""}
                + BM25 {row['bm25_score']:.4f}
              </div>
              <div style="font-size:11px;color:#666">
                image cosine={row['image_similarity']:.4f}
                {f"· caption cosine={row['caption_similarity']:.4f} " if show_caption_leg else ""}
                · BM25 raw={row['bm25_raw']:.2f}
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
