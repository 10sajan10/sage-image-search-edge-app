"""Shared captioning, embedding, Qdrant, and weighted-search primitives."""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from image_search.config import CaptionConfig, CommonConfig, SearchConfig

LOGGER = logging.getLogger("image_search.pipeline")
LEGS = ("image", "caption", "bm25")
RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}
CAPTION_PREFIXES = (
    "this image shows",
    "this image depicts",
    "the image shows",
    "the image depicts",
    "here is a caption",
    "here is",
    "caption:",
)


class ApiError(RuntimeError):
    def __init__(self, url: str, status: int, body: str):
        super().__init__(f"{url}: HTTP {status}: {body[:300]}")
        self.url = url
        self.status = status
        self.body = body


class HttpClient:
    """Small JSON client with bounded retries for edge-friendly dependencies."""

    def __init__(self, retries: int, backoff_seconds: float):
        self.retries = retries
        self.backoff_seconds = backoff_seconds

    def json(
        self,
        url: str,
        method: str = "GET",
        payload: Any = None,
        timeout: int = 60,
    ) -> Any:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                url, data=data, method=method, headers=headers
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = response.read()
                return json.loads(body) if body else None
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", "replace")
                api_error = ApiError(url, error.code, body)
                if error.code not in RETRYABLE_HTTP_STATUS:
                    raise api_error from error
                last_error = api_error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = error

            if attempt < self.retries:
                delay = self.backoff_seconds * (2**attempt)
                LOGGER.warning(
                    "dependency request failed; retrying",
                    extra={
                        "url": url,
                        "attempt": attempt + 1,
                        "delay_seconds": delay,
                        "error": str(last_error),
                    },
                )
                time.sleep(delay)
        raise RuntimeError(
            f"{url}: failed after {self.retries + 1} attempts: {last_error}"
        ) from last_error


class QdrantStore:
    """Qdrant collection contract shared by ingestion and search."""

    def __init__(self, config: CommonConfig):
        self.config = config
        self.http = HttpClient(config.http_retries, config.http_backoff_seconds)

    def request(
        self, path: str, method: str = "GET", payload: Any = None, timeout: int = 60
    ) -> Any:
        return self.http.json(
            f"{self.config.qdrant_url}{path}", method, payload, timeout
        )

    def collection_info(self) -> dict:
        return self.request(f"/collections/{self.config.collection}")["result"]

    def ensure_collection(self) -> None:
        """Create or strictly validate the fixed three-vector schema."""
        try:
            info = self.collection_info()
        except ApiError as error:
            if error.status != 404:
                raise
            self._create_collection()
            return
        self._validate_collection_info(info)

    def validate_collection(self) -> None:
        """Validate an existing collection without creating or modifying it."""
        self._validate_collection_info(self.collection_info())

    def _validate_collection_info(self, info: dict) -> None:
        params = info["config"]["params"]
        vectors = params.get("vectors") or {}
        sparse = params.get("sparse_vectors") or {}
        missing = {"image", "caption"} - set(vectors)
        if missing:
            raise RuntimeError(
                f"{self.config.collection} is missing dense vectors: {sorted(missing)}"
            )
        wrong_dimensions = {
            name: spec.get("size")
            for name, spec in vectors.items()
            if name in {"image", "caption"}
            and spec.get("size") != self.config.vector_dim
        }
        if wrong_dimensions:
            raise RuntimeError(
                f"{self.config.collection} has incompatible dimensions "
                f"{wrong_dimensions}; expected {self.config.vector_dim}"
            )
        if "caption_bm25" not in sparse:
            raise RuntimeError(
                f"{self.config.collection} has no caption_bm25 sparse vector; "
                "use a new collection or re-index it"
            )
        LOGGER.info(
            "validated qdrant collection",
            extra={
                "collection": self.config.collection,
                "points": info.get("points_count"),
            },
        )

    def _create_collection(self) -> None:
        self.request(
            f"/collections/{self.config.collection}",
            "PUT",
            {
                "vectors": {
                    "image": {
                        "size": self.config.vector_dim,
                        "distance": "Cosine",
                    },
                    "caption": {
                        "size": self.config.vector_dim,
                        "distance": "Cosine",
                    },
                },
                "sparse_vectors": {"caption_bm25": {"modifier": "idf"}},
            },
        )
        for field, schema in (
            ("source", "keyword"),
            ("node_id", "keyword"),
            ("camera_id", "keyword"),
            ("timestamp", "integer"),
        ):
            self.request(
                f"/collections/{self.config.collection}/index",
                "PUT",
                {"field_name": field, "field_schema": schema},
            )
        LOGGER.info(
            "created qdrant collection",
            extra={
                "collection": self.config.collection,
                "vector_dim": self.config.vector_dim,
            },
        )

    def upsert(
        self,
        point_id: str,
        image_vector: list[float],
        caption_vector: list[float],
        caption: str,
        payload: dict,
    ) -> None:
        self.request(
            f"/collections/{self.config.collection}/points?wait=true",
            "PUT",
            {
                "points": [
                    {
                        "id": point_id,
                        "vector": {
                            "image": image_vector,
                            "caption": caption_vector,
                            "caption_bm25": {
                                "text": caption,
                                "model": self.config.bm25_model,
                            },
                        },
                        "payload": payload,
                    }
                ]
            },
            timeout=120,
        )

    def query(
        self,
        using: str,
        query: Any,
        limit: int,
        query_filter: dict | None,
    ) -> list[dict]:
        body: dict[str, Any] = {
            "query": query,
            "using": using,
            "limit": limit,
            "with_payload": True,
        }
        if query_filter:
            body["filter"] = query_filter
        response = self.request(
            f"/collections/{self.config.collection}/points/query",
            "POST",
            body,
        )
        return response["result"]["points"]


class Captioner:
    def __init__(self, common: CommonConfig, config: CaptionConfig):
        self.config = config
        self.http = HttpClient(common.http_retries, common.http_backoff_seconds)

    def caption(self, image_path: Path) -> tuple[str, str]:
        image_b64 = self._encode_image(image_path)
        result = self.http.json(
            f"{self.config.ollama_url}/api/generate",
            "POST",
            {
                "model": self.config.model,
                "prompt": self.config.prompt,
                "images": [image_b64],
                "think": self.config.think,
                "keep_alive": "10m",
                "stream": False,
                "options": {
                    "num_predict": self.config.num_predict,
                    "num_ctx": self.config.num_ctx,
                    "temperature": self.config.temperature,
                },
            },
            timeout=self.config.timeout_seconds,
        )
        done_reason = str(result.get("done_reason") or "")
        text = clean_caption(str(result.get("response") or ""))
        if not text:
            raise RuntimeError(f"Ollama returned an empty caption ({done_reason=})")
        if done_reason == "length":
            text = trim_to_last_sentence(text)
        return text, done_reason

    def _encode_image(self, image_path: Path) -> str:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.thumbnail(
                (self.config.max_image_edge, self.config.max_image_edge),
                Image.Resampling.LANCZOS,
            )
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


class Embedder:
    """One Jina model per process, with serialized inference."""

    def __init__(self, config: CommonConfig):
        self.config = config
        self._model = None
        self._device = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str | None:
        return self._device

    def load(self):
        with self._load_lock:
            if self._model is not None:
                return self._model
            if not self.config.embed_model_dir.is_dir():
                raise FileNotFoundError(
                    f"EMBED_MODEL_DIR does not exist: {self.config.embed_model_dir}"
                )
            import torch
            from transformers import AutoModel

            if self.config.require_gpu and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is unavailable and REQUIRE_GPU=true; refusing silent CPU fallback"
                )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            started = time.time()
            model = AutoModel.from_pretrained(
                self.config.embed_model_dir,
                trust_remote_code=True,
                use_text_flash_attn=False,
                use_vision_xformers=False,
            )
            self._model = model.eval().to(device)
            self._device = device
            LOGGER.info(
                "loaded embedding model",
                extra={
                    "model_dir": str(self.config.embed_model_dir),
                    "device": device,
                    "seconds": round(time.time() - started, 2),
                },
            )
        return self._model

    def encode_text(self, text: str) -> list[float]:
        import torch

        with self._inference_lock, torch.inference_mode():
            values = self.load().encode_text(
                [text], truncate_dim=self.config.vector_dim
            )[0]
        return self._validated_vector(values, "text")

    def encode_image(self, image_path: Path) -> list[float]:
        import torch

        with Image.open(image_path) as source:
            image = source.convert("RGB")
            with self._inference_lock, torch.inference_mode():
                values = self.load().encode_image(
                    [image], truncate_dim=self.config.vector_dim
                )[0]
        return self._validated_vector(values, "image")

    def _validated_vector(self, values: Any, kind: str) -> list[float]:
        if hasattr(values, "detach"):
            values = values.detach().float().cpu().tolist()
        elif hasattr(values, "tolist"):
            values = values.tolist()
        vector = [float(value) for value in values]
        if len(vector) != self.config.vector_dim:
            raise RuntimeError(
                f"{kind} embedding has {len(vector)} dimensions; "
                f"expected {self.config.vector_dim}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise RuntimeError(f"{kind} embedding contains non-finite values")
        return vector


@dataclass(frozen=True)
class Frame:
    point_id: str
    image_id: str
    image_path: Path
    source: str
    node_id: str
    camera_id: str
    timestamp: int


@dataclass(frozen=True)
class IndexedFrame:
    point_id: str
    image_id: str
    caption: str
    caption_words: int
    done_reason: str
    elapsed_seconds: float


class FrameIndexer:
    def __init__(
        self,
        store: QdrantStore,
        captioner: Captioner,
        embedder: Embedder,
    ):
        self.store = store
        self.captioner = captioner
        self.embedder = embedder

    def index(self, frame: Frame) -> IndexedFrame:
        started = time.time()
        caption, done_reason = self.captioner.caption(frame.image_path)
        image_vector = self.embedder.encode_image(frame.image_path)
        caption_vector = self.embedder.encode_text(caption)
        self.store.upsert(
            frame.point_id,
            image_vector,
            caption_vector,
            caption,
            {
                "source": frame.source,
                "node_id": frame.node_id,
                "camera_id": frame.camera_id,
                "image_id": frame.image_id,
                "image_path": str(frame.image_path),
                "caption": caption,
                "timestamp": frame.timestamp,
                "mime_type": "image/jpeg",
                "vector_dim": self.store.config.vector_dim,
            },
        )
        return IndexedFrame(
            point_id=frame.point_id,
            image_id=frame.image_id,
            caption=caption,
            caption_words=len(caption.split()),
            done_reason=done_reason,
            elapsed_seconds=time.time() - started,
        )


class SearchEngine:
    def __init__(
        self, config: SearchConfig, store: QdrantStore, embedder: Embedder
    ):
        self.config = config
        self.store = store
        self.embedder = embedder

    def resolve_weights(
        self, requested: dict[str, float] | None
    ) -> dict[str, float]:
        weights = dict(self.config.default_weights)
        if requested:
            unknown = set(requested) - set(LEGS)
            if unknown:
                raise ValueError(
                    f"unknown legs {sorted(unknown)}; expected {list(LEGS)}"
                )
            weights.update({key: float(value) for key, value in requested.items()})
        if any(not math.isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError("weights must be finite numbers >= 0")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("at least one weight must be > 0")
        return {key: value / total for key, value in weights.items()}

    def search(
        self,
        query: str,
        top_k: int,
        requested_weights: dict[str, float] | None = None,
        since: int | None = None,
        source: str | None = None,
        node_id: str | None = None,
        camera_id: str | None = None,
    ) -> dict:
        started = time.time()
        weights = self.resolve_weights(requested_weights)
        query_filter = build_filter(since, source, node_id, camera_id)
        candidate_limit = max(self.config.candidates_per_leg, top_k)
        dense_vector = None
        if weights["image"] > 0 or weights["caption"] > 0:
            dense_vector = self.embedder.encode_text(query)

        raw_by_leg: dict[str, dict[str, float]] = {}
        payloads: dict[str, dict] = {}
        for leg in LEGS:
            if weights[leg] <= 0:
                continue
            qdrant_query = (
                {"text": query, "model": self.config.common.bm25_model}
                if leg == "bm25"
                else dense_vector
            )
            points = self.store.query(
                "caption_bm25" if leg == "bm25" else leg,
                qdrant_query,
                candidate_limit,
                query_filter,
            )
            raw_by_leg[leg] = {
                str(point["id"]): float(point["score"]) for point in points
            }
            for point in points:
                payloads.setdefault(str(point["id"]), point.get("payload") or {})

        normalized_by_leg = {
            leg: normalize(scores) for leg, scores in raw_by_leg.items()
        }
        fused: dict[str, dict[str, Any]] = {}
        for leg, scores in normalized_by_leg.items():
            for point_id, score in scores.items():
                entry = fused.setdefault(
                    point_id,
                    {"score": 0.0, "scores": {}, "raw_scores": {}},
                )
                entry["scores"][leg] = round(score, 6)
                entry["raw_scores"][leg] = round(raw_by_leg[leg][point_id], 6)
                entry["score"] += weights[leg] * score

        ranked = sorted(
            fused.items(), key=lambda item: (-item[1]["score"], item[0])
        )[:top_k]
        results = []
        for point_id, entry in ranked:
            payload = payloads.get(point_id, {})
            results.append(
                {
                    "id": point_id,
                    "score": round(entry["score"], 6),
                    "scores": entry["scores"],
                    "raw_scores": entry["raw_scores"],
                    "image_id": payload.get("image_id"),
                    "image_path": payload.get("image_path"),
                    "source": payload.get("source"),
                    "node_id": payload.get("node_id"),
                    "camera_id": payload.get("camera_id"),
                    "timestamp": payload.get("timestamp"),
                    "caption": payload.get("caption"),
                }
            )
        return {
            "query": query,
            "weights_used": {
                key: round(value, 6) for key, value in weights.items()
            },
            "legs_queried": sorted(raw_by_leg),
            "candidates_considered": len(fused),
            "returned": len(results),
            "took_ms": round((time.time() - started) * 1000, 1),
            "results": results,
        }


def clean_caption(text: str) -> str:
    text = " ".join(text.split())
    lowered = text.lower()
    for prefix in CAPTION_PREFIXES:
        if lowered.startswith(prefix):
            return text[len(prefix) :].lstrip(" :,-").strip()
    return text


def trim_to_last_sentence(text: str) -> str:
    end = max(
        text.rfind(". "), text.rfind(".\n"), text.rfind("! "), text.rfind("? ")
    )
    if end == -1 and text.endswith((".", "!", "?")):
        return text
    return text[: end + 1].strip() if end > len(text) * 0.6 else text


def normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high - low < 1e-12:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def build_filter(
    since: int | None,
    source: str | None,
    node_id: str | None,
    camera_id: str | None,
) -> dict | None:
    must = []
    if since is not None:
        must.append({"key": "timestamp", "range": {"gte": since}})
    for key, value in (
        ("source", source),
        ("node_id", node_id),
        ("camera_id", camera_id),
    ):
        if value:
            must.append({"key": key, "match": {"value": value}})
    return {"must": must} if must else None
