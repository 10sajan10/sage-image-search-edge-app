"""Validated environment configuration for both application roles."""

from __future__ import annotations

import os
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def env_bool(name: str, default: bool) -> bool:
    value = env(name, str(default)).lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    try:
        value = int(env(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    try:
        value = float(env(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def normalized_url(name: str, default: str) -> str:
    value = env(name, default).rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name} must be an http(s) URL, got {value!r}")
    return value


def redacted_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.username is None:
        return value
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, f"***:***@{host}", parsed.path, parsed.query, parsed.fragment))


@dataclass(frozen=True)
class CommonConfig:
    qdrant_url: str
    collection: str
    vector_dim: int
    bm25_model: str
    embed_model_dir: Path
    node_id: str
    camera_id: str
    require_gpu: bool
    http_retries: int
    http_backoff_seconds: float

    @classmethod
    def from_env(cls) -> "CommonConfig":
        collection = env("COLLECTION", "edge_v3_live")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}", collection):
            raise ValueError(f"COLLECTION contains unsupported characters: {collection!r}")
        return cls(
            qdrant_url=normalized_url("QDRANT_URL", "http://127.0.0.1:6333"),
            collection=collection,
            vector_dim=env_int("VECTOR_DIM", 1024, minimum=1),
            bm25_model=env("BM25_MODEL", "Qdrant/bm25"),
            embed_model_dir=Path(env("EMBED_MODEL_DIR", "/model/weights/jina-clip-v2")),
            node_id=env("NODE_ID", os.environ.get("WAGGLE_NODE_ID", socket.gethostname())),
            camera_id=env("CAMERA_ID", "camera"),
            require_gpu=env_bool("REQUIRE_GPU", True),
            http_retries=env_int("HTTP_RETRIES", 3, minimum=0),
            http_backoff_seconds=env_float("HTTP_BACKOFF_SECONDS", 1.0, minimum=0.0),
        )

    def public_dict(self) -> dict:
        return {
            "qdrant_url": redacted_url(self.qdrant_url),
            "collection": self.collection,
            "vector_dim": self.vector_dim,
            "bm25_model": self.bm25_model,
            "embed_model_dir": str(self.embed_model_dir),
            "node_id": self.node_id,
            "camera_id": self.camera_id,
            "require_gpu": self.require_gpu,
        }


@dataclass(frozen=True)
class CaptionConfig:
    ollama_url: str
    model: str
    timeout_seconds: int
    think: bool
    temperature: float
    target_words: int
    num_ctx: int
    num_predict: int
    max_image_edge: int
    prompt: str

    @classmethod
    def from_env(cls) -> "CaptionConfig":
        target_words = env_int("CAPTION_TARGET_WORDS", 250, minimum=1)
        tokens_per_word = env_float("TOKENS_PER_WORD", 1.16, minimum=0.1)
        trim_margin = env_float("TRIM_MARGIN", 1.10, minimum=1.0)
        configured_predict = env_int("CAPTION_NUM_PREDICT", 0, minimum=0)
        return cls(
            ollama_url=normalized_url("OLLAMA_URL", "http://127.0.0.1:11434"),
            model=env("CAPTION_MODEL", "gemma4:e2b"),
            timeout_seconds=env_int("CAPTION_TIMEOUT", 300, minimum=1),
            think=env_bool("CAPTION_THINK", False),
            temperature=env_float("CAPTION_TEMPERATURE", 0.0, minimum=0.0),
            target_words=target_words,
            num_ctx=env_int("CAPTION_NUM_CTX", 4096, minimum=256),
            num_predict=configured_predict
            or round(target_words * tokens_per_word * trim_margin),
            max_image_edge=env_int("MAX_IMAGE_EDGE", 1024, minimum=64),
            prompt=env(
                "CAPTION_PROMPT",
                "Describe this image in one factual paragraph of about 550 words. "
                "Mention the main subjects, setting, and key visual details. "
                "Do not begin with phrases like 'This image shows' or 'Here is'. "
                "Return only the caption, no preamble, no markdown.",
            ),
        )

    def public_dict(self) -> dict:
        return {
            "ollama_url": redacted_url(self.ollama_url),
            "caption_model": self.model,
            "caption_timeout_seconds": self.timeout_seconds,
            "caption_think": self.think,
            "caption_temperature": self.temperature,
            "caption_target_words": self.target_words,
            "caption_num_ctx": self.num_ctx,
            "caption_num_predict": self.num_predict,
            "max_image_edge": self.max_image_edge,
        }


@dataclass(frozen=True)
class IngestConfig:
    common: CommonConfig
    caption: CaptionConfig
    capture_source: str
    camera: str
    capture_interval_seconds: int
    image_dir: Path
    image_glob: str
    frame_dir: Path
    spool_dir: Path
    max_discover_per_cycle: int
    max_attempts: int
    retry_base_seconds: float
    retry_max_seconds: float
    max_captures: int
    exit_when_drained: bool
    publish_to_beehive: bool
    heartbeat_path: Path

    @classmethod
    def from_env(cls) -> "IngestConfig":
        common = CommonConfig.from_env()
        source = env("CAPTURE_SOURCE", "camera").lower()
        if source not in {"camera", "directory"}:
            raise ValueError("CAPTURE_SOURCE must be 'camera' or 'directory'")
        camera = env("CAMERA")
        if source == "camera" and not camera:
            raise ValueError("CAMERA is required when CAPTURE_SOURCE=camera")
        spool_dir = Path(env("SPOOL_DIR", "/data/spool"))
        return cls(
            common=common,
            caption=CaptionConfig.from_env(),
            capture_source=source,
            camera=camera,
            capture_interval_seconds=env_int(
                "CAPTURE_INTERVAL_SECONDS", 300, minimum=1
            ),
            image_dir=Path(env("IMAGE_DIR", "/data/images")),
            image_glob=env("IMAGE_GLOB", "**/*.jpg"),
            frame_dir=Path(env("FRAME_DIR", "/data/frames")),
            spool_dir=spool_dir,
            max_discover_per_cycle=env_int(
                "MAX_DISCOVER_PER_CYCLE", 100, minimum=1
            ),
            max_attempts=env_int("MAX_INGEST_ATTEMPTS", 20, minimum=1),
            retry_base_seconds=env_float(
                "INGEST_RETRY_BASE_SECONDS", 5.0, minimum=0.1
            ),
            retry_max_seconds=env_float(
                "INGEST_RETRY_MAX_SECONDS", 300.0, minimum=1.0
            ),
            max_captures=env_int("MAX_CAPTURES", 0, minimum=0),
            exit_when_drained=env_bool("EXIT_WHEN_DRAINED", False),
            publish_to_beehive=env_bool("PUBLISH_TO_BEEHIVE", True),
            heartbeat_path=Path(
                env("HEARTBEAT_PATH", str(spool_dir / "heartbeat.json"))
            ),
        )

    def public_dict(self) -> dict:
        return {
            **self.common.public_dict(),
            **self.caption.public_dict(),
            "capture_source": self.capture_source,
            "camera": self.camera or None,
            "capture_interval_seconds": self.capture_interval_seconds,
            "image_dir": str(self.image_dir),
            "image_glob": self.image_glob,
            "frame_dir": str(self.frame_dir),
            "spool_dir": str(self.spool_dir),
            "max_discover_per_cycle": self.max_discover_per_cycle,
            "max_ingest_attempts": self.max_attempts,
            "retry_base_seconds": self.retry_base_seconds,
            "retry_max_seconds": self.retry_max_seconds,
            "max_captures": self.max_captures,
            "exit_when_drained": self.exit_when_drained,
            "publish_to_beehive": self.publish_to_beehive,
            "heartbeat_path": str(self.heartbeat_path),
        }


@dataclass(frozen=True)
class SearchConfig:
    common: CommonConfig
    host: str
    port: int
    default_weights: dict[str, float]
    default_top_k: int
    max_top_k: int
    candidates_per_leg: int
    max_concurrency: int
    request_timeout_seconds: int
    api_key: str

    @classmethod
    def from_env(cls) -> "SearchConfig":
        weights = {
            "image": env_float("WEIGHT_IMAGE", 0.60, minimum=0.0),
            "caption": env_float("WEIGHT_CAPTION", 0.25, minimum=0.0),
            "bm25": env_float("WEIGHT_BM25", 0.15, minimum=0.0),
        }
        if sum(weights.values()) <= 0:
            raise ValueError("At least one default search weight must be > 0")
        return cls(
            common=CommonConfig.from_env(),
            host=env("SEARCH_HOST", "0.0.0.0"),
            port=env_int("SEARCH_PORT", 8099, minimum=1),
            default_weights=weights,
            default_top_k=env_int("DEFAULT_TOP_K", 25, minimum=1),
            max_top_k=env_int("MAX_TOP_K", 200, minimum=1),
            candidates_per_leg=env_int("CANDIDATES_PER_LEG", 100, minimum=1),
            max_concurrency=env_int("SEARCH_MAX_CONCURRENCY", 1, minimum=1),
            request_timeout_seconds=env_int(
                "SEARCH_REQUEST_TIMEOUT_SECONDS", 60, minimum=1
            ),
            api_key=env("SEARCH_API_KEY"),
        )

    def public_dict(self) -> dict:
        return {
            **self.common.public_dict(),
            "host": self.host,
            "port": self.port,
            "default_weights": self.default_weights,
            "default_top_k": self.default_top_k,
            "max_top_k": self.max_top_k,
            "candidates_per_leg": self.candidates_per_leg,
            "max_concurrency": self.max_concurrency,
            "request_timeout_seconds": self.request_timeout_seconds,
            "authentication_enabled": bool(self.api_key),
        }
