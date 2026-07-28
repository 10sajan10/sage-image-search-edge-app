"""Core pipeline: config, captioning, embedding, Qdrant access.

Imported by main.py, which owns the capture loop and the HTTP API. Everything
here is synchronous and blocking on purpose -- main.py runs it in a worker
thread so the event loop stays responsive while a frame is being captioned.

Per frame, three retrievable representations are written:

    image  ──▶ jina-clip-v2 ──▶ 1024-dim "image"   vector
    caption ─▶ jina-clip-v2 ──▶ 1024-dim "caption" vector
    caption ─▶ Qdrant       ──▶ sparse "caption_bm25" (server-side BM25+IDF)
"""
from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

# --------------------------------------------------------------- config -----

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)

def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))

def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))

def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --- the fixed database instance ---
QDRANT_URL      = _env("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION      = _env("COLLECTION", "edge_v3_live")
VECTOR_DIM      = _env_int("VECTOR_DIM", 1024)
BM25_MODEL      = _env("BM25_MODEL", "Qdrant/bm25")

# --- captioning ---
OLLAMA_URL      = _env("OLLAMA_URL", "http://127.0.0.1:11434")
CAPTION_MODEL   = _env("CAPTION_MODEL", "gemma4:e2b")
CAPTION_TIMEOUT = _env_int("CAPTION_TIMEOUT", 300)
CAPTION_THINK   = _env_bool("CAPTION_THINK", False)
CAPTION_TEMP    = _env_float("CAPTION_TEMPERATURE", 0.0)   # 0 = reproducible captions
TARGET_WORDS    = _env_int("CAPTION_TARGET_WORDS", 250)
NUM_CTX         = _env_int("CAPTION_NUM_CTX", 4096)
RETRIES         = _env_int("RETRIES", 3)
MAX_IMAGE_EDGE  = _env_int("MAX_IMAGE_EDGE", 1024)

# --- embeddings ---
EMBED_MODEL_DIR = _env("EMBED_MODEL_DIR", "/model/weights/jina-clip-v2")

# --- capture ---
CAPTURE_ENABLED = _env_bool("CAPTURE_ENABLED", True)
CAPTURE_EVERY   = _env_int("CAPTURE_INTERVAL_SECONDS", 300)
CAMERA          = _env("CAMERA", "").strip()     # RTSP url / device / name; empty -> watch IMAGE_DIR
IMAGE_DIR       = Path(_env("IMAGE_DIR", "/data/images"))
IMAGE_GLOB      = _env("IMAGE_GLOB", "**/*.jpg")
FRAME_DIR       = Path(_env("FRAME_DIR", "/data/frames"))
STATE_PATH      = Path(_env("STATE_PATH", "/data/state/ingested.jsonl"))
ERRORS_PATH     = Path(_env("ERRORS_PATH", "/data/state/errors.jsonl"))
MAX_PER_CYCLE   = _env_int("MAX_PER_CYCLE", 25)  # bound work per cycle so one tick can't run away

# --- search defaults (overridable per request) ---
W_IMAGE         = _env_float("WEIGHT_IMAGE", 0.60)
W_CAPTION       = _env_float("WEIGHT_CAPTION", 0.25)
W_BM25          = _env_float("WEIGHT_BM25", 0.15)
DEFAULT_TOP_K   = _env_int("DEFAULT_TOP_K", 25)
MAX_TOP_K       = _env_int("MAX_TOP_K", 200)
CANDIDATES      = _env_int("CANDIDATES_PER_LEG", 100)

# --- server ---
HOST            = _env("SEARCH_HOST", "0.0.0.0")
PORT            = _env_int("SEARCH_PORT", 8099)
PUBLISH         = _env_bool("PUBLISH_TO_BEEHIVE", True)

# The prompt asks for far more words than we want: gemma4:e2b under-delivers on
# length, so the TOKEN CAP below is what actually controls it. Measured on this
# model: asking "250" yields ~153 words, "550" yields ~311 -- and the spread
# widens as you ask for more. A cap is exact where wording is not.
CAPTION_PROMPT = _env("CAPTION_PROMPT", (
    "Describe this image in one factual paragraph of about 550 words. "
    "Mention the main subjects, setting, and key visual details. "
    "Do not begin with phrases like 'This image shows' or 'Here is'. "
    "Return only the caption, no preamble, no markdown."
))
TOKENS_PER_WORD = _env_float("TOKENS_PER_WORD", 1.16)   # measured on gemma4:e2b output
TRIM_MARGIN     = _env_float("TRIM_MARGIN", 1.10)       # headroom for sentence-trim loss
NUM_PREDICT     = _env_int("CAPTION_NUM_PREDICT", 0) or round(TARGET_WORDS * TOKENS_PER_WORD * TRIM_MARGIN)

POINT_NAMESPACE = uuid.UUID("6f9d5c1e-3a7b-4e2f-9c88-0f5a1b2c3d4e")
CAPTION_PREFIXES = ("this image shows", "this image depicts", "the image shows",
                    "the image depicts", "here is a caption", "here is", "caption:")

# Runtime counters, surfaced by /stats.
STATS: dict[str, Any] = {
    "indexed": 0, "failed": 0, "skipped": 0, "cycles": 0,
    "last_cycle_started": None, "last_indexed_at": None, "last_error": None,
}


def describe_config() -> dict:
    return {
        "mode": "camera" if CAMERA else "directory-watch",
        "camera": CAMERA or None,
        "image_dir": str(IMAGE_DIR),
        "capture_enabled": CAPTURE_ENABLED,
        "capture_interval_s": CAPTURE_EVERY,
        "qdrant_url": QDRANT_URL,
        "collection": COLLECTION,
        "vector_dim": VECTOR_DIM,
        "ollama_url": OLLAMA_URL,
        "caption_model": CAPTION_MODEL,
        "caption_target_words": TARGET_WORDS,
        "caption_num_predict": NUM_PREDICT,
        "caption_temperature": CAPTION_TEMP,
        "embed_model_dir": EMBED_MODEL_DIR,
        "default_weights": {"image": W_IMAGE, "caption": W_CAPTION, "bm25": W_BM25},
    }


# ------------------------------------------------------------------ http ----

class ApiError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status


def http_json(url: str, method: str = "GET", payload: Any = None, timeout: int = 60) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        raise ApiError(error.code, error.read().decode("utf-8", "replace")) from error
    return json.loads(body) if body else None


def qdrant(path: str, method: str = "GET", payload: Any = None, timeout: int = 60) -> Any:
    return http_json(QDRANT_URL + path, method, payload, timeout)


# ----------------------------------------------------------------- qdrant ---

def ensure_collection() -> None:
    """Create the collection if absent: 2 dense vectors + 1 sparse BM25/IDF.

    The sparse vector MUST be declared here, at creation: Qdrant cannot add one
    to an existing collection without a full re-index, and the BM25 leg of
    search depends on it.
    """
    try:
        params = qdrant(f"/collections/{COLLECTION}")["result"]["config"]["params"]
        missing = {"image", "caption"} - set(params.get("vectors") or {})
        if missing:
            raise RuntimeError(f"{COLLECTION} exists without dense vectors: {missing}")
        if "caption_bm25" not in (params.get("sparse_vectors") or {}):
            raise RuntimeError(
                f"{COLLECTION} exists without the 'caption_bm25' sparse vector. It cannot "
                "be added in place -- drop the collection or use a different COLLECTION name.")
        print(f"collection {COLLECTION}: OK", flush=True)
        return
    except ApiError as error:
        if error.status != 404:
            raise

    qdrant(f"/collections/{COLLECTION}", "PUT", {
        "vectors": {"image":   {"size": VECTOR_DIM, "distance": "Cosine"},
                    "caption": {"size": VECTOR_DIM, "distance": "Cosine"}},
        "sparse_vectors": {"caption_bm25": {"modifier": "idf"}},
    })
    for field, schema in (("source", "keyword"), ("timestamp", "integer")):
        qdrant(f"/collections/{COLLECTION}/index", "PUT",
               {"field_name": field, "field_schema": schema})
    print(f"created collection {COLLECTION} (2 x {VECTOR_DIM}d dense + sparse bm25/idf)", flush=True)


def upsert(point_id: str, image_vec, caption_vec, caption: str, payload: dict) -> None:
    """Dense vectors are computed here; the sparse one is generated by Qdrant
    from the raw caption text, so no client-side tokenizer exists to drift."""
    qdrant(f"/collections/{COLLECTION}/points", "PUT", {
        "points": [{
            "id": point_id,
            "vector": {"image": image_vec, "caption": caption_vec,
                       "caption_bm25": {"text": caption, "model": BM25_MODEL}},
            "payload": payload,
        }]})


# ---------------------------------------------------------------- caption ---

def encode_image_b64(path: Path) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def clean_caption(text: str) -> str:
    text = " ".join(text.split())
    lowered = text.lower()
    for prefix in CAPTION_PREFIXES:
        if lowered.startswith(prefix):
            return text[len(prefix):].lstrip(" :,-").strip()
    return text


def trim_to_last_sentence(text: str) -> str:
    end = max(text.rfind(". "), text.rfind(".\n"), text.rfind("! "), text.rfind("? "))
    if end == -1 and text.endswith((".", "!", "?")):
        return text
    return text[: end + 1].strip() if end > len(text) * 0.6 else text


def caption_image(image_b64: str) -> tuple[str, str]:
    """Returns (caption, done_reason).

    done_reason == "length" is the EXPECTED path: the token cap bound the
    output, which is how length is controlled. Those are trimmed back to the
    last complete sentence so none end mid-word.
    """
    last_error: Exception | None = None
    budget = NUM_PREDICT
    for attempt in range(1, RETRIES + 1):
        try:
            result = http_json(f"{OLLAMA_URL}/api/generate", "POST", {
                "model": CAPTION_MODEL, "prompt": CAPTION_PROMPT, "images": [image_b64],
                "think": CAPTION_THINK, "keep_alive": "10m", "stream": False,
                "options": {"num_predict": budget, "num_ctx": NUM_CTX,
                            "temperature": CAPTION_TEMP},
            }, CAPTION_TIMEOUT)
            done_reason = str(result.get("done_reason") or "")
            text = clean_caption((result.get("response") or "").strip())
            if not text:
                # With thinking on, an exhausted budget returns nothing; an
                # identical retry would fail identically, so escalate instead.
                raise RuntimeError(f"empty response (done_reason={done_reason!r})")
            return (trim_to_last_sentence(text) if done_reason == "length" else text), done_reason
        except (ApiError, urllib.error.URLError, TimeoutError, RuntimeError, OSError) as error:
            last_error = error
            if "empty response" in str(error):
                budget *= 2
            if attempt < RETRIES:
                time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"captioning failed after {RETRIES} attempts: {last_error}")


# -------------------------------------------------------------- embedding ---

_model = None
_model_lock = threading.Lock()


def load_embedder():
    """Load jina-clip-v2 once. Shared by capture and search -- the whole reason
    the two live in one process.

    use_text_flash_attn=False / use_vision_xformers=False are required on Thor
    (sm_110: triton's ptxas rejects flash-attn's rotary kernel). Do NOT pass
    use_fast -- it is an image-processor kwarg and JinaCLIPModel raises TypeError.
    """
    global _model
    with _model_lock:
        if _model is None:
            import torch
            from transformers import AutoModel
            started = time.time()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _model = AutoModel.from_pretrained(
                EMBED_MODEL_DIR, trust_remote_code=True,
                use_text_flash_attn=False, use_vision_xformers=False).eval().to(device)
            name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
            print(f"embedder loaded on {name} in {time.time() - started:.1f}s", flush=True)
            if device == "cpu":
                print("  WARNING: no GPU visible -- embedding will be very slow", flush=True)
    return _model


def embed_text(text: str) -> list[float]:
    import torch
    with torch.inference_mode():
        return [float(x) for x in load_embedder().encode_text([text], truncate_dim=VECTOR_DIM)[0]]


def embed_image(image: Image.Image) -> list[float]:
    import torch
    with torch.inference_mode():
        return [float(x) for x in load_embedder().encode_image([image], truncate_dim=VECTOR_DIM)[0]]


# ------------------------------------------------------------------ frames --

@dataclass(frozen=True)
class Frame:
    image_id: str
    path: Path
    source: str
    timestamp: int

    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(POINT_NAMESPACE, f"{self.source}/{self.image_id}"))


def load_done() -> set[str]:
    done: set[str] = set()
    if STATE_PATH.exists():
        with STATE_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)["image_id"])
    return done


_camera = None


def collect_frames(done: set[str]) -> list[Frame]:
    """One cycle's worth of new frames.

    camera mode    -> take a snapshot, save it, return that one frame.
    directory mode -> return files not yet indexed (bounded by MAX_PER_CYCLE).
    """
    global _camera
    now = int(time.time() * 1e9)

    if CAMERA:
        from waggle.data.vision import Camera
        FRAME_DIR.mkdir(parents=True, exist_ok=True)
        if _camera is None:
            _camera = Camera(CAMERA).__enter__()
            print(f"camera opened: {CAMERA}", flush=True)
        snapshot = _camera.snapshot()
        timestamp = int(getattr(snapshot, "timestamp", None) or now)
        path = FRAME_DIR / f"{timestamp}.jpg"
        Image.fromarray(snapshot.data).convert("RGB").save(
            path, format="JPEG", quality=90, optimize=True)
        return [Frame(path.name, path, "camera", timestamp)]

    if not IMAGE_DIR.is_dir():
        return []
    frames = []
    for path in sorted(IMAGE_DIR.glob(IMAGE_GLOB)):
        if not path.is_file():
            continue
        relative = path.relative_to(IMAGE_DIR)
        image_id = str(relative)
        if image_id in done:
            continue
        frames.append(Frame(image_id, path,
                            relative.parts[0] if len(relative.parts) > 1 else IMAGE_DIR.name,
                            now))
        if len(frames) >= MAX_PER_CYCLE:
            break
    return frames


def index_frame(frame: Frame) -> int:
    """Caption -> embed -> upsert one frame. Returns caption word count.

    Blocking; main.py calls it in a worker thread.
    """
    caption, done_reason = caption_image(encode_image_b64(frame.path))
    with Image.open(frame.path) as source:
        image_vec = embed_image(source.convert("RGB"))
    caption_vec = embed_text(caption)
    upsert(frame.point_id, image_vec, caption_vec, caption, {
        "source": frame.source, "image_id": frame.image_id,
        "image_path": str(frame.path), "caption": caption, "timestamp": frame.timestamp,
    })
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("a", encoding="utf-8") as checkpoint:
        checkpoint.write(json.dumps({"image_id": frame.image_id,
                                     "caption_words": len(caption.split()),
                                     "done_reason": done_reason}) + "\n")
    return len(caption.split())


def record_error(image_id: str, error: Exception) -> None:
    ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ERRORS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"image_id": image_id, "at": time.time(),
                                 "error": f"{type(error).__name__}: {error}"}) + "\n")


# ------------------------------------------------------------------ search --

LEGS = ("image", "caption", "bm25")


def resolve_weights(requested: dict[str, float] | None) -> dict[str, float]:
    weights = {"image": W_IMAGE, "caption": W_CAPTION, "bm25": W_BM25}
    if requested:
        unknown = set(requested) - set(LEGS)
        if unknown:
            raise ValueError(f"unknown legs {sorted(unknown)}; expected {list(LEGS)}")
        weights.update({k: float(v) for k, v in requested.items()})
    if any(v < 0 for v in weights.values()):
        raise ValueError("weights must be >= 0")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("at least one weight must be > 0")
    return {k: v / total for k, v in weights.items()}


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max one leg into 0..1.

    Legs are on different scales -- cosine sits ~0.4-0.8 while BM25 is unbounded
    -- so raw scores cannot be summed. Doing this per query also absorbs BM25's
    IDF drifting as the corpus grows.
    """
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high - low < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - low) / (high - low) for k, v in scores.items()}


def search(query: str, top_k: int, weights: dict[str, float],
           since: int | None = None, source: str | None = None) -> dict:
    """Weighted 3-leg retrieval, fused in-process.

    Fusion is not delegated to Qdrant: its built-in Fusion.RRF is rank-based and
    takes no weights, so it cannot express an arbitrary image/caption/bm25 split.
    """
    started = time.time()
    must = []
    if since is not None:
        must.append({"key": "timestamp", "range": {"gte": since}})
    if source:
        must.append({"key": "source", "match": {"value": source}})
    query_filter = {"must": must} if must else None
    limit = max(CANDIDATES, top_k)

    vector = embed_text(query) if (weights["image"] > 0 or weights["caption"] > 0) else None

    raw: dict[str, dict[str, float]] = {}
    payloads: dict[str, dict] = {}
    for leg in LEGS:
        if weights[leg] <= 0:
            continue                                     # weight 0 -> skip retrieval entirely
        body: dict[str, Any] = (
            {"query": {"text": query, "model": BM25_MODEL}, "using": "caption_bm25"}
            if leg == "bm25" else {"query": vector, "using": leg})
        body.update({"limit": limit, "with_payload": True})
        if query_filter:
            body["filter"] = query_filter
        points = qdrant(f"/collections/{COLLECTION}/points/query", "POST", body)["result"]["points"]
        raw[leg] = {str(p["id"]): float(p["score"]) for p in points}
        for point in points:
            payloads.setdefault(str(point["id"]), point.get("payload") or {})

    fused: dict[str, dict] = {}
    for leg, scores in raw.items():
        for point_id, score in _normalize(scores).items():
            entry = fused.setdefault(point_id, {"scores": {}, "score": 0.0})
            entry["scores"][leg] = round(score, 6)
            entry["score"] += weights[leg] * score

    ranked = sorted(fused.items(), key=lambda kv: -kv[1]["score"])[:top_k]
    results = []
    for point_id, entry in ranked:
        payload = payloads.get(point_id, {})
        results.append({"id": point_id, "score": round(entry["score"], 6),
                        "scores": entry["scores"],
                        "image_id": payload.get("image_id"),
                        "image_path": payload.get("image_path"),
                        "source": payload.get("source"),
                        "timestamp": payload.get("timestamp"),
                        "caption": payload.get("caption")})
    return {"query": query,
            "weights_used": {k: round(v, 4) for k, v in weights.items()},
            "legs_queried": sorted(raw), "candidates_considered": len(fused),
            "returned": len(results), "took_ms": round((time.time() - started) * 1000, 1),
            "results": results}
