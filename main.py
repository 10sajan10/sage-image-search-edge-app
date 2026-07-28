"""Sage edge app — populate the image-search database.

Pipeline (mirrors edge_v3):

    image ──▶ Ollama gemma4:e2b ──▶ caption (~250 words, thinking off)
          ├─▶ jina-clip-v2 ───────▶ image vector   (1024-dim)
          └─▶ jina-clip-v2 ───────▶ caption vector (1024-dim)
              caption text ───────▶ Qdrant BM25 sparse vector (server-side)

Three retrievable representations per image, so search can weight
image-similarity, caption-similarity and keyword matching independently.

Everything is configured by environment variable — nothing is hardcoded to this
node. Run with SHOW_CONFIG=1 to print the resolved configuration and exit.

Assumes (all injectable):
  * Ollama already serving a vision model      -> OLLAMA_URL, CAPTION_MODEL
  * a Qdrant instance already running          -> QDRANT_URL, COLLECTION
  * embedding weights already on disk          -> EMBED_MODEL_DIR

Resumable: every indexed image is appended to a checkpoint file and skipped on
re-run, so the job can be killed and restarted without losing or duplicating
work.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
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

def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# Captioning
OLLAMA_URL      = _env("OLLAMA_URL", "http://127.0.0.1:11434")
CAPTION_MODEL   = _env("CAPTION_MODEL", "gemma4:e2b")
CAPTION_TIMEOUT = _env_int("CAPTION_TIMEOUT", 300)
CAPTION_THINK   = _env_bool("CAPTION_THINK", False)
# Ollama's default for this model is temperature 1.0 -> non-reproducible
# captions. 0 makes the pipeline deterministic; captions are input data.
CAPTION_TEMP    = float(_env("CAPTION_TEMPERATURE", "0"))
TARGET_WORDS    = _env_int("CAPTION_TARGET_WORDS", 250)
NUM_CTX         = _env_int("CAPTION_NUM_CTX", 4096)

# Vector database
QDRANT_URL      = _env("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION      = _env("COLLECTION", "edge_v3_live")
VECTOR_DIM      = _env_int("VECTOR_DIM", 1024)
BM25_MODEL      = _env("BM25_MODEL", "Qdrant/bm25")   # server-side, no weights to download

# Embeddings
EMBED_MODEL_DIR = _env("EMBED_MODEL_DIR", "/model/weights/jina-clip-v2")

# Input source.
#
# CAMERA selects live capture. It is passed straight to pywaggle's Camera(),
# which accepts an RTSP/HTTP stream URL, a /dev/video* path, a device index, a
# name resolved from the node's data-config.json, or a still image file:
#     CAMERA=rtsp://user:pass@10.0.0.5:554/stream1
#     CAMERA=bottom_camera
#     CAMERA=0
# Leave CAMERA empty to instead batch-index an existing directory of images.
CAMERA          = _env("CAMERA", "").strip()
CAPTURE_EVERY   = _env_int("CAPTURE_INTERVAL_SECONDS", 300)
FRAME_DIR       = Path(_env("FRAME_DIR", "/data/frames"))   # where captured frames are kept
MAX_FRAMES      = _env_int("MAX_FRAMES", 0)                 # 0 = run forever

IMAGE_DIR       = Path(_env("IMAGE_DIR", "/data/images"))
IMAGE_GLOB      = _env("IMAGE_GLOB", "**/*.jpg")
STATE_PATH      = Path(_env("STATE_PATH", "/data/state/ingested.jsonl"))
ERRORS_PATH     = Path(_env("ERRORS_PATH", "/data/state/errors.jsonl"))
MAX_IMAGES      = _env_int("MAX_IMAGES", 0)           # 0 = no limit
MAX_IMAGE_EDGE  = _env_int("MAX_IMAGE_EDGE", 1024)
RETRIES         = _env_int("RETRIES", 3)
PUBLISH         = _env_bool("PUBLISH_TO_BEEHIVE", True)

# Reused from edge_v2 with only the word count changed, so caption style stays
# comparable. The number is deliberately ABOVE the target: gemma4:e2b
# under-delivers on length, so the token cap below is what actually controls it.
CAPTION_PROMPT = _env("CAPTION_PROMPT", (
    "Describe this image in one factual paragraph of about 550 words. "
    "Mention the main subjects, setting, and key visual details. "
    "Do not begin with phrases like 'This image shows' or 'Here is'. "
    "Return only the caption, no preamble, no markdown."
))

# Measured on gemma4:e2b output (169/147, 315/273, 359/306 tokens/words), plus
# ~10% headroom for what sentence-trimming discards.
TOKENS_PER_WORD = float(_env("TOKENS_PER_WORD", "1.16"))
TRIM_MARGIN     = float(_env("TRIM_MARGIN", "1.10"))
NUM_PREDICT     = _env_int("CAPTION_NUM_PREDICT", 0) or round(TARGET_WORDS * TOKENS_PER_WORD * TRIM_MARGIN)

# Stable namespace so re-running upserts the same point instead of duplicating.
POINT_NAMESPACE = uuid.UUID("6f9d5c1e-3a7b-4e2f-9c88-0f5a1b2c3d4e")

CAPTION_PREFIXES = ("this image shows", "this image depicts", "the image shows",
                    "the image depicts", "here is a caption", "here is", "caption:")


def show_config() -> None:
    print("configuration:", flush=True)
    for key, value in [
        ("OLLAMA_URL", OLLAMA_URL), ("CAPTION_MODEL", CAPTION_MODEL),
        ("CAPTION_THINK", CAPTION_THINK), ("CAPTION_TEMPERATURE", CAPTION_TEMP),
        ("CAPTION_TARGET_WORDS", f"{TARGET_WORDS} -> num_predict={NUM_PREDICT} tokens"),
        ("QDRANT_URL", QDRANT_URL), ("COLLECTION", COLLECTION),
        ("VECTOR_DIM", VECTOR_DIM), ("BM25_MODEL", BM25_MODEL),
        ("EMBED_MODEL_DIR", EMBED_MODEL_DIR),
        ("mode", "LIVE CAPTURE" if CAMERA else "BATCH (directory)"),
        ("CAMERA", CAMERA or "(unset -> batch mode)"),
        ("CAPTURE_INTERVAL_S", CAPTURE_EVERY if CAMERA else "n/a"),
        ("FRAME_DIR", FRAME_DIR if CAMERA else "n/a"),
        ("IMAGE_DIR", "n/a" if CAMERA else f"{IMAGE_DIR}  (glob {IMAGE_GLOB})"),
        ("STATE_PATH", STATE_PATH),
    ]:
        print(f"  {key:22s} {value}", flush=True)


# ------------------------------------------------------------------ http ----

class ApiError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status


def http_json(url: str, method: str = "GET", payload: Any = None, timeout: int = 60) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        raise ApiError(error.code, error.read().decode("utf-8", "replace")) from error
    return json.loads(body) if body else None


# ----------------------------------------------------------------- qdrant ---

def ensure_collection() -> None:
    """Create the collection if absent.

    Two dense vectors plus a sparse BM25 vector with the IDF modifier. The
    sparse vector MUST exist from the start -- Qdrant cannot add one to an
    existing collection without a full re-index, and the search API's BM25
    weight depends on it.
    """
    try:
        existing = http_json(f"{QDRANT_URL}/collections/{COLLECTION}")
        params = existing["result"]["config"]["params"]
        missing = {"image", "caption"} - set(params.get("vectors") or {})
        if missing:
            raise RuntimeError(f"{COLLECTION} exists without required vectors: {missing}")
        if "caption_bm25" not in (params.get("sparse_vectors") or {}):
            raise RuntimeError(
                f"{COLLECTION} exists without the 'caption_bm25' sparse vector. "
                "It cannot be added in place -- recreate the collection or use a new name."
            )
        print(f"collection {COLLECTION} already exists", flush=True)
        return
    except ApiError as error:
        if error.status != 404:
            raise

    http_json(f"{QDRANT_URL}/collections/{COLLECTION}", "PUT", {
        "vectors": {
            "image":   {"size": VECTOR_DIM, "distance": "Cosine"},
            "caption": {"size": VECTOR_DIM, "distance": "Cosine"},
        },
        "sparse_vectors": {"caption_bm25": {"modifier": "idf"}},
    })
    for field, schema in (("source", "keyword"), ("timestamp", "integer")):
        http_json(f"{QDRANT_URL}/collections/{COLLECTION}/index", "PUT",
                  {"field_name": field, "field_schema": schema})
    print(f"created collection {COLLECTION} "
          f"(2 x {VECTOR_DIM}-dim dense + sparse BM25/idf)", flush=True)


def upsert(point_id: str, image_vec: list[float], caption_vec: list[float],
           caption: str, payload: dict) -> None:
    """Store both dense vectors plus the caption TEXT for BM25.

    Qdrant tokenizes, stems, removes stopwords and applies IDF itself, so no
    client-side tokenizer is needed -- and none should be introduced, or ingest
    and query could disagree.
    """
    http_json(f"{QDRANT_URL}/collections/{COLLECTION}/points", "PUT", {
        "points": [{
            "id": point_id,
            "vector": {
                "image": image_vec,
                "caption": caption_vec,
                "caption_bm25": {"text": caption, "model": BM25_MODEL},
            },
            "payload": payload,
        }]
    })


# ---------------------------------------------------------------- caption ---

def encode_image_b64(path: Path) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def clean_caption(caption: str) -> str:
    text = " ".join(caption.split())
    lowered = text.lower()
    for prefix in CAPTION_PREFIXES:
        if lowered.startswith(prefix):
            return text[len(prefix):].lstrip(" :,-").strip()
    return text


def trim_to_last_sentence(text: str) -> str:
    """Cut a cap-truncated caption back to its last complete sentence."""
    end = max(text.rfind(". "), text.rfind(".\n"), text.rfind("! "), text.rfind("? "))
    if end == -1 and text.endswith((".", "!", "?")):
        return text
    return text[: end + 1].strip() if end > len(text) * 0.6 else text


def caption_image(image_b64: str) -> tuple[str, str]:
    """Returns (caption, done_reason).

    done_reason == "length" is the expected path: the token cap bound the
    output, which is how caption length is controlled. Such captions are
    trimmed to their last complete sentence so none end mid-word.
    """
    last_error: Exception | None = None
    budget = NUM_PREDICT
    for attempt in range(1, RETRIES + 1):
        payload = {
            "model": CAPTION_MODEL,
            "prompt": CAPTION_PROMPT,
            "images": [image_b64],
            "think": CAPTION_THINK,
            "options": {"num_predict": budget, "num_ctx": NUM_CTX, "temperature": CAPTION_TEMP},
            "keep_alive": "10m",
            "stream": False,
        }
        try:
            result = http_json(f"{OLLAMA_URL}/api/generate", "POST", payload, CAPTION_TIMEOUT)
            done_reason = str(result.get("done_reason") or "")
            text = clean_caption((result.get("response") or "").strip())
            if not text:
                # With thinking enabled an exhausted budget yields an empty
                # response; retrying identically would fail identically, so the
                # budget is escalated below rather than repeated.
                raise RuntimeError(f"empty response (done_reason={done_reason!r})")
            if done_reason == "length":
                text = trim_to_last_sentence(text)
            return text, done_reason
        except (ApiError, urllib.error.URLError, TimeoutError, RuntimeError, OSError) as error:
            last_error = error
            if "empty response" in str(error):
                budget *= 2
            if attempt < RETRIES:
                time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"captioning failed after {RETRIES} attempts: {last_error}")


# -------------------------------------------------------------- embedding ---

def load_embedder():
    """Load jina-clip-v2: 1024-dim vectors for BOTH images and text.

    Two kwargs are load-bearing on Jetson Thor, not stylistic:
      use_text_flash_attn=False -- Thor reports sm_110a and the bundled triton's
        ptxas rejects it while building flash-attn's rotary kernel.
      use_vision_xformers=False -- xformers is not installed.
    Do NOT add use_fast here; it is an image-processor kwarg and JinaCLIPModel
    raises TypeError on it.
    """
    import torch
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        EMBED_MODEL_DIR,
        trust_remote_code=True,
        use_text_flash_attn=False,
        use_vision_xformers=False,
    ).eval().to(device)
    name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"embedder loaded from {EMBED_MODEL_DIR} on {name} "
          f"(dtype={next(model.parameters()).dtype})", flush=True)
    if device == "cpu":
        print("  WARNING: no GPU visible -- embedding will be very slow", flush=True)
    return model


def embed_text(model, text: str) -> list[float]:
    import torch
    with torch.inference_mode():
        return [float(x) for x in model.encode_text([text], truncate_dim=VECTOR_DIM)[0]]


def embed_image(model, image: Image.Image) -> list[float]:
    import torch
    with torch.inference_mode():
        return [float(x) for x in model.encode_image([image], truncate_dim=VECTOR_DIM)[0]]


# ------------------------------------------------------------------ input ---

@dataclass(frozen=True)
class Frame:
    image_id: str
    path: Path
    source: str
    timestamp: int | None = None

    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(POINT_NAMESPACE, f"{self.source}/{self.image_id}"))


def discover_frames() -> list[Frame]:
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"IMAGE_DIR does not exist: {IMAGE_DIR}")
    frames = []
    for path in sorted(IMAGE_DIR.glob(IMAGE_GLOB)):
        if path.is_file():
            relative = path.relative_to(IMAGE_DIR)
            frames.append(Frame(
                image_id=str(relative),
                path=path,
                source=relative.parts[0] if len(relative.parts) > 1 else IMAGE_DIR.name,
            ))
    return frames


def load_done() -> set[str]:
    done: set[str] = set()
    if not STATE_PATH.exists():
        return done
    with STATE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                done.add(json.loads(line)["image_id"])
    return done


def capture_frames():
    """Yield frames from a live source (RTSP/device/name) forever.

    Fixed-RATE, not fixed-delay: each iteration sleeps until the next interval
    boundary, so processing time doesn't cause the schedule to drift. If a cycle
    overruns the interval the next capture is taken immediately and a warning is
    logged rather than queueing -- an unbounded backlog would turn a transient
    stall into permanent lag.

    Capture is deliberately cheap and isolated from captioning: the frame is
    written to disk first, so a downstream failure (Ollama down, GPU busy)
    cannot lose the image.
    """
    from waggle.data.vision import Camera

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    captured = 0
    print(f"opening camera: {CAMERA}", flush=True)

    with Camera(CAMERA) as camera:
        while True:
            cycle_started = time.monotonic()
            try:
                snapshot = camera.snapshot()
                timestamp = int(getattr(snapshot, "timestamp", None) or time.time() * 1e9)
                path = FRAME_DIR / f"{timestamp}.jpg"
                Image.fromarray(snapshot.data).convert("RGB").save(
                    path, format="JPEG", quality=90, optimize=True)
                yield Frame(image_id=path.name, path=path, source="camera",
                            timestamp=timestamp)
                captured += 1
            except Exception as error:        # a camera hiccup must not end the run
                print(f"  CAPTURE FAILED: {type(error).__name__}: {error}", flush=True)

            if MAX_FRAMES and captured >= MAX_FRAMES:
                print(f"reached MAX_FRAMES={MAX_FRAMES}, stopping", flush=True)
                return

            elapsed = time.monotonic() - cycle_started
            if elapsed > CAPTURE_EVERY:
                print(f"  WARNING: cycle took {elapsed:.0f}s > interval {CAPTURE_EVERY}s "
                      f"-- capturing immediately, schedule is saturated", flush=True)
            else:
                time.sleep(CAPTURE_EVERY - elapsed)


# ------------------------------------------------------------------- main ---

def main() -> int:
    show_config()
    if _env_bool("SHOW_CONFIG", False):
        return 0

    ensure_collection()

    if CAMERA:
        # Live mode: an endless generator, so there is no "pending" count and
        # the loop below simply never runs out.
        pending = capture_frames()
        total = None
        print(f"live capture every {CAPTURE_EVERY}s from {CAMERA}", flush=True)
    else:
        frames = discover_frames()
        done = load_done()
        pending_list = [frame for frame in frames if frame.image_id not in done]
        if MAX_IMAGES:
            pending_list = pending_list[:MAX_IMAGES]
        total = len(pending_list)
        pending = iter(pending_list)
        print(f"found={len(frames)} already_done={len(done)} pending={total}", flush=True)
        if not total:
            print("nothing to do", flush=True)
            return 0

    model = load_embedder()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # pywaggle is optional: without a beehive it queues to PYWAGGLE_LOG_DIR, and
    # the job must work fully airgapped regardless.
    plugin = None
    if PUBLISH:
        try:
            from waggle.plugin import Plugin
            plugin = Plugin()
            plugin.__enter__()
        except Exception as error:            # noqa: BLE001 - telemetry is never fatal
            print(f"  (pywaggle unavailable, continuing without publish: {error})", flush=True)

    indexed = failed = 0
    started = time.time()
    with STATE_PATH.open("a", encoding="utf-8") as checkpoint, \
         ERRORS_PATH.open("a", encoding="utf-8") as errors:
        for index, frame in enumerate(pending, start=1):
            try:
                caption, done_reason = caption_image(encode_image_b64(frame.path))
                with Image.open(frame.path) as source:
                    image_vec = embed_image(model, source.convert("RGB"))
                caption_vec = embed_text(model, caption)
                timestamp = frame.timestamp or int(time.time() * 1e9)
                upsert(frame.point_id, image_vec, caption_vec, caption, {
                    "source": frame.source,
                    "image_id": frame.image_id,
                    "image_path": str(frame.path),
                    "caption": caption,
                    "timestamp": timestamp,
                })
            except Exception as error:        # one bad image must not end the run
                failed += 1
                errors.write(json.dumps({"image_id": frame.image_id,
                                         "error": f"{type(error).__name__}: {error}"}) + "\n")
                errors.flush()
                print(f"  FAILED {frame.image_id}: {error}", flush=True)
                continue

            checkpoint.write(json.dumps({
                "image_id": frame.image_id,
                "caption_words": len(caption.split()),
                "done_reason": done_reason,
            }) + "\n")
            checkpoint.flush()
            indexed += 1

            if plugin is not None:
                try:
                    plugin.publish("imagesearch.indexed", indexed, timestamp=timestamp)
                    plugin.publish("imagesearch.caption.words", len(caption.split()),
                                   timestamp=timestamp)
                except Exception:             # noqa: BLE001
                    pass

            if total is None:
                # Live mode: no total to count down to, so report each frame.
                print(f"indexed {frame.image_id} ({len(caption.split())} words) "
                      f"| total={indexed} failed={failed}", flush=True)
            elif index % 20 == 0 or index == total:
                rate = index / max(time.time() - started, 1e-6)
                print(f"progress {index}/{total} indexed={indexed} failed={failed} "
                      f"({rate * 3600:.0f}/h, ~{(total - index) / rate / 3600:.1f}h left)",
                      flush=True)

    if plugin is not None:
        try:
            plugin.__exit__(None, None, None)
        except Exception:                     # noqa: BLE001
            pass

    print(f"done. indexed={indexed} failed={failed}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
