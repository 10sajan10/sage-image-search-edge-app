"""Sage edge app — search API over the image-search database.

Runs as a separate process from main.py. The two share nothing but Qdrant, so
search stays available while population runs, and either can be restarted
without touching the other.

Three retrieval legs, each independently weightable per request:

    query text ─┬─ jina-clip-v2 ─▶ 1024-dim vector ─┬─▶ vs image vectors    (leg 1)
                │                                   └─▶ vs caption vectors  (leg 2)
                └─ raw text ────────────────────────────▶ Qdrant BM25       (leg 3)

Leg 3 needs no embedding: Qdrant tokenizes, stems, removes stopwords and applies
IDF server-side. Legs 1 and 2 share a SINGLE text embedding.

Fusion is computed here rather than in Qdrant, deliberately. Qdrant's built-in
Fusion.RRF is rank-based and accepts no weights, so it cannot express a
60/25/15 split. Retrieving each leg separately and blending here also keeps the
scoring identical to the offline benchmark notebooks, so live results and
reported metrics come from the same arithmetic.

    POST /search {"query": "...", "top_k": 25,
                  "weights": {"image": 0.6, "caption": 0.25, "bm25": 0.15}}
    GET  /healthz
    GET  /stats
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --------------------------------------------------------------- config -----

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)

def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))

def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


QDRANT_URL      = _env("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION      = _env("COLLECTION", "edge_v3_live")
VECTOR_DIM      = _env_int("VECTOR_DIM", 1024)
BM25_MODEL      = _env("BM25_MODEL", "Qdrant/bm25")
EMBED_MODEL_DIR = _env("EMBED_MODEL_DIR", "/model/weights/jina-clip-v2")

# Default weights, overridable per request.
W_IMAGE   = _env_float("WEIGHT_IMAGE", 0.60)
W_CAPTION = _env_float("WEIGHT_CAPTION", 0.25)
W_BM25    = _env_float("WEIGHT_BM25", 0.15)

DEFAULT_TOP_K = _env_int("DEFAULT_TOP_K", 25)
MAX_TOP_K     = _env_int("MAX_TOP_K", 200)
# How many candidates to pull per leg before fusing. Must exceed top_k or the
# blend only sees each leg's head and loses items ranked highly by one leg only.
CANDIDATES    = _env_int("CANDIDATES_PER_LEG", 100)

HOST = _env("SEARCH_HOST", "127.0.0.1")   # localhost by default: this serves camera imagery
# 8099, not 8080: on a Waggle node 8080 is commonly taken (here by Weaviate),
# and with --network host uvicorn then dies with "address already in use".
PORT = _env_int("SEARCH_PORT", 8099)

LEGS = ("image", "caption", "bm25")

app = FastAPI(title="image-search-app", description="Weighted 3-leg image search")
_model = None          # lazily loaded jina-clip-v2


# ------------------------------------------------------------------ http ----

def qdrant(path: str, method: str = "GET", payload: Any = None, timeout: int = 60) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        QDRANT_URL + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        raise HTTPException(status_code=502,
                            detail=f"qdrant {error.code}: {error.read().decode()[:300]}") from error
    except urllib.error.URLError as error:
        raise HTTPException(status_code=503, detail=f"qdrant unreachable: {error}") from error
    return json.loads(body) if body else None


# ------------------------------------------------------------- embeddings ---

def get_model():
    """Load jina-clip-v2 once, on first use.

    use_text_flash_attn=False / use_vision_xformers=False are required on Thor
    (sm_110 -- triton's ptxas rejects flash-attn's kernel). Do NOT pass
    use_fast: it is an image-processor kwarg and JinaCLIPModel raises TypeError.
    """
    global _model
    if _model is None:
        import torch
        from transformers import AutoModel
        started = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = AutoModel.from_pretrained(
            EMBED_MODEL_DIR, trust_remote_code=True,
            use_text_flash_attn=False, use_vision_xformers=False,
        ).eval().to(device)
        print(f"embedder loaded on {device} in {time.time() - started:.1f}s", flush=True)
    return _model


def embed_query(text: str) -> list[float]:
    import torch
    model = get_model()
    with torch.inference_mode():
        return [float(x) for x in model.encode_text([text], truncate_dim=VECTOR_DIM)[0]]


# ------------------------------------------------------------------ model ---

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language query")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1)
    weights: dict[str, float] | None = Field(
        default=None,
        description="Per-leg weights: image / caption / bm25. Normalized to sum 1. "
                    "A leg weighted 0 is skipped entirely.",
    )
    since: int | None = Field(default=None, description="Only frames with timestamp >= this (ns)")
    source: str | None = Field(default=None, description="Restrict to one source/dataset")


def resolve_weights(requested: dict[str, float] | None) -> dict[str, float]:
    weights = {"image": W_IMAGE, "caption": W_CAPTION, "bm25": W_BM25}
    if requested:
        unknown = set(requested) - set(LEGS)
        if unknown:
            raise HTTPException(400, f"unknown legs {sorted(unknown)}; expected {list(LEGS)}")
        weights.update({k: float(v) for k, v in requested.items()})
    if any(v < 0 for v in weights.values()):
        raise HTTPException(400, "weights must be >= 0")
    total = sum(weights.values())
    if total <= 0:
        raise HTTPException(400, "at least one weight must be > 0")
    return {k: v / total for k, v in weights.items()}      # normalize to sum 1


def build_filter(request: SearchRequest) -> dict | None:
    must = []
    if request.since is not None:
        must.append({"key": "timestamp", "range": {"gte": request.since}})
    if request.source:
        must.append({"key": "source", "match": {"value": request.source}})
    return {"must": must} if must else None


def normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max a single leg's scores into 0..1.

    Legs are on wildly different scales -- cosine similarity sits around 0.4-0.8
    while BM25 is unbounded -- so raw scores cannot be summed. Normalizing per
    leg per query also makes the result robust to BM25's IDF drifting as the
    corpus grows.
    """
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high - low < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - low) / (high - low) for k, v in scores.items()}


# ------------------------------------------------------------------ routes --

@app.post("/search")
def search(request: SearchRequest) -> dict:
    started = time.time()
    if request.top_k > MAX_TOP_K:
        raise HTTPException(400, f"top_k exceeds MAX_TOP_K={MAX_TOP_K}")
    weights = resolve_weights(request.weights)
    query_filter = build_filter(request)
    limit = max(CANDIDATES, request.top_k)

    # One text embedding serves both dense legs; skip it entirely if neither is used.
    vector = None
    if weights["image"] > 0 or weights["caption"] > 0:
        vector = embed_query(request.query)

    raw: dict[str, dict[str, float]] = {}
    payloads: dict[str, dict] = {}

    for leg in LEGS:
        if weights[leg] <= 0:
            continue                                    # weight 0 -> no retrieval at all
        if leg == "bm25":
            body: dict[str, Any] = {"query": {"text": request.query, "model": BM25_MODEL},
                                    "using": "caption_bm25"}
        else:
            body = {"query": vector, "using": leg}
        body.update({"limit": limit, "with_payload": True})
        if query_filter:
            body["filter"] = query_filter
        points = qdrant(f"/collections/{COLLECTION}/points/query", "POST", body)["result"]["points"]
        raw[leg] = {str(p["id"]): float(p["score"]) for p in points}
        for point in points:
            payloads.setdefault(str(point["id"]), point.get("payload") or {})

    fused: dict[str, dict] = {}
    for leg, scores in raw.items():
        for point_id, score in normalize(scores).items():
            entry = fused.setdefault(point_id, {"scores": {}, "score": 0.0})
            entry["scores"][leg] = round(score, 6)
            entry["score"] += weights[leg] * score

    ranked = sorted(fused.items(), key=lambda kv: -kv[1]["score"])[: request.top_k]
    results = []
    for point_id, entry in ranked:
        payload = payloads.get(point_id, {})
        results.append({
            "id": point_id,
            "score": round(entry["score"], 6),
            "scores": entry["scores"],              # per-leg, so ranking is explainable
            "image_id": payload.get("image_id"),
            "image_path": payload.get("image_path"),
            "source": payload.get("source"),
            "timestamp": payload.get("timestamp"),
            "caption": payload.get("caption"),
        })

    return {
        "query": request.query,
        "weights_used": {k: round(v, 4) for k, v in weights.items()},
        "legs_queried": sorted(raw),
        "candidates_considered": len(fused),
        "returned": len(results),
        "took_ms": round((time.time() - started) * 1000, 1),
        "results": results,
    }


@app.get("/healthz")
def healthz() -> dict:
    try:
        info = qdrant(f"/collections/{COLLECTION}")["result"]
    except HTTPException as error:
        return {"ok": False, "detail": error.detail}
    return {"ok": True, "collection": COLLECTION, "points": info["points_count"],
            "status": info["status"], "embedder_loaded": _model is not None}


@app.get("/stats")
def stats() -> dict:
    info = qdrant(f"/collections/{COLLECTION}")["result"]
    params = info["config"]["params"]
    newest = qdrant(f"/collections/{COLLECTION}/points/scroll", "POST",
                    {"limit": 1, "with_payload": True,
                     "order_by": {"key": "timestamp", "direction": "desc"}})
    points = newest["result"]["points"] if newest else []
    return {
        "collection": COLLECTION,
        "points": info["points_count"],
        "indexed_vectors": info.get("indexed_vectors_count"),
        "dense_vectors": {k: v["size"] for k, v in (params.get("vectors") or {}).items()},
        "sparse_vectors": list((params.get("sparse_vectors") or {})),
        "default_weights": {"image": W_IMAGE, "caption": W_CAPTION, "bm25": W_BM25},
        "newest_frame": (points[0].get("payload") or {}).get("image_id") if points else None,
    }


if __name__ == "__main__":
    import uvicorn
    print(f"serving on http://{HOST}:{PORT}  (collection={COLLECTION})", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
