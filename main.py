"""Sage edge app — always-running image capture + search, in one process.

    ┌─ background task ─────────────────┐   ┌─ HTTP API ──────────────┐
    │ every CAPTURE_INTERVAL_SECONDS:   │   │ POST /search            │
    │   capture -> caption -> embed     │   │ GET  /healthz /stats    │
    │   -> upsert into Qdrant           │   │ GET  /config            │
    └───────────────────┬───────────────┘   └────────────┬────────────┘
                        └────────── shared ──────────────┘
                              ONE jina-clip-v2 instance
                              ONE Qdrant collection

Why one process rather than two containers:

  * The embedder is loaded ONCE (~2 GB, ~30 s) instead of twice.
  * COLLECTION / QDRANT_URL / EMBED_MODEL_DIR cannot disagree between the
    writer and the reader -- a whole class of "search returns nothing and
    nothing errors" bugs becomes structurally impossible.
  * One container to deploy and supervise.

Why a loop rather than a scheduled job: at a 300 s interval a cron-style job
would reload the model every run (~10% of the budget) and lose the warm cache.
A loop keeps it resident. Job scheduling only wins when the interval is far
larger than the model load time.

Capture never blocks the API: every blocking step runs in a worker thread via
asyncio.to_thread, so queries stay responsive while a frame is being captioned.

Set CAPTURE_ENABLED=false to run a search-only instance against the same
database (e.g. a second replica that only serves queries).
"""
from __future__ import annotations

import asyncio
import contextlib
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import pipeline as P

_capture_task: asyncio.Task | None = None
_plugin = None          # pywaggle Plugin, optional telemetry


# ------------------------------------------------------------ capture loop --

async def capture_cycle(done: set[str]) -> None:
    """One tick: collect new frames and index them."""
    P.STATS["last_cycle_started"] = time.time()
    frames = await asyncio.to_thread(P.collect_frames, done)
    if not frames:
        return
    for frame in frames:
        try:
            words = await asyncio.to_thread(P.index_frame, frame)
        except Exception as error:                # one bad frame must not end the run
            P.STATS["failed"] += 1
            P.STATS["last_error"] = f"{type(error).__name__}: {error}"
            await asyncio.to_thread(P.record_error, frame.image_id, error)
            print(f"  FAILED {frame.image_id}: {error}", flush=True)
            continue
        done.add(frame.image_id)
        P.STATS["indexed"] += 1
        P.STATS["last_indexed_at"] = time.time()
        print(f"indexed {frame.image_id} ({words} words) "
              f"| total={P.STATS['indexed']} failed={P.STATS['failed']}", flush=True)
        if _plugin is not None:
            with contextlib.suppress(Exception):   # telemetry is never fatal
                _plugin.publish("imagesearch.indexed", P.STATS["indexed"],
                                timestamp=frame.timestamp)


async def capture_loop() -> None:
    """Fixed-RATE scheduling: sleep to the next interval boundary so processing
    time does not make the schedule drift. If a cycle overruns the interval we
    log and continue immediately rather than queueing -- an unbounded backlog
    would turn a transient stall into permanent lag.
    """
    done = await asyncio.to_thread(P.load_done)
    print(f"capture loop: every {P.CAPTURE_EVERY}s, {len(done)} already indexed", flush=True)
    while True:
        started = time.monotonic()
        try:
            await capture_cycle(done)
        except asyncio.CancelledError:
            raise
        except Exception as error:                # the loop must survive anything
            P.STATS["last_error"] = f"cycle: {type(error).__name__}: {error}"
            print(f"  CYCLE FAILED: {error}", flush=True)
        P.STATS["cycles"] += 1
        elapsed = time.monotonic() - started
        if elapsed > P.CAPTURE_EVERY:
            P.STATS["skipped"] += 1
            print(f"  WARNING: cycle took {elapsed:.0f}s > interval {P.CAPTURE_EVERY}s", flush=True)
        else:
            await asyncio.sleep(P.CAPTURE_EVERY - elapsed)


# ----------------------------------------------------------------- startup --

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global _capture_task, _plugin
    for key, value in P.describe_config().items():
        print(f"  {key:22s} {value}", flush=True)

    await asyncio.to_thread(P.ensure_collection)
    # Load before serving so the first query is not a ~10 s stall.
    await asyncio.to_thread(P.load_embedder)

    if P.PUBLISH:
        try:
            from waggle.plugin import Plugin
            _plugin = Plugin()
            _plugin.__enter__()
        except Exception as error:               # airgapped / no beehive is fine
            print(f"  (pywaggle unavailable: {error})", flush=True)

    if P.CAPTURE_ENABLED:
        _capture_task = asyncio.create_task(capture_loop())
    else:
        print("capture DISABLED (CAPTURE_ENABLED=false) -- serving search only", flush=True)

    yield

    if _capture_task:
        _capture_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _capture_task
    if _plugin is not None:
        with contextlib.suppress(Exception):
            _plugin.__exit__(None, None, None)


app = FastAPI(title="sage-image-search-edge-app",
              description="Always-running capture + weighted 3-leg image search",
              lifespan=lifespan)


# ------------------------------------------------------------------ routes --

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=P.DEFAULT_TOP_K, ge=1)
    weights: dict[str, float] | None = Field(
        default=None,
        description="Per-leg weights: image / caption / bm25. Normalized to sum 1. "
                    "A leg weighted 0 is skipped entirely.")
    since: int | None = Field(default=None, description="Only frames with timestamp >= this (ns)")
    source: str | None = None


@app.post("/search")
async def search(request: SearchRequest) -> dict:
    if request.top_k > P.MAX_TOP_K:
        raise HTTPException(400, f"top_k exceeds MAX_TOP_K={P.MAX_TOP_K}")
    try:
        weights = P.resolve_weights(request.weights)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    try:
        # Off the event loop: embedding + several Qdrant round-trips.
        return await asyncio.to_thread(P.search, request.query, request.top_k,
                                       weights, request.since, request.source)
    except P.ApiError as error:
        raise HTTPException(502, f"qdrant: {error}") from error


@app.get("/healthz")
async def healthz() -> dict:
    try:
        info = await asyncio.to_thread(
            lambda: P.qdrant(f"/collections/{P.COLLECTION}")["result"])
    except Exception as error:
        raise HTTPException(503, f"qdrant unreachable: {error}") from error
    return {"ok": True, "collection": P.COLLECTION, "points": info["points_count"],
            "capture_enabled": P.CAPTURE_ENABLED,
            "capture_running": bool(_capture_task and not _capture_task.done())}


@app.get("/stats")
async def stats() -> dict:
    info = await asyncio.to_thread(
        lambda: P.qdrant(f"/collections/{P.COLLECTION}")["result"])
    now = time.time()
    return {
        "collection": P.COLLECTION,
        "points": info["points_count"],
        "indexed_vectors": info.get("indexed_vectors_count"),
        **{k: P.STATS[k] for k in ("indexed", "failed", "skipped", "cycles", "last_error")},
        "seconds_since_last_index": (round(now - P.STATS["last_indexed_at"], 1)
                                     if P.STATS["last_indexed_at"] else None),
        "capture": {"enabled": P.CAPTURE_ENABLED, "interval_s": P.CAPTURE_EVERY,
                    "running": bool(_capture_task and not _capture_task.done()),
                    "mode": "camera" if P.CAMERA else "directory-watch"},
    }


@app.get("/config")
async def config() -> dict:
    return P.describe_config()


if __name__ == "__main__":
    print(f"serving on http://{P.HOST}:{P.PORT}", flush=True)
    uvicorn.run(app, host=P.HOST, port=P.PORT, log_level="info")
