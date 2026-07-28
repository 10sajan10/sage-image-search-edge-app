"""Production search API for the fixed Qdrant image-search collection."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from image_search.config import SearchConfig
from image_search.logging import configure_logging
from image_search.pipeline import ApiError, Embedder, QdrantStore, SearchEngine

LOGGER = logging.getLogger("image_search.search_api")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int | None = Field(default=None, ge=1)
    weights: dict[str, float] | None = Field(
        default=None,
        description="Optional image/caption/bm25 weights; normalized to sum to one.",
    )
    since: int | None = Field(
        default=None, description="Only include timestamps >= this nanosecond value."
    )
    source: str | None = None
    node_id: str | None = None
    camera_id: str | None = None


class AppState:
    config: SearchConfig | None = None
    store: QdrantStore | None = None
    embedder: Embedder | None = None
    engine: SearchEngine | None = None
    executor: ThreadPoolExecutor | None = None
    semaphore: asyncio.Semaphore | None = None
    ready: bool = False
    startup_error: str | None = None


STATE = AppState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    try:
        config = SearchConfig.from_env()
        store = QdrantStore(config.common)
        embedder = Embedder(config.common)
        LOGGER.info(
            "resolved search configuration", extra={"config": config.public_dict()}
        )
        await asyncio.to_thread(store.validate_collection)
        await asyncio.to_thread(embedder.load)
        STATE.config = config
        STATE.store = store
        STATE.embedder = embedder
        STATE.engine = SearchEngine(config, store, embedder)
        STATE.executor = ThreadPoolExecutor(
            max_workers=config.max_concurrency,
            thread_name_prefix="search-inference",
        )
        STATE.semaphore = asyncio.Semaphore(config.max_concurrency)
        STATE.ready = True
        LOGGER.info("search API is ready")
    except Exception as error:
        STATE.startup_error = f"{type(error).__name__}: {error}"
        LOGGER.exception("search API startup failed")
        raise

    yield

    STATE.ready = False
    if STATE.executor is not None:
        STATE.executor.shutdown(wait=True, cancel_futures=True)


app = FastAPI(
    title="Sage Image Search API",
    version="0.3.3",
    description="Weighted image, caption, and BM25 search over a fixed Qdrant collection.",
    lifespan=lifespan,
)


def require_api_key(
    request: Request, x_api_key: str | None = Header(default=None)
) -> None:
    config = STATE.config
    if config is None or not config.api_key:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, config.api_key):
        raise HTTPException(status_code=401, detail="invalid API key")


@app.get("/livez")
async def livez() -> dict:
    return {"ok": True, "role": "search-api"}


@app.get("/readyz")
@app.get("/healthz")
async def readyz() -> dict:
    if not STATE.ready or STATE.store is None or STATE.config is None:
        raise HTTPException(
            status_code=503,
            detail=STATE.startup_error or "search API is not ready",
        )
    try:
        info = await asyncio.to_thread(STATE.store.collection_info)
    except Exception as error:
        raise HTTPException(
            status_code=503, detail=f"Qdrant is unavailable: {error}"
        ) from error
    return {
        "ok": True,
        "role": "search-api",
        "collection": STATE.config.common.collection,
        "points": info.get("points_count"),
        "embedder_loaded": bool(STATE.embedder and STATE.embedder.loaded),
        "device": STATE.embedder.device if STATE.embedder else None,
    }


@app.get("/config", dependencies=[Depends(require_api_key)])
async def config() -> dict:
    if STATE.config is None:
        raise HTTPException(status_code=503, detail="configuration unavailable")
    return STATE.config.public_dict()


@app.get("/stats", dependencies=[Depends(require_api_key)])
async def stats() -> dict:
    if STATE.store is None or STATE.config is None:
        raise HTTPException(status_code=503, detail="search API is not ready")
    try:
        info = await asyncio.to_thread(STATE.store.collection_info)
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "collection": STATE.config.common.collection,
        "points": info.get("points_count"),
        "indexed_vectors": info.get("indexed_vectors_count"),
        "status": info.get("status"),
        "device": STATE.embedder.device if STATE.embedder else None,
        "max_concurrency": STATE.config.max_concurrency,
    }


@app.post("/search", dependencies=[Depends(require_api_key)])
async def search(request: SearchRequest) -> dict:
    if (
        not STATE.ready
        or STATE.engine is None
        or STATE.config is None
        or STATE.executor is None
        or STATE.semaphore is None
    ):
        raise HTTPException(status_code=503, detail="search API is not ready")
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be blank")
    top_k = request.top_k or STATE.config.default_top_k
    if top_k > STATE.config.max_top_k:
        raise HTTPException(
            status_code=400,
            detail=f"top_k exceeds MAX_TOP_K={STATE.config.max_top_k}",
        )

    try:
        STATE.engine.resolve_weights(request.weights)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    loop = asyncio.get_running_loop()
    async with STATE.semaphore:
        future = loop.run_in_executor(
            STATE.executor,
            STATE.engine.search,
            query,
            top_k,
            request.weights,
            request.since,
            request.source,
            request.node_id,
            request.camera_id,
        )
        try:
            return await asyncio.wait_for(
                future, timeout=STATE.config.request_timeout_seconds
            )
        except TimeoutError as error:
            raise HTTPException(status_code=504, detail="search timed out") from error
        except ApiError as error:
            raise HTTPException(status_code=502, detail=f"Qdrant: {error}") from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error


def main() -> None:
    configure_logging()
    config = SearchConfig.from_env()
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
        access_log=True,
        workers=1,
    )


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
