FROM nvcr.io/nvidia/pytorch:26.06-py3

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    QDRANT_URL=http://127.0.0.1:6333 \
    COLLECTION=edge_v4_live \
    EMBED_MODEL_DIR=/model/jina-clip-v2 \
    HF_HOME=/model/hf-cache \
    REQUIRE_GPU=true \
    SEARCH_HOST=0.0.0.0 \
    SEARCH_PORT=8099 \
    WEIGHT_IMAGE=0.60 \
    WEIGHT_CAPTION=0.25 \
    WEIGHT_BM25=0.15 \
    DEFAULT_TOP_K=25 \
    MAX_TOP_K=200 \
    CANDIDATES_PER_LEG=100 \
    SEARCH_MAX_CONCURRENCY=1 \
    SEARCH_REQUEST_TIMEOUT_SECONDS=60 \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json

COPY image_search /app/image_search
COPY api.py cli.py /app/
COPY tests /app/tests
COPY pytest.ini /app/pytest.ini

RUN python3 -m compileall -q /app/image_search /app/api.py /app/cli.py \
    && python3 -m pytest -q /app/tests \
    && python3 -m pip check

STOPSIGNAL SIGTERM
ENTRYPOINT ["python3", "/app/api.py"]
