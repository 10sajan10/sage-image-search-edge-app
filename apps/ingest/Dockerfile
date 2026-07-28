FROM nvcr.io/nvidia/pytorch:26.06-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libjpeg-turbo8 \
        libpng16-16 \
        libtiff6 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    RUN_MODE=daemon \
    CAPTURE_SOURCE=camera \
    CAMERA_ID=camera \
    CAPTURE_INTERVAL_SECONDS=300 \
    QDRANT_URL=http://127.0.0.1:6333 \
    COLLECTION=edge_v4_live \
    OLLAMA_URL=http://127.0.0.1:11434 \
    CAPTION_MODEL=gemma4:e2b \
    EMBED_MODEL_DIR=/model/jina-clip-v2 \
    HF_HOME=/model/hf-cache \
    REQUIRE_GPU=true \
    FRAME_DIR=/data/frames \
    SPOOL_DIR=/data/spool \
    PUBLISH_TO_BEEHIVE=true \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json

COPY image_search /app/image_search
COPY main.py healthcheck.py /app/
COPY tests /app/tests
COPY pytest.ini /app/pytest.ini

RUN python3 -m compileall -q /app/image_search /app/main.py /app/healthcheck.py \
    && python3 -m pytest -q /app/tests \
    && python3 -m pip check

STOPSIGNAL SIGTERM
ENTRYPOINT ["python3", "/app/main.py"]
