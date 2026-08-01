FROM nvcr.io/nvidia/pytorch:26.06-py3

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Torch, torchvision, and NumPy must stay the NVIDIA-compiled builds from the
# base image. Record their local NVIDIA versions before pip runs and assert the
# exact values after installation. Using those local versions as PyPI
# constraints does not work: pip cannot resolve NVIDIA's non-PyPI build as a
# candidate for timm's torch dependency.
RUN python3 -c "import json, numpy, torch, torchvision; \
json.dump({'torch': torch.__version__, 'torchvision': torchvision.__version__, \
'numpy': numpy.__version__}, open('/tmp/nvidia-base-versions.json', 'w'))" \
    && pip3 install --no-cache-dir -r /app/requirements.txt

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    QDRANT_URL=http://127.0.0.1:6333 \
    COLLECTION=edge_v3_live \
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
    REQUIRE_ACCESSIBLE_IMAGES=true \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json

COPY image_search /app/image_search
COPY api.py cli.py /app/
COPY tests /app/tests
COPY pytest.ini /app/pytest.ini

RUN python3 -m compileall -q /app/image_search /app/api.py /app/cli.py \
    && python3 -m pytest -q /app/tests \
    && python3 -m pip check \
    # The unit tests stub the embedder, so a downgraded torch would otherwise
    # only surface on the node as a CPU fallback or a missing-kernel error.
    && python3 -c "import json, numpy, torch, torchvision; \
expected=json.load(open('/tmp/nvidia-base-versions.json')); \
actual={'torch': torch.__version__, 'torchvision': torchvision.__version__, \
'numpy': numpy.__version__}; assert actual == expected, \
f'NVIDIA base packages changed: expected {expected}, got {actual}'; print(actual)"

STOPSIGNAL SIGTERM
ENTRYPOINT ["python3", "/app/api.py"]
