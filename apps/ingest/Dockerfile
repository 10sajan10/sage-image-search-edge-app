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

# Torch and NumPy must stay the NVIDIA-compiled builds from the base image.
# Generic PyPI torch has no Thor (sm_110) kernels, and a transitive NumPy 2.x
# upgrade breaks torch's NumPy 1.x C ABI. transformers/timm/peft will happily
# pull both in, so freeze all three before installing anything.
RUN TORCH_VER="$(python3 -c 'import torch; print(torch.__version__)')" \
    && TV_VER="$(python3 -c 'import torchvision; print(torchvision.__version__)')" \
    && NP_VER="$(python3 -c 'import numpy; print(numpy.__version__)')" \
    && printf 'torch==%s\ntorchvision==%s\nnumpy==%s\n' \
        "$TORCH_VER" "$TV_VER" "$NP_VER" > /tmp/constraints.txt \
    && pip3 install --no-cache-dir -c /tmp/constraints.txt -r /app/requirements.txt

# pywaggle's waggle.data.vision does `import cv2` at module scope but does not
# depend on opencv, so the camera source needs a working cv2. The base image
# ships its own GUI opencv which shadows the pip wheel; installing over it
# leaves stale .so files and fails with "numpy.core.multiarray failed to
# import". Neither --force-reinstall nor pip uninstall alone clears them --
# the dist-packages cv2* directories must be removed explicitly.
RUN pip3 uninstall -y opencv-python opencv-python-headless 2>/dev/null; \
    rm -rf /usr/local/lib/python3.*/dist-packages/cv2* \
    && pip3 install --no-cache-dir -c /tmp/constraints.txt \
        "opencv-python-headless==5.0.0.93"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    RUN_MODE=daemon \
    CAPTURE_SOURCE=camera \
    CAMERA_ID=camera \
    CAPTURE_INTERVAL_SECONDS=180 \
    QDRANT_URL=http://127.0.0.1:6333 \
    COLLECTION=edge_v3_live \
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
    && python3 -m pip check \
    # The unit tests never touch cv2 or torch: main.py imports the camera
    # source lazily, so a broken cv2 or a downgraded torch would otherwise
    # only surface on the node, inside the capture thread, as a
    # CrashLoopBackOff. Import them here so the build fails instead.
    && python3 -c "import cv2, numpy, torch, torchvision, waggle.data.vision; \
print('cv2', cv2.__version__, '| numpy', numpy.__version__, \
'| torch', torch.__version__)"

STOPSIGNAL SIGTERM
ENTRYPOINT ["python3", "/app/main.py"]
