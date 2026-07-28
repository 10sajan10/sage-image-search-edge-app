# Base: NVIDIA's PyTorch container, NOT waggle/sage-thor-base:0.1.0.
#
# Both ship a Jetson torch compiled with sm_110 kernels (Thor's compute
# capability), but sage-thor-base pairs its torch 2.8 with numpy 2.5.1, and that
# torch rejects every numpy 2.x -- `import torch` warns "Failed to initialize
# NumPy" and any tensor->numpy conversion raises "RuntimeError: Numpy is not
# available", which is exactly what jina-clip-v2's encode_text/encode_image do.
# Measured on this node: on sage-thor-base, numpy 2.5.1/2.3.4/2.2.6/2.1.0 all
# FAIL and only <2 works; this base ships numpy 2.1.0 and works as-is.
#
# This is also the base claude.md specifies for the project.
FROM nvcr.io/nvidia/pytorch:26.06-py3

# System libraries OpenCV and ffmpeg need at runtime. nvcr ships no OpenCV at
# all, so none of these are present. Package list taken from
# waggle/sage-thor-base's own Dockerfile so camera behaviour matches Waggle's.
# ffmpeg is required by pywaggle for video/RTSP camera streams.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libjpeg-turbo8 \
        libpng16-16 \
        libtiff6 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Airgap: never reach for the HF hub at runtime. Embedding weights are mounted
# in at EMBED_MODEL_DIR (default /model/weights/jina-clip-v2), and jina's
# trust_remote_code modules must already be cached under HF_HOME.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY . .

# Fail the build on syntax errors even when the target role is not the default.
RUN python3 -m compileall -q \
        app_config.py \
        healthcheck.py \
        logging_setup.py \
        main.py \
        pipeline.py \
        search_api.py \
        search_cli.py \
        sources.py \
        spool.py

# One image, two supervised roles:
#   docker run IMAGE                  -> persistent ingestion
#   docker run IMAGE search_api.py    -> search API
# Kubernetes may override command/args explicitly for either role.
STOPSIGNAL SIGTERM
ENTRYPOINT ["python3"]
CMD ["main.py"]
