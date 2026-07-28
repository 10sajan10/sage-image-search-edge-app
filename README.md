# Sage Image Search at the Edge

An always-running Sage camera ingestion pipeline and a separately supervised
weighted search service. Camera frames are captioned through an Ollama-hosted
vision-language model, embedded with Jina CLIP v2, and written into a fixed
Qdrant collection with three independently weighted retrieval representations.

## Architecture

```text
Sage camera node
  image-search-ingest Deployment
    scheduled camera producer
       -> durable filesystem spool
       -> Ollama caption
       -> Jina image + caption embeddings
       -> Qdrant upsert

Reachable Kubernetes environment
  image-search-api Deployment
    terminal/UI query
       -> Jina text embedding (once)
       -> Qdrant image-vector query
       -> Qdrant caption-vector query
       -> Qdrant BM25 query
       -> normalize + weighted fusion
       -> ranked captions, scores, metadata, and image reference
```

The roles use the same immutable image but run as different processes:

```text
python3 main.py        # persistent capture and ingestion
python3 search_api.py  # HTTP search service
python3 search_cli.py  # terminal client for the HTTP service
```

They are separate so camera, Ollama, or ingest restarts do not interrupt search.
Within each role Jina is loaded once and remains warm.

## Why a loop instead of a CronJob?

The ingestion role is a Kubernetes Deployment with an internal fixed-rate
capture scheduler. A CronJob would reload the multi-gigabyte Jina model on every
capture, repeatedly open the camera, and risk overlapping jobs.

Capture and processing are decoupled. The camera producer saves the JPEG and an
atomic spool record at the requested interval; the processing worker consumes
the spool independently. Temporary Ollama, Qdrant, or GPU failures therefore do
not lose the frame. Pending items survive pod restarts and retry with bounded
exponential backoff.

## Qdrant schema

The ingestion role creates the collection if absent and otherwise validates it
strictly:

- `image`: 1024-dimensional dense cosine vector
- `caption`: 1024-dimensional dense cosine vector
- `caption_bm25`: sparse BM25 vector with Qdrant's IDF modifier

Each payload includes:

```json
{
  "source": "camera",
  "node_id": "H01E",
  "camera_id": "bottom-camera",
  "image_id": "H01E/bottom-camera/2026/07/28/....jpg",
  "image_path": "/data/frames/H01E/bottom-camera/2026/07/28/....jpg",
  "caption": "...",
  "timestamp": 1785220000000000000,
  "mime_type": "image/jpeg",
  "vector_dim": 1024
}
```

Point IDs are deterministic over node, camera, source, and image identity, so
replaying a spool item performs an idempotent upsert rather than creating a
duplicate.

`image_path` is an edge-local reference. Cross-node image delivery requires a
shared object store and an `image_uri`; it is intentionally not exposed by this
version because the current access requirement is terminal search.

## Search contract

```http
POST /search
X-API-Key: configured-secret
Content-Type: application/json

{
  "query": "wildfire smoke above trees",
  "top_k": 10,
  "weights": {
    "image": 0.60,
    "caption": 0.25,
    "bm25": 0.15
  },
  "node_id": "H01E"
}
```

Weights are non-negative and normalized to sum to one. A zero-weight leg is not
queried. Results include the fused score, normalized per-leg scores, original
Qdrant scores, caption, timestamps, node/camera identity, and image reference.

Health endpoints:

- `GET /livez`: process liveness
- `GET /readyz` or `/healthz`: model and Qdrant readiness
- `GET /stats`: collection statistics
- `GET /config`: redacted effective configuration

`/search`, `/stats`, and `/config` require `X-API-Key` when `SEARCH_API_KEY` is
configured. `/livez` and `/readyz` remain available to Kubernetes probes.

## Local container usage

The image targets Linux ARM64 and NVIDIA Thor:

```bash
docker build -t localhost/sage-image-search-edge-app:0.2.0 .
```

Run ingestion against a directory for a bounded smoke test:

```bash
docker run --rm \
  --device nvidia.com/gpu=all \
  --network host \
  -v /path/to/models:/model:ro \
  -v /path/to/images:/data/images:ro \
  -v /path/to/state:/data \
  -e HF_HOME=/model/hf-cache \
  -e EMBED_MODEL_DIR=/model/jina-clip-v2 \
  -e CAPTURE_SOURCE=directory \
  -e IMAGE_DIR=/data/images \
  -e EXIT_WHEN_DRAINED=true \
  -e PUBLISH_TO_BEEHIVE=false \
  -e QDRANT_URL=http://127.0.0.1:6333 \
  -e COLLECTION=image_search_smoke \
  -e OLLAMA_URL=http://127.0.0.1:11434 \
  localhost/sage-image-search-edge-app:0.2.0
```

To exercise the actual PyWaggle `Camera` path without node hardware, use a
supported `file://` source and set `MAX_CAPTURES=1`:

```bash
-e CAPTURE_SOURCE=camera \
-e CAMERA=file:///data/input/example.jpg \
-e MAX_CAPTURES=1 \
-e EXIT_WHEN_DRAINED=true
```

Run the search API:

```bash
docker run --rm \
  --device nvidia.com/gpu=all \
  --network host \
  -v /path/to/models:/model:ro \
  -e HF_HOME=/model/hf-cache \
  -e EMBED_MODEL_DIR=/model/jina-clip-v2 \
  -e QDRANT_URL=http://127.0.0.1:6333 \
  -e COLLECTION=image_search_smoke \
  -e SEARCH_API_KEY=development-only-key \
  localhost/sage-image-search-edge-app:0.2.0 search_api.py
```

Query it from a terminal:

```bash
SEARCH_API_KEY=development-only-key \
python3 search_cli.py "an elephant" --top-k 5
```

## Configuration

Shared:

| Variable | Default | Purpose |
|---|---:|---|
| `QDRANT_URL` | `http://127.0.0.1:6333` | Fixed private Qdrant endpoint |
| `COLLECTION` | `edge_v3_live` | Fixed collection |
| `EMBED_MODEL_DIR` | `/model/weights/jina-clip-v2` | Local Jina model |
| `VECTOR_DIM` | `1024` | Required dense-vector size |
| `NODE_ID` | hostname | Payload and point identity |
| `CAMERA_ID` | `camera` | Payload and point identity |
| `REQUIRE_GPU` | `true` | Refuse silent CPU fallback |

Ingestion:

| Variable | Default | Purpose |
|---|---:|---|
| `CAPTURE_SOURCE` | `camera` | `camera` or `directory` |
| `CAMERA` | required | Sage name, stream URL, device, or camera index |
| `CAPTURE_INTERVAL_SECONDS` | `300` | Fixed-rate capture interval |
| `FRAME_DIR` | `/data/frames` | Durable captured JPEGs |
| `SPOOL_DIR` | `/data/spool` | Pending/completed/failed records |
| `MAX_INGEST_ATTEMPTS` | `20` | Attempts before dead letter |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Private caption service |
| `CAPTION_MODEL` | `gemma4:e2b` | Ollama model |
| `PUBLISH_TO_BEEHIVE` | `true` | Publish operational telemetry |

Search:

| Variable | Default | Purpose |
|---|---:|---|
| `SEARCH_HOST` | `0.0.0.0` | Listen address inside the pod |
| `SEARCH_PORT` | `8099` | Service target port |
| `SEARCH_API_KEY` | empty | Enables API-key enforcement when set |
| `SEARCH_MAX_CONCURRENCY` | `1` | Serialized GPU request workers |
| `WEIGHT_IMAGE` | `0.60` | Default image-vector weight |
| `WEIGHT_CAPTION` | `0.25` | Default caption-vector weight |
| `WEIGHT_BM25` | `0.15` | Default BM25 weight |

See [deploy/sage/README.md](deploy/sage/README.md) for the Sage plugin,
Kubernetes Service, and terminal port-forward workflow.

## Verification

Run deterministic unit tests and dependency validation inside the final image:

```bash
docker run --rm --entrypoint python3 IMAGE -m pytest -q /app/tests
docker run --rm --entrypoint python3 IMAGE -m pip check
```

Production acceptance additionally requires:

1. Real Sage camera capture using injected `data-config.json`
2. Retry across an intentional dependency outage and pod restart
3. A known-image query from a second terminal through the ClusterIP port-forward
4. Authentication rejection with a missing or invalid API key
5. Confirmation that a rollout never starts two camera pods simultaneously

The repeatable container acceptance test uses a uniquely named temporary
collection and never deletes `edge_v3_live`:

```bash
scripts/e2e.sh localhost/sage-image-search-edge-app:production-test
```
