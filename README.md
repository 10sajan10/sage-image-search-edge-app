# Sage image search at the edge

Production-oriented image ingestion and hybrid search for NVIDIA Thor Sage
nodes. The repository contains two independently built applications and one
shared, versioned collection contract.

## Standalone application repositories

Each application also has a dedicated repository with its build context at the
repository root:

- [sage-image-search-ingest](https://github.com/10sajan10/sage-image-search-ingest)
  — scheduled or continuous camera capture, captioning, embedding, and Qdrant
  ingestion.
- [sage-image-search-search](https://github.com/10sajan10/sage-image-search-search)
  — always-running authenticated hybrid-search API and terminal client.

This repository is the integration and deployment hub: it documents the full
architecture, keeps the Kubernetes/Sage deployment resources together, and
coordinates compatible versions of the two applications.

## NDP/NRP workspace notebook

[`notebooks/ndp_workspace/ndp_benchmark_search.ipynb`](notebooks/ndp_workspace/ndp_benchmark_search.ipynb)
is a standalone Jupyter workflow for downloading all five pinned benchmark
datasets, building or restoring MobileCLIP2 image and caption vectors,
populating a local SQLite vector database, generating edge_v1 benchmark
comparisons, and returning 25 visual search results. Its isolated requirements,
environment template, bundled Gemma 3 caption export, bundled comparison
results, and setup guide live in the same directory; no caption model or
database container is required.

```text
RTSP or Sage camera
  -> image-search-ingest
     -> durable frame + pending receipt
     -> Ollama caption
     -> Jina image and caption embeddings
     -> Qdrant image/caption/BM25 vectors
     -> ingested or failed receipt

terminal or future UI
  -> image-search-api:8099
     -> Jina query embedding
     -> weighted Qdrant retrieval
     -> ranked scores, captions, metadata, and image paths
```

The apps communicate only through Qdrant and persistent image storage. A
camera or Ollama failure cannot take search offline, and restarting search
cannot interrupt capture.

## Repository layout

```text
apps/
  ingest/
    sage.yaml
    Dockerfile
    requirements.txt
    .env.example
    .dockerignore
    main.py
    healthcheck.py
    image_search/
    tests/
    pytest.ini
    README.md
  search/
    sage.yaml
    Dockerfile
    requirements.txt
    .env.example
    .dockerignore
    api.py
    cli.py
    image_search/
    tests/
    pytest.ini
    README.md
deploy/sage/
  jobs/
  ingest-production-patch.yaml
  ingest.env.example
  search-config.yaml
  search-deployment.yaml
  search-secret.example.yaml
scripts/
  build-apps.sh
  deploy-ingest.sh
  deploy-search.sh
  submit-ingest-job.sh
  query.sh
  e2e.sh
```

Each directory below `apps/` is a complete Sage build context. It can be
built without any file from the parent directory. The two applications share
only the versioned Qdrant collection contract at runtime. If an app is moved
to a separate repository, update its `source.url` and change
`source.directory` from the monorepo subdirectory to `"."`.

## App 1: scheduled ingestion

The ingestion image has two explicit modes.

### Daemon mode

```text
RUN_MODE=daemon
```

The process remains warm and captures every `CAPTURE_INTERVAL_SECONDS` (180
seconds by default). Use this for frequent sampling because Jina is loaded
only once. The production
Deployment uses `Recreate` so two pods never open the camera concurrently.

### One-shot job mode

```text
RUN_MODE=oneshot
```

For a camera source, the process captures exactly one frame, drains pending
work, and exits. For a directory source, it performs one discovery cycle,
drains the queue, and exits. Use this mode with a Sage `cronjob` science rule.
Do not combine a Sage cron schedule with daemon mode.

Runtime values can be supplied in a Sage job:

```yaml
plugins:
  - name: image-search-ingest-once
    pluginSpec:
      image: registry.sagecontinuum.org/USERNAME/sage-image-search-ingest:0.3.4
      selector:
        resource.gpu: "true"
      env:
        RUN_MODE: "oneshot"
        CAPTURE_SOURCE: "camera"
        CAMERA: "{secret.image-search-ingest.CAMERA}"
        CAMERA_ID: "rtsp-main"
        QDRANT_URL: "http://QDRANT_PRIVATE_HOST:6333"
        COLLECTION: "edge_v3_live"
        OLLAMA_URL: "http://OLLAMA_PRIVATE_HOST:11434"
```

The scheduler accepts `pluginSpec.env` as a map. Secret references use
`{secret.<secret-name>.<alphanumeric-key>}` and are resolved on the target
node. The complete example is
[ingest-oneshot.example.yaml](deploy/sage/jobs/ingest-oneshot.example.yaml).

`sesctl` does not have individual `--env` override flags. To choose values at
the time of submission, use the repository helper:

```bash
INGEST_IMAGE=registry.example/image-search-ingest:0.3.4 \
QDRANT_URL=http://qdrant.example:6333 \
OLLAMA_URL=http://ollama.example:11434 \
COLLECTION=edge_v3_live \
SCHEDULE='*/3 * * * *' \
scripts/submit-ingest-job.sh
```

It renders `pluginSpec.env` into a mode-600 temporary document and immediately
submits that document with `sesctl`. Use `DRY_RUN=true` to inspect it first.

Important for this Thor node: the installed scheduler schema accepts `env`,
volumes, selectors, and resources, but does not expose `runtimeClassName`.
H01E currently needs the `nvidia` RuntimeClass to make CUDA visible. An
administrator must configure GPU RuntimeClass injection/defaulting before
remote one-shot Sage jobs can run with `REQUIRE_GPU=true`. The persistent
`pluginctl deploy` flow applies that RuntimeClass explicitly and works now.

## Durable ingestion state

All state lives on a hostPath or persistent volume mounted at `/data`:

```text
/data/
  frames/<node>/<camera>/<year>/<month>/<day>/<timestamp>.jpg
  spool/
    pending/      queued and retrying JSON records
    ingested/     successful ingestion receipts
    failed/       dead-letter and corrupt records
    heartbeat.json
```

Images never move after indexing, so the `image_path` stored in Qdrant remains
stable. Point IDs are deterministic; replaying a record performs an idempotent
upsert. Releases which used `spool/completed` are migrated automatically to
`spool/ingested`.

## App 2: always-running search

Search is a normal Kubernetes Deployment with `replicas: 1`, startup,
readiness, and liveness probes, and a private ClusterIP Service. It is not a
scheduled Sage job.

The default fusion weights are environment variables:

```text
WEIGHT_IMAGE=0.60
WEIGHT_CAPTION=0.25
WEIGHT_BM25=0.15
```

Every request may override them. Non-negative values are normalized to sum to
one, and a zero-weight retrieval leg is skipped.

```http
POST /search
X-API-Key: configured-secret
Content-Type: application/json

{
  "query": "white plush toy on a metal shelf",
  "top_k": 10,
  "weights": {
    "image": 0.60,
    "caption": 0.25,
    "bm25": 0.15
  },
  "node_id": "H01E"
}
```

The response includes fused and per-leg scores, Qdrant raw scores, caption,
timestamp, node, camera, image ID, and the persistent image path. Search mounts
the ingest frame store read-only and, by default, skips records whose image
file is no longer available.

Endpoints:

- `POST /search` — authenticated search
- `GET /stats` — authenticated collection status
- `GET /config` — authenticated redacted configuration
- `GET /livez` — process liveness
- `GET /readyz` and `/healthz` — model and Qdrant readiness

## Build both images

From the repository root:

```bash
PUSH=true scripts/build-apps.sh
```

Defaults:

```text
10.31.81.1:5000/local/image-search-ingest:0.3.4
10.31.81.1:5000/local/image-search-api:0.3.4
```

Override tags when publishing elsewhere:

```bash
INGEST_IMAGE=registry.example/image-search-ingest:0.3.4 \
SEARCH_IMAGE=registry.example/image-search-api:0.3.4 \
PUSH=true \
scripts/build-apps.sh
```

Each image has a separate Dockerfile, dependency set, entrypoint, and
`sage.yaml`. `build-apps.sh` invokes each application directory as its own
Docker build context.

Build either app independently:

```bash
(cd apps/ingest && docker build -t image-search-ingest:0.3.4 .)
(cd apps/search && docker build -t image-search-api:0.3.4 .)
```

## Docker environment values

Each Dockerfile declares safe operational defaults with `ENV`. Environment
files beside the applications document every role-specific setting:

- `apps/ingest/.env.example`
- `apps/search/.env.example`

Copy an example outside Git, replace its placeholders, and pass it at runtime:

```bash
docker run --env-file /secure/path/ingest.env IMAGE
docker run --env-file /secure/path/search.env IMAGE
```

Runtime values override the Dockerfile defaults. Camera credentials and
`SEARCH_API_KEY` are deliberately absent from Docker image layers. Kubernetes
uses the same variables through `pluginSpec.env`, ConfigMaps, and Secrets.

## Deploy on H01E

The node-specific secret-bearing files are outside Git under
`~/.config/sage-image-search`.

### Persistent ingestion

```bash
MODEL_ROOT=/home/sajanneupane137/models \
ENV_FILE=/home/sajanneupane137/.config/sage-image-search/ingest-h01e.env \
scripts/deploy-ingest.sh \
10.31.81.1:5000/local/image-search-ingest:0.3.4
```

### Always-running search

```bash
MODEL_ROOT=/home/sajanneupane137/models \
DATA_ROOT=/var/lib/sage-image-search \
CONFIG_FILE=/home/sajanneupane137/.config/sage-image-search/search-config-h01e.yaml \
scripts/deploy-search.sh \
10.31.81.1:5000/local/image-search-api:0.3.4
```

`deploy-search.sh` requires an existing `image-search-secret`, or a
`SECRET_FILE` containing it. The Deployment is always restored to one replica.

### Query from the same node

Terminal 1:

```bash
sudo kubectl port-forward service/image-search-api 8099:8099
```

Terminal 2:

```bash
SEARCH_API_KEY="$(<~/.config/sage-image-search/search-api-key)" \
scripts/query.sh "white plush toy" --top-k 5
```

Custom weights:

```bash
SEARCH_API_KEY="$(<~/.config/sage-image-search/search-api-key)" \
scripts/query.sh "server equipment" \
  --image-weight 0.50 \
  --caption-weight 0.30 \
  --bm25-weight 0.20
```

## Qdrant collection contract

Ingestion creates the collection if absent; search only validates it.

- `image`: 1024-dimensional dense cosine vector
- `caption`: 1024-dimensional dense cosine vector
- `caption_bm25`: sparse BM25 vector with Qdrant IDF

Use a new collection name for an incompatible schema or embedding-model
migration. Do not delete or recreate a live collection during deployment.

## Verification

Run local unit tests:

```bash
(cd apps/ingest && python3 -m pytest -q)
(cd apps/search && python3 -m pytest -q)
```

Both Docker builds run the same tests during image creation. Full acceptance:

```bash
scripts/e2e.sh \
  10.31.81.1:5000/local/image-search-ingest:0.3.4 \
  10.31.81.1:5000/local/image-search-api:0.3.4
```

The acceptance test uses a uniquely named temporary Qdrant collection and
never modifies `edge_v3_live`.
