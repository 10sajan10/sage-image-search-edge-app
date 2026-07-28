# Ingestion app

The ingestion app captures from a Sage camera or discovers a directory, writes
durable spool state, captions images with Ollama, embeds images and captions
with Jina CLIP v2, and performs idempotent Qdrant upserts.

It supports two runtime modes:

- `RUN_MODE=daemon`: remain running and capture every
  `CAPTURE_INTERVAL_SECONDS`.
- `RUN_MODE=oneshot`: capture once (or perform one directory discovery cycle),
  drain pending work, and exit. Use this mode in a Sage scheduled job.

All runtime configuration is supplied through environment variables. Sage job
files can set them under `pluginSpec.env`; credentials should use
`{secret.<secret-name>.<key>}` references.

This folder owns:

- `sage.yaml` — ECR metadata pointing at this directory as the source.
- `requirements.txt` — the complete ingestion dependency set.
- `.env.example` — runtime configuration template.
- `Dockerfile` — safe defaults and the ingestion entrypoint.
- `image_search/` — all Python runtime modules.
- `tests/` and `pytest.ini` — app-local verification.

Build from this directory:

```bash
cd apps/ingest
docker build -t 10.31.81.1:5000/local/image-search-ingest:0.3.3 .
```

In this monorepo, `sage.yaml` sets `source.directory: apps/ingest`. If this
folder becomes its own repository, change that value to `"."` and update the
source URL.

Run directly with Docker:

```bash
cp .env.example /secure/path/ingest.env
# Edit /secure/path/ingest.env before starting.
docker run -d \
  --name image-search-ingest \
  --restart unless-stopped \
  --device nvidia.com/gpu=all \
  --network host \
  --env-file /secure/path/ingest.env \
  -v /home/sajanneupane137/models:/model:ro \
  -v /var/lib/sage-image-search:/data \
  10.31.81.1:5000/local/image-search-ingest:0.3.3
```

Values passed through `--env-file` override the Dockerfile `ENV` defaults.
Never put the real camera password in the Dockerfile or commit the copied
environment file.

Persistent data layout:

```text
/data/frames/             captured JPEGs (stable paths)
/data/spool/pending/      queued or retrying records
/data/spool/ingested/     successful ingestion receipts
/data/spool/failed/       dead-letter and corrupt records
```
