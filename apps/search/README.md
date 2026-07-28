# Search API app

The search app is an always-running FastAPI Deployment. It loads Jina once,
validates the fixed Qdrant collection, and exposes authenticated hybrid search
over image embeddings, caption embeddings, and BM25.

Default fusion proportions come from `WEIGHT_IMAGE`, `WEIGHT_CAPTION`, and
`WEIGHT_BM25`. Every `/search` request may override them.

With `REQUIRE_ACCESSIBLE_IMAGES=true`, stale Qdrant records whose
`image_path` is absent from the read-only `/data` mount are excluded.

This folder owns:

- `sage.yaml` — ECR metadata pointing at this directory as the source.
- `requirements.txt` — the complete search dependency set.
- `.env.example` — Qdrant, model, authentication, and scoring settings.
- `Dockerfile` — safe defaults and the search API entrypoint.
- `image_search/` — all Python runtime modules.
- `tests/` and `pytest.ini` — app-local verification.

Build from this directory:

```bash
cd apps/search
docker build -t 10.31.81.1:5000/local/image-search-api:0.3.4 .
```

In this monorepo, `sage.yaml` sets `source.directory: apps/search`. If this
folder becomes its own repository, change that value to `"."` and update the
source URL.

Run it as an always-running Docker service:

```bash
cp .env.example /secure/path/search.env
# Edit /secure/path/search.env and replace SEARCH_API_KEY.
docker run -d \
  --name image-search-api \
  --restart unless-stopped \
  --device nvidia.com/gpu=all \
  --network host \
  --env-file /secure/path/search.env \
  -v /home/sajanneupane137/models:/model:ro \
  -v /var/lib/sage-image-search:/data:ro \
  10.31.81.1:5000/local/image-search-api:0.3.4
```

Values passed through `--env-file` override Dockerfile defaults. The
Kubernetes Deployment instead provides the same variables through its
ConfigMap and Secret. It mounts the ingest frame store at `/data` read-only,
so returned `image_path` values resolve inside the search container.

Kubernetes runs this image with one replica behind the private
`image-search-api` ClusterIP Service. It is not a scheduled Sage job.
