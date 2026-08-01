# Sage deployment

This directory deploys two separate images:

| Workload | Lifecycle | Purpose |
| --- | --- | --- |
| `image-search-ingest` | daemon Deployment or scheduled one-shot job | capture, caption, embed, and index |
| `image-search-api` | always-running Deployment with one replica | authenticated hybrid search |

Both workloads use the same Qdrant collection and Jina model contract. Only
ingestion needs the camera, Ollama, frame storage, and durable spool.

## Versions and image tags

`apps/<role>/sage.yaml:version` is the single source of truth. Both apps are
released together on one version because they share one Qdrant collection
contract. That version appears in four kinds of place:

| Where | What it means |
| --- | --- |
| `apps/*/sage.yaml` `version:` | the release Sage ECR builds and tags |
| `apps/search/api.py` `version=` | what `/openapi.json` advertises to clients |
| image tags in `deploy/` and `scripts/` | which build a manifest deploys |
| `COLLECTION` (`edge_v3_live`) | the *data* contract — deliberately not the app version |

`COLLECTION` is versioned separately and on purpose: app releases are frequent
and backward compatible, whereas changing the embedding model or vector schema
is a migration that needs a new collection name. Bumping the app version must
never require re-indexing.

Nothing generates these references, so bump `apps/*/sage.yaml` first and then
run the checker, which reports every reference still on the old tag:

```bash
python3 scripts/check-versions.py
python3 scripts/check-app-sync.py
```

## Building for Thor (arm64)

Sage ECR builds `linux/arm64` under QEMU emulation on an amd64 host, and that
emulation aborts when an `nvcr.io/nvidia/pytorch` image runs `import torch`
during a `RUN` step — which these Dockerfiles do deliberately, to pin torch,
torchvision, and NumPy to the base image's NVIDIA-compiled builds. Until Sage
offers a native arm64 builder, build on the node itself and import into k3s:

```bash
sudo docker build -t image-search-ingest:0.3.4 apps/ingest
sudo docker save image-search-ingest:0.3.4 | sudo k3s ctr images import -
```

Do not remove the pinning to make an ECR build pass. Generic PyPI torch has no
Thor (`sm_110`) kernels and silently falls back to CPU, and a transitive NumPy
2.x upgrade breaks torch's NumPy 1.x C ABI at runtime.

## Prerequisites

- Both images have been built and pushed to a registry visible from the node.
- `jina-clip-v2/` and `hf-cache/` exist below the model host directory.
- Qdrant and Ollama are reachable through routable private URLs or Kubernetes
  Service DNS names.
- The collection name is identical in ingestion and search configuration.
- The node administrator can use `sudo pluginctl` and `sudo kubectl`.

`127.0.0.1` inside a pod refers to that pod, not the Thor host. Do not use a
host-loopback Qdrant or Ollama URL unless the dependency is in the same pod.

## Option A: long-running ingestion

Use this for frequent sampling. Jina remains loaded between captures, and
`CAPTURE_INTERVAL_SECONDS=180` captures one frame every three minutes.

Copy `ingest.env.example` outside Git and replace its placeholders. The file
must contain only `KEY=VALUE` lines because `pluginctl --env-from` rejects
comments and blank lines.

```bash
MODEL_ROOT=/home/sajanneupane137/models \
ENV_FILE=/home/sajanneupane137/.config/sage-image-search/ingest-h01e.env \
scripts/deploy-ingest.sh \
10.31.81.1:5000/local/image-search-ingest:0.3.4
```

The script submits the environment at deployment time, then applies
`ingest-production-patch.yaml`. The patch sets `Recreate`, the NVIDIA
RuntimeClass, a termination grace period, and the ingest heartbeat probe.

The heartbeat is written by a dedicated thread every
`HEARTBEAT_INTERVAL_SECONDS` (10s), not by the ingestion loop, and
`healthcheck.py` fails at `HEARTBEAT_MAX_AGE_SECONDS` (30s). This separation is
required: captioning one frame is a blocking Ollama call bounded by
`CAPTION_TIMEOUT` (300s) and retried with backoff, so a work-loop heartbeat
would go stale during entirely healthy work and the liveness probe would
restart the pod mid-caption, on every frame. The probe therefore answers "is
this process alive"; the heartbeat file's `spool` and `runtime` counters
answer "is it making progress". Keep the interval well below the max age if
you tune either.

Inspect it with Kubernetes because `pluginctl logs` expects a scheduler pod
name and may not resolve a persistent Deployment:

```bash
sudo kubectl get deployment,pod -l app=image-search-ingest
sudo kubectl logs -f deployment/image-search-ingest \
  -c image-search-ingest
sudo kubectl exec deployment/image-search-ingest \
  -c image-search-ingest -- \
  python3 /app/healthcheck.py
```

## Option B: scheduled one-shot ingestion

Set `RUN_MODE=oneshot`. Each invocation captures one camera frame, indexes all
pending work that is ready, and exits. The scheduler starts it again according
to the cron science rule.

Runtime values belong in the submitted job's `pluginSpec.env` map:

```yaml
pluginSpec:
  image: registry.example/image-search-ingest:0.3.4
  env:
    RUN_MODE: "oneshot"
    COLLECTION: "edge_v3_live"
    QDRANT_URL: "http://qdrant.example:6333"
    CAMERA: "{secret.image-search-ingest.CAMERA}"
```

The camera value uses a scheduler secret rather than plaintext. Secret
references have the form
`{secret.<lowercase-secret-name>.<alphanumeric-key>}`.

`jobs/ingest-oneshot.example.yaml` is a complete static example. For values
selected at submission time, use the checked-in helper:

```bash
INGEST_IMAGE=registry.example/image-search-ingest:0.3.4 \
NODE_ID=H01E \
QDRANT_URL=http://qdrant.example:6333 \
OLLAMA_URL=http://ollama.example:11434 \
COLLECTION=edge_v3_live \
SCHEDULE='*/3 * * * *' \
scripts/submit-ingest-job.sh
```

The helper renders a temporary job description, shows its non-secret
configuration, and calls:

```bash
sesctl submit --file-path TEMPORARY_JOB_FILE
```

Set `DRY_RUN=true` to print the rendered job without submitting it. Optional
settings can be supplied in `JOB_ENV_FILE` as `KEY=VALUE` lines. Values in that
file override defaults, except `RUN_MODE`, which is always forced to
`oneshot`.

### GPU limitation for remotely scheduled jobs

The current edge-scheduler `pluginSpec` supports environment, volumes, node
selectors, and resource settings, but it does not expose
`runtimeClassName`. H01E needs `runtimeClassName: nvidia` for CUDA. Therefore,
remote scheduled jobs require an administrator to configure automatic/default
NVIDIA RuntimeClass injection. Until then, use the long-running
`pluginctl deploy` path, whose production patch sets the RuntimeClass
explicitly.

## Always-running search

Copy `search-config.yaml` outside Git and set its private Qdrant URL. Create a
secret from `search-secret.example.yaml` outside Git, or reuse an existing
`image-search-secret`.

```bash
MODEL_ROOT=/home/sajanneupane137/models \
DATA_ROOT=/var/lib/sage-image-search \
CONFIG_FILE=/home/sajanneupane137/.config/sage-image-search/search-config-h01e.yaml \
SECRET_FILE=/home/sajanneupane137/.config/sage-image-search/search-secret-h01e.yaml \
scripts/deploy-search.sh \
10.31.81.1:5000/local/image-search-api:0.3.4
```

The Deployment declares `replicas: 1`; Kubernetes restarts the process after a
failure or node restart. The deploy helper also explicitly restores it to one
replica. The ingest data directory is mounted at `/data` read-only, so the API
can verify that each returned `image_path` exists without allowing search to
modify captured frames. `REQUIRE_ACCESSIBLE_IMAGES=true` excludes stale
Qdrant records and reports their count as `skipped_unavailable_images`.

Check and query it:

```bash
sudo kubectl get deployment,pod,service \
  -l app.kubernetes.io/name=image-search-api
sudo kubectl logs -f deployment/image-search-api
sudo kubectl port-forward service/image-search-api 8099:8099
```

In a second terminal:

```bash
SEARCH_API_KEY="$(<~/.config/sage-image-search/search-api-key)" \
scripts/query.sh "wildfire smoke above trees" --top-k 10
```

Keep the Service as `ClusterIP` for same-node access. Cross-node access needs
an administrator-managed authenticated Ingress or a centrally hosted search
service. Do not expose it as an unauthenticated NodePort.
