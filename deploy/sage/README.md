# Sage deployment

This directory deploys two separate images:

| Workload | Lifecycle | Purpose |
| --- | --- | --- |
| `image-search-ingest` | daemon Deployment or scheduled one-shot job | capture, caption, embed, and index |
| `image-search-api` | always-running Deployment with one replica | authenticated hybrid search |

Both workloads use the same Qdrant collection and Jina model contract. Only
ingestion needs the camera, Ollama, frame storage, and durable spool.

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
