# Sage deployment

The production topology uses one immutable image in two independently
supervised roles:

1. `image-search-ingest`: a Sage plugin Deployment. `pluginctl` is used because
   it injects `/run/waggle/data-config.json`, camera configuration, and the
   PyWaggle transport.
2. `image-search-api`: an ordinary Kubernetes Deployment and ClusterIP Service.
   It does not need camera access or Ollama.

Qdrant and Ollama stay private. Only the search Service is a future ingress
candidate.

## Prerequisites

- The application image is built by Sage ECR and its exact registry tag is
  known.
- `/opt/sage/image-search-models/jina-clip-v2` contains the model weights.
- `/opt/sage/image-search-models/hf-cache` contains Jina's trusted-code cache.
- Qdrant and Ollama are reachable from the plugin pod using private URLs.
- An administrator can run `sudo pluginctl` and `sudo kubectl`.

The generated WES plugin Deployment does **not** use host networking. Therefore
`127.0.0.1` inside the plugin is the plugin pod, not the Thor host. Use routable
private endpoints or Kubernetes Service DNS names. If WES network policy blocks
an external private endpoint, ask the administrator to allow it; use
`pluginctl --develop` only for a temporary connectivity test, not as the
production security policy.

## Deploy ingestion

Copy `ingest.env.example` outside the repository and replace every placeholder.
Never commit camera credentials or service tokens.

```bash
sudo pluginctl deploy \
  --name image-search-ingest \
  --type deployment \
  --selector resource.gpu=true \
  --resource request.memory=8Gi,limit.memory=16Gi \
  --env-from /secure/path/ingest.env \
  -v /opt/sage/image-search-models:/model \
  -v /var/lib/sage-image-search:/data \
  REGISTRY/IMAGE:TAG

sudo kubectl patch deployment image-search-ingest \
  --type strategic \
  --patch-file deploy/sage/ingest-production-patch.yaml
```

The patch changes updates to `Recreate`, preventing two pods from opening the
same camera during a rollout, and adds an ingest-heartbeat liveness probe.

Inspect without changing the deployment:

```bash
sudo pluginctl logs image-search-ingest
sudo kubectl get deployment,pod -l app=image-search-ingest
sudo kubectl exec deploy/image-search-ingest -- python3 /app/healthcheck.py
```

## Deploy search

1. Replace `QDRANT_PRIVATE_HOST` in `search-config.yaml`.
2. Replace the image in `search-deployment.yaml`.
3. Copy `search-secret.example.yaml` outside the repository and replace its key.
4. Apply all three resources.

```bash
sudo kubectl apply -f deploy/sage/search-config.yaml
sudo kubectl apply -f /secure/path/search-secret.yaml
sudo kubectl apply -f deploy/sage/search-deployment.yaml
sudo kubectl rollout status deployment/image-search-api --timeout=10m
```

For terminal access today, keep the Service private and port-forward it:

```bash
sudo kubectl port-forward service/image-search-api 8099:8099
```

In another terminal:

```bash
SEARCH_API_KEY='the same key' \
python3 search_cli.py "wildfire smoke above trees" \
  --url http://127.0.0.1:8099 \
  --top-k 10
```

This uses the same HTTP contract that a later UI, authenticated Ingress, or
other service will use. Do not expose the ClusterIP by changing it to NodePort
without TLS, authentication, firewall policy, and administrator review.

Each Sage edge node normally has its own WES/k3s cluster. A ClusterIP on one
node is not automatically reachable from another Sage node. Cross-node access
requires an administrator-managed authenticated Ingress or a centrally hosted
search Deployment beside the fixed Qdrant service.
