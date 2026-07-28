# Operational scripts

- `build-apps.sh` builds and optionally pushes both role-specific images.
- `deploy-ingest.sh` creates the long-running Sage ingestion Deployment.
- `submit-ingest-job.sh` renders submit-time environment values and submits a
  scheduled one-shot ingestion job.
- `deploy-search.sh` applies the private, always-running search service.
- `query.sh` calls that service from a terminal.
- `check-dependencies.sh` checks Qdrant, Ollama, model files, and CUDA.
- `e2e.sh` exercises ingestion, restart idempotence, authentication, and
  ranking against a temporary collection.

Run these scripts from any directory; they resolve the repository root
internally. Deployment scripts never store secrets in this repository.
