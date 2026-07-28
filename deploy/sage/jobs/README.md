# Scheduled ingestion jobs

`ingest-oneshot.example.yaml` shows the scheduler-native job schema. It fixes
`RUN_MODE=oneshot`, so every scheduled invocation captures one frame, drains
ready spool work, and exits.

The scheduler supports application environment values under
`pluginSpec.env`. `sesctl` does not provide individual `--env` overrides, so
`scripts/submit-ingest-job.sh` renders those values from the submitter's
environment into a temporary job document before calling `sesctl`.

Required submission values:

```bash
INGEST_IMAGE=registry.example/image-search-ingest:0.3.3
QDRANT_URL=http://qdrant.example:6333
OLLAMA_URL=http://ollama.example:11434
```

Common optional values:

```bash
NODE_ID=H01E
COLLECTION=edge_v4_live
SCHEDULE='*/5 * * * *'
CAMERA_SECRET_NAME=image-search-ingest
CAMERA_SECRET_KEY=CAMERA
MODEL_ROOT=/opt/sage/image-search-models
DATA_ROOT=/var/lib/sage-image-search
JOB_ENV_FILE=/secure/path/ingest-job.env
DRY_RUN=false
```

`JOB_ENV_FILE` accepts additional `KEY=VALUE` settings such as
`CAPTION_TARGET_WORDS=250`. Keep RTSP credentials in the Sage scheduler secret;
the helper deliberately prevents `CAMERA` from being supplied as plaintext.

Inspect before submission:

```bash
DRY_RUN=true scripts/submit-ingest-job.sh
```

Submit:

```bash
scripts/submit-ingest-job.sh
```

On H01E, scheduled GPU jobs also require administrator-provided NVIDIA
RuntimeClass injection. See the parent deployment README before enabling the
cron rule.
