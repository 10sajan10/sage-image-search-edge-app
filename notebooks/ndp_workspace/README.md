# NDP workspace benchmark search

This directory is a standalone Jupyter workflow for all five Sage image-search
benchmarks. It optionally downloads the pinned Hugging Face datasets,
materializes their images, restores edge_v1 and edge_v2 vector exports into
embedded SQLite files, generates fresh benchmark comparisons, and runs the
same custom query against both versions with 25 rendered results each.

## Models and retrieval

The version descriptions deliberately focus on the captioning and embedding
models; the notebook's local storage mechanism is an implementation detail.

### edge_v1

- Captioner: `gemma-3-4b-it`, served through vLLM, deterministic generation,
  approximately 150 words per image.
- Embedder: `apple/DFN5B-CLIP-ViT-H-14-378`, FP16, normalized 1024-dimensional
  image vectors.
- Caption representation: stored text searched with BM25; no dense caption
  vector.
- Benchmark fusion: 75% image vector and 25% caption BM25.

The notebook can restore the exported edge_v1 NPZ or build a user-owned image
index with `timm/MobileCLIP2-S0-OpenCLIP`. Build mode reuses the bundled Gemma
3 captions and never launches the caption model. It does not create caption
vectors.

### edge_v2

- Captioner: Ollama `gemma4:e2b`, thinking enabled, approximately 50 words per
  image so the caption fits DFN5B-CLIP's 77-token text limit.
- Embedder: `apple/DFN5B-CLIP-ViT-H-14-378`, FP16, normalized 1024-dimensional
  image and caption vectors.
- Caption representations: dense caption vector plus caption text for BM25.
- Benchmark fusion: 60% image vector, 25% caption vector, and 15% caption BM25.

edge_v2 is never rebuilt in this notebook. `data/edge_v2_benchmarks.npz` is an
export from the completed run and contains all 32,177 caption strings, image
vectors, caption vectors, image IDs, and relative image paths. Both NPZ files
are tracked with Git LFS. The tracked `edge_v1_export_manifest.json` and
`edge_v2_export_manifest.json` identify them by model, shape, dataset counts,
byte size, and SHA-256 checksum.

`PortableIndex.caption_vectors`, the NPZ `caption_vectors` entry, and SQLite's
`caption_vector` column persist the dense caption leg. Requesting a positive
caption weight against an index without that leg raises an error rather than
silently changing the configured fusion.

## Configuration choices

The notebook asks the user to:

- download all benchmark data with `HF_TOKEN`, or reuse local Parquet files;
- build edge_v1 image vectors or restore its NPZ export;
- confirm the path to the exported edge_v2 NPZ.

The two portable indexes are copied into separate serverless SQLite files:

```text
data/vector_database/edge_v1_benchmarks.sqlite3
data/vector_database/edge_v2_benchmarks.sqlite3
```

Each file stores dataset, image ID, caption, relative image path, image vector,
optional caption vector, and the embedding model ID. SQLite runs directly in
Python; no database container is required by the NDP notebook.

The original edge_v1 population missed 35 CloudBench images after caption
requests timed out. They were later regenerated with the original Gemma 3
pipeline, bringing both version exports to the complete 32,177-image corpus.

## Benchmark output

The notebook creates fresh local edge_v1 and edge_v2 runs from their exported
vectors and public Parquet relevance labels. It reports every per-query row and
summaries for FireBench, CloudBench, INQUIRE, CommonObjectsBench, SageBench,
and equal-weight overall results. Metrics include MRR, Success@25,
Diversity@25, the two-metric primary score, and the primary-plus-diversity
score.

Generated files are written below:

```text
data/generated_benchmarks/edge_v1
data/generated_benchmarks/edge_v2
data/generated_benchmarks/all_comparison_query_results.csv
```

Bundled results include `baseline`, `v10`, `v11`, `v12`, `edge_v1`, and
`edge_v2`. The saved Edge CSVs are available for direct inspection but are not
loaded by the notebook; both Edge rows are generated from the selected
portable files. Historical systems do not all share one evaluation protocol,
so the tables remain demonstration comparisons rather than
publication-quality head-to-head claims.

## NDP/Jupyter setup

```bash
git clone https://github.com/10sajan10/sage-image-search-edge-app
cd sage-image-search-edge-app/notebooks/ndp_workspace
git lfs install
git lfs pull
cp .env.example .env
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user \
  --name sage-image-search-ndp \
  --display-name "Sage image search (NRP)"
```

Set `HF_TOKEN` in `.env` when downloading datasets or building the optional
edge_v1 image index. The two NPZ exports are versioned through Git LFS;
downloaded benchmark data, extracted images, model caches, generated SQLite
files, and newly generated result files under `data/` remain ignored.

The final cell prompts once, runs the query against both versions, and displays
the fused score, weighted leg contributions, raw similarities, caption, image,
image ID, and stored path.
