# NRP workspace benchmark search

This directory is a standalone Jupyter workflow for all five Sage image-search
benchmarks. It optionally downloads the pinned Hugging Face datasets,
materializes their images, builds or restores MobileCLIP2 vectors, populates an
embedded SQLite database, generates fresh edge_v1 benchmark results, and runs
custom visual searches returning 25 results.

The configuration cell asks the user to make two choices:

- Download all benchmark data with the `HF_TOKEN` stored in `.env`, or reuse
  benchmark data already present under this workspace.
- Build image and caption vectors for every Gemma-covered benchmark image with
  MobileCLIP2, or restore those vectors from an NPZ backup supplied by the
  user.

In build mode, captions come exclusively from the bundled original Gemma 3
export:

```text
assets/gemma3_4b_it_edge_v1_benchmark_captions.jsonl
```

Each source row identifies `model` as `gemma-3-4b-it`. The notebook does not
launch Gemma and does not replace these captions with benchmark summaries. It
resolves the captions to downloaded image paths, embeds each covered image and
caption with MobileCLIP2, then saves the result as
`data/backups/mobileclip2_benchmarks.npz`. In backup mode, the user enters the
path to an existing compatible NPZ file.

Both modes populate `data/vector_database/mobileclip2_benchmarks.sqlite3`.
SQLite is the file-based database: it stores dataset, image ID, caption, image
path, image vector, caption vector, and model metadata. It runs directly in
Python and requires no container, service, or external database.

The Gemma export covers 32,142 of the 32,177 benchmark images. The 35 missing
CloudBench captions are original generation failures. They are reported as a
coverage gap and are not silently replaced with non-Gemma text.

Before the free-form search demo, the notebook generates a new edge_v1
per-query run from the SQLite database and public Parquet relevance labels. It
writes those results below `data/generated_benchmarks/`, then shows one
comparison cell each for FireBench, CloudBench, INQUIRE, CommonObjectsBench,
SageBench, and an equal-weight overall table. Each table contains MRR,
Success@25, Diversity@25, the two-metric primary score, and the three-metric
primary-plus-diversity score. Each benchmark cell also exposes every per-query
row, and the complete combined dataset is saved as
`data/generated_benchmarks/all_comparison_query_results.csv`.

Every per-query comparison CSV from
`ImageSearchatEdge@049f6384d7e80c11666701bb320a09727a7d8133` is bundled under
`results/benchmarks/*/results/{baseline,v10,v11,v12}`. The notebook therefore
does not depend on the separate ImageSearchatEdge GitHub repository. Existing
`edge_v1` and `edge_v2` result directories are intentionally absent: the
selected index's fresh row is computed by this notebook. Historical versions
do not all share an identical evaluation protocol, so the tables are
demonstration comparisons rather than publication-quality head-to-head claims.

## NRP/Jupyter setup

```bash
git clone https://github.com/10sajan10/sage-image-search-edge-app
cd sage-image-search-edge-app/notebooks/ndp_workspace
cp .env.example .env
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user \
  --name sage-image-search-ndp \
  --display-name "Sage image search (NRP)"
```

Set `HF_TOKEN` in `.env`. The notebook requires it whenever the user chooses to
download benchmark data or build MobileCLIP2 vectors. Interactive choices and
internal paths are deliberately not stored in `.env`.

Benchmark paths are not environment variables. The notebook creates its own
clone-relative directories automatically:

```text
ndp_workspace/data/benchmarking/datasets
ndp_workspace/data/benchmarking/images
ndp_workspace/data/gemma3_4b_it_edge_v1_benchmark_captions_resolved.jsonl
ndp_workspace/data/backups/mobileclip2_benchmarks.npz
ndp_workspace/data/vector_database/mobileclip2_benchmarks.sqlite3
ndp_workspace/data/generated_benchmarks
```

The final cell prompts for a custom text query and prints each result's fused
score, component scores, caption, rendered image, image ID, and stored path.
