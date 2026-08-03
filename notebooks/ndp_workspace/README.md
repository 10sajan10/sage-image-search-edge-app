# NDP workspace benchmark search

This directory is a standalone Jupyter workflow for all five Sage image-search
benchmarks. It optionally downloads the pinned Hugging Face datasets,
restores edge_v1 and edge_v2 vector exports into embedded Milvus Lite files,
generates fresh benchmark comparisons, and runs the same custom query against
both versions with 25 lazily loaded results each.

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

The notebook always restores the exported Edge v1 NPZ. This guarantees that
the benchmark and custom search use the original DFN5B-CLIP vectors rather
than silently building a different configuration with another embedding
model.

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

Managed Jupyter users do not need to install or run Git LFS. If a normal clone
contains only a small LFS pointer, the notebook downloads the corresponding
NPZ into an ignored local cache, verifies its byte size and SHA-256, and
continues. Later runs reuse the cache. A clone that already contains the real
NPZ is reused as-is.

For direct inspection without loading NumPy, the exact Edge v2 caption text
and image paths are also tracked in
`assets/ollama_gemma4_e2b_edge_v2_benchmark_captions.jsonl`.

`PortableIndex.caption_vectors`, the NPZ `caption_vectors` entry, and Milvus's
`caption_vector` field persist the dense caption leg. Requesting a positive
caption weight against an index without that leg raises an error rather than
silently changing the configured fusion.

## Configuration choices

The notebook asks the user to:

- download all benchmark data with `HF_TOKEN`, or reuse local Parquet files;
- confirm the path to the exported edge_v1 NPZ;
- confirm the path to the exported edge_v2 NPZ.

The two portable indexes are copied into separate embedded Milvus Lite files:

```text
data/vector_database/edge_v1_benchmarks.milvus.db
data/vector_database/edge_v2_benchmarks.milvus.db
```

Each file stores dataset, image ID, caption, relative image path, image vector,
optional caption vector, and the embedding model ID. Milvus generates a sparse
BM25 field from every caption and searches it natively. Milvus Lite runs in the
Python process; no database container is required by the NDP notebook.

Benchmark metrics retain the historical exact fusion implementation so their
values stay comparable with the bundled runs. The interactive demo searches
the Milvus dense fields and built-in BM25 index. Milvus's analyzer can produce
slightly different lexical scores from the historical `rank-bm25` package.

The original edge_v1 population missed 35 CloudBench images after caption
requests timed out. They were later regenerated with the original Gemma 3
pipeline, bringing both version exports to the complete 32,177-image corpus.

## Benchmark output

The notebook creates fresh local edge_v1 and edge_v2 runs from their exported
vectors and public Parquet relevance labels. It reports every per-query row and
summaries for FireBench, CloudBench, INQUIRE, CommonObjectsBench, SageBench,
and equal-weight overall results. Metrics include MRR, Success@25,
Diversity@25, the two-metric primary score, and the primary-plus-diversity
score. Every summary table is followed by side-by-side bar charts for the two
composite scores across all compared systems.

Immediately after the overall comparison, the notebook sweeps Edge v1 alpha
from `0.00` through `1.00` in `0.05` increments. Alpha is the image-vector
weight and `1 - alpha` is the caption-BM25 weight. It displays every overall
score, saves `data/generated_benchmarks/edge_v1_alpha_sweep_overall.csv`, and
draws separate Primary and Primary-plus-diversity graphs with the best alpha
and the original `0.75` setting marked.

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
cp .env.example .env
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user \
  --name sage-image-search-ndp \
  --display-name "Sage image search (NDP)"
```

Set `HF_TOKEN` in `.env` when downloading datasets. The two NPZ exports are
versioned through Git LFS but are resolved automatically by the notebook;
downloaded benchmark data, model caches, generated Milvus
files, and newly generated result files under `data/` remain ignored.

The five pinned benchmark datasets are intentionally not committed to this
Git repository. Their Parquet files occupy about 12 GiB before image
extraction, exceed a typical free Git LFS storage allowance, and would make
every lab checkout unnecessarily large. For a classroom deployment, publish
the downloaded `data/benchmarking/datasets` directory as one NDP catalog asset
and mount or copy it to that same path before opening the notebook. Users can
then answer `no` to the notebook's download prompt; it reuses those Parquet
files. The notebook never extracts all 32,177 images: benchmark cells read only
scalar label columns, and the custom-query cell reads only result images from
the necessary Parquet row groups. In a local timing check, loading one result
image took about 0.23 seconds; actual batches vary with shard layout and disk.

The final cell prompts once, runs the query against both versions, and displays
the fused score, weighted leg contributions, raw similarities, caption, image,
image ID, and stored path.
