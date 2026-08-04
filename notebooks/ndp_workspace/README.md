# NDP workspace benchmark search

This directory is a standalone Jupyter workflow for all five Sage image-search
benchmarks. It optionally downloads the pinned Hugging Face datasets,
restores edge_v1, edge_v2, and edge_v3 vector exports into embedded Milvus Lite files,
generates fresh benchmark comparisons, and runs the same custom query against
all three versions with 25 lazily loaded results each.

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
vectors, caption vectors, image IDs, and relative image paths. The tracked
`edge_v1_export_manifest.json`, `edge_v2_export_manifest.json`, and
`edge_v3_export_manifest.json` identify the portable indexes by model, shape,
dataset counts, byte size, and SHA-256 checksum.

### edge_v3

- Captioner: Ollama `gemma4:e2b`, thinking disabled, approximately 250 words
  per image.
- Embedder: `jinaai/jina-clip-v2`, normalized 1024-dimensional image and
  caption vectors.
- Query encoding: Jina `retrieval.query`, stored in the portable export so it
  cannot be accidentally omitted.
- Caption representations: dense caption vector plus caption text for BM25.
- Benchmark fusion: 60% image vector, 25% caption vector, and 15% caption BM25.

`data/edge_v3_benchmarks.npz` is the complete 32,177-record export from the
finished Qdrant collection. Its manifest is
`edge_v3_export_manifest.json`; all three NPZ exports are tracked with Git LFS.

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
- confirm the path to the exported edge_v2 NPZ;
- confirm the path to the exported edge_v3 NPZ.

The three portable indexes are copied into separate embedded Milvus Lite files:

```text
data/vector_database/edge_v1_benchmarks.milvus.db
data/vector_database/edge_v2_benchmarks.milvus.db
data/vector_database/edge_v3_benchmarks.milvus.db
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
pipeline, bringing both DFN5B version exports to the complete 32,177-image corpus.

## Benchmark output

The notebook creates fresh local edge_v1, edge_v2, and edge_v3 runs from their exported
vectors and public Parquet relevance labels. It reports every per-query row and
summaries for FireBench, CloudBench, INQUIRE, CommonObjectsBench, SageBench,
and equal-weight overall results. Metrics include MRR, Success@25,
Diversity@25, the two-metric primary score, and the primary-plus-diversity
score. Every summary table is followed by side-by-side bar charts for the two
composite scores across all compared systems.

The checked comparison below uses the original fusion weights and Edge v3's
required `retrieval.query` task. Values are the primary score,
`(MRR + Success@25) / 2`; overall gives every benchmark equal weight.

| benchmark | edge_v1 | edge_v2 | edge_v3 |
|---|---:|---:|---:|
| FireBench | 0.5509 | 0.5492 | 0.4604 |
| CloudBench | 0.1769 | 0.1669 | 0.1505 |
| INQUIRE | 0.8084 | 0.7339 | 0.6232 |
| CommonObjectsBench | 0.6292 | 0.6173 | 0.5781 |
| SageBench | 0.3787 | 0.3644 | 0.2756 |
| **Overall** | **0.5088** | **0.4863** | **0.4176** |

Overall Edge v3 MRR is `0.2332`, Success@25 is `0.6019`, Diversity@25 is
`0.1530`, and its three-metric score is `0.3294`. Because this rerun explicitly
used `retrieval.query`, the remaining gap is real for this benchmark and fusion
configuration; it is not explained by the previously omitted Jina task alone.

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
data/generated_benchmarks/edge_v3
data/generated_benchmarks/all_comparison_query_results.csv
```

Bundled results include `baseline`, `v10`, `v11`, `v12`, `edge_v1`, and
`edge_v2`, and `edge_v3`. The saved Edge CSVs are available for direct inspection but are not
loaded by the notebook; all three Edge rows are generated from the selected
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

Set `HF_TOKEN` in `.env` when downloading datasets. The three NPZ exports are
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

Edge v3's trusted Jina implementation requires `transformers==4.53.3` and
imports `flash-attn` even though flash attention is disabled at model load.
Run this workspace in an NVIDIA/CUDA NDP image capable of installing that
dependency. The workspace also pins the Jina model and trusted implementation
revisions so a later upstream Python-code update cannot silently change a
benchmark run.

The final cell prompts once, runs the query against all three versions, and displays
the fused score, weighted leg contributions, raw similarities, caption, image,
image ID, and stored path.
