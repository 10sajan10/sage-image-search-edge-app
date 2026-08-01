# NRP workspace benchmark search

This directory is a standalone Jupyter workflow for all five Sage image-search
benchmarks. It optionally downloads the pinned Hugging Face datasets,
materializes their images, builds or restores image vectors, populates an
embedded SQLite database, generates fresh edge_v1 benchmark results, and runs
custom visual searches returning 25 results.

## Retrieval legs

Search fuses up to three legs. **edge_v1 uses two**: a CLIP **image vector**,
and **BM25 over the Gemma captions**. Captions are not embedded as dense
vectors, because `baseline`, `v10`, `v11`, and `v12` all ran
`clip_hybrid_query` against a single `clip` image vector with the caption text
searched lexically (`alpha=0.4` for v10–v12, `1.0` for baseline) — a dense
caption leg would not be comparable to any of them. The default
`IMAGE_WEIGHT=0.40` / `BM25_WEIGHT=0.60` mirrors that `alpha=0.4` split.

The third leg stays first-class for **edge_v2**, which will have caption
vectors:

```python
EMBED_CAPTIONS = True     # build_index embeds captions with the same encoder
CAPTION_WEIGHT = 0.25     # then the dense caption leg contributes
```

`PortableIndex.caption_vectors`, the NPZ `caption_vectors` entry, and the
SQLite `caption_vector` column all persist the leg; they are `None`/NULL for an
image-only index. Passing `caption_weight > 0` to `search_index` or
`evaluate_benchmarks` against an index with no caption vectors **raises**
rather than silently reweighting — quietly zeroing it would let an edge_v1 and
an edge_v2 run report identical configured weights while fusing differently,
making their benchmark scores look comparable when they are not.

Reported component scores are normalized the same way as the fused total, so
`score == image_score + caption_score + bm25_score` exactly. Raw
`image_similarity` (cosine) and `bm25_raw` are reported alongside.

## Configuration choices

The configuration cell asks the user to make two choices:

- Download all benchmark data with the `HF_TOKEN` stored in `.env`, or reuse
  benchmark data already present under this workspace.
- Build vectors for every Gemma-covered benchmark image, or restore vectors
  from an NPZ backup.

In build mode, captions come exclusively from the bundled original Gemma 3
export:

```text
assets/gemma3_4b_it_edge_v1_benchmark_captions.jsonl
```

Each source row identifies `model` as `gemma-3-4b-it`. The notebook does not
launch Gemma and does not replace these captions with benchmark summaries. It
resolves the captions to downloaded image paths, embeds each covered image
(and, with `EMBED_CAPTIONS = True`, each caption) using `MOBILECLIP_MODEL_ID`,
then saves the result as `data/backups/<model>_benchmarks.npz` — a separate
file, so a build never overwrites the bundled backup. In backup mode the user
enters the path to an existing compatible NPZ file; that default is the bundled
edge_v1 `apple/DFN5B-CLIP-ViT-H-14-378` image index, which is gitignored, so on
a fresh clone choose **build**.

Both modes populate `data/vector_database/mobileclip2_benchmarks.sqlite3`.
SQLite is the file-based database: it stores dataset, image ID, caption, image
path, image vector, an optional caption vector, and model metadata. It runs
directly in Python and requires no container, service, or external database.

The Gemma export covers all 32,177 benchmark images.

The original edge_v1 run left 35 CloudBench images uncaptioned: every one failed
with `RuntimeError('timed out')`, in three short bursts on a single node, so
their images extracted fine but the population step skipped them and the
collection settled at 32,142. They were backfilled with
`ImageSearchatEdge/scripts/backfill_missing_captions.py`, which re-runs the
original pipeline's own functions -- same `gemma-3-4b-it` endpoint, prompt,
caption cleaner, DFN5B encoder, deterministic UUID, and object payload -- so the
backfilled records are structurally identical to the rest. They were never
replaced with non-Gemma text.

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
download benchmark data or build vectors. Interactive choices and
internal paths are deliberately not stored in `.env`.

Benchmark paths are not environment variables. The notebook creates its own
clone-relative directories automatically:

```text
ndp_workspace/data/benchmarking/datasets
ndp_workspace/data/benchmarking/images
ndp_workspace/data/gemma3_4b_it_edge_v1_benchmark_captions_resolved.jsonl
ndp_workspace/data/edge_v1_benchmarks.npz          (bundled, backup mode)
ndp_workspace/data/backups/<model>_benchmarks.npz  (written by build mode)
ndp_workspace/data/vector_database/mobileclip2_benchmarks.sqlite3
ndp_workspace/data/generated_benchmarks
```

The final cell prompts for a custom text query and prints each result's fused
score, its weighted per-leg components (which sum to the score), the raw
cosine and BM25 values, caption, rendered image, image ID, and stored path.
