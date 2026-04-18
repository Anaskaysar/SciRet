# SciRet v2 Code Documentation — Part 2 (Examples, Failure Cases, Testing)

This is the practical companion to `0_docs/codebase_line_by_line_documentation.md`.

It answers:
- What goes in and out of each module?
- What usually breaks?
- What exact command should I run to test each stage?

---

## 1) End-to-End Data Flow (Quick Map)

1. `1_data/raw/metadata.csv`
2. `01_data_exploration.ipynb` -> `1_data/processed/papers_clean.parquet`
3. `02_text_chunking.ipynb` -> `1_data/processed/chunks.parquet`
4. `03_embedding_baseline.ipynb` -> `1_data/embeddings/text_embeddings.parquet`
5. `04` + `05` -> retrieval and rerank artifacts in `4_results/retrieval/`
6. `06` + `07` -> figure artifacts in `1_data/processed` and `1_data/embeddings`
7. `08` -> multimodal traces in `4_results/multimodal/`
8. `09` -> report tables in `4_results/`

---

## 2) Module-by-Module Input/Output Examples

## `2_src/data/loader.py`

**Input example**
- `metadata.csv` row:
  - `cord_uid = "abc123"`
  - `title = "Clinical Features..."`
  - `abstract = "COVID-19 patients showed..."`

**Output example**
- Filtered/sampled DataFrame, then file:
  - `1_data/processed/papers_clean.parquet`

**What to verify**
- `cord_uid` duplicates removed
- short/missing abstracts excluded
- sample count matches `tier_size` (or less if source is smaller)

---

## `2_src/data/chunker.py`

**Input example**
- One row text:
  - `"Title A ... Abstract words ..."`
- Config:
  - `chunk_size=400`, `overlap=50`

**Output example**
- Chunk records:
  - `abc123_chunk_000`
  - `abc123_chunk_001`
  - ...

**What to verify**
- `chunk_id` stable and deterministic
- `chunk_index` increments correctly
- overlapping windows are produced

---

## `2_src/embeddings/text_embedder.py`

**Input example**
- `chunks_df` with:
  - `chunk_id`
  - `chunk_text`

**Output example**
- `text_embeddings.parquet` with:
  - `chunk_id`
  - `vector` (list of floats)
- `embed_manifest.json` with:
  - `model_name`
  - `dim`
  - `count`

**What to verify**
- second run reuses cache (no recompute)
- vector count equals chunk count

---

## `2_src/retrieval/bm25_retriever.py`

**Input example**
- Docs:
  - `["covid fever cough", "cardiac complications"]`
- Query:
  - `"covid cough"`

**Output example**
- Ranked pairs:
  - `[("doc_1", 1.23), ("doc_2", 0.02)]`

**What to verify**
- exact term matches rank higher
- empty query returns near-zero scores

---

## `2_src/retrieval/dense_retriever.py`

**Input example**
- Embedding table:
  - `chunk_id`
  - `vector`
- Query text

**Output example**
- Top-k dense matches:
  - `[("abc123_chunk_000", 0.88), ...]`

**What to verify**
- matrix shape is `(N, dim)`
- query returns `top_k` or fewer when corpus is small

---

## `2_src/retrieval/hybrid_retriever.py`

**Input example**
- Run 1: dense ranked list
- Run 2: BM25 ranked list

**Output example**
- Fused RRF ranking:
  - docs that appear in both runs rise to top

**What to verify**
- fused ranking is deterministic
- changing `k` changes smoothing behavior

---

## `2_src/retrieval/reranker.py`

**Input example**
- Candidate tuples:
  - `(doc_id, doc_text, base_score)`
- Query string

**Output example**
- Reordered top-k candidates based on blended score

**What to verify**
- overlap-sensitive reranking
- top-k cutoff respected

---

## `2_src/generation/text_generator.py`

**Input example**
- Query + list of `(chunk_id, chunk_text)`

**Output example**
- Multi-line answer string with citation-like IDs `[chunk_id]`

**What to verify**
- empty contexts -> fallback message
- non-empty contexts include IDs and previews

---

## `2_src/evaluation/ragas_eval.py`

**Input example**
- Relevant IDs: `["d1", "d3"]`
- Retrieved IDs: `["d2", "d3", "d1"]`

**Output example**
- `Recall@2 = 0.5`
- `MRR = 0.5`
- `NDCG@3` in `[0,1]`

**What to verify**
- metrics match hand-calculated toy examples

---

## `2_src/pipeline.py`

**Input example**
- Build mode: raw `metadata.csv`
- Query mode: user question text

**Output example**
- `QueryResult`:
  - `answer` (string)
  - `sources` (chunk IDs)
  - `debug` (dense/sparse/fused snapshots)

**What to verify**
- build writes chunks + embeddings
- query returns at least one source for common COVID terms

---

## 3) Common Failure Cases and Fixes

## Data / path errors

- **Error:** `FileNotFoundError: metadata.csv not found`
  - **Cause:** raw file missing in `1_data/raw/`
  - **Fix:** place Kaggle `metadata.csv` there or update loader path.

- **Error:** `Missing chunk file`
  - **Cause:** query before running build/index
  - **Fix:** run `build_from_metadata(...)` or `01`->`03` notebooks first.

## Environment errors

- **Error:** `ModuleNotFoundError: pandas`
  - **Cause:** venv not activated
  - **Fix:** `source venv/bin/activate` then rerun.

- **Error:** parquet read/write engine missing
  - **Cause:** missing `pyarrow`/fastparquet
  - **Fix:** install from `requirements.txt`.

## Retrieval quality issues

- **Issue:** irrelevant top results
  - **Cause:** baseline hash embedder is simplistic
  - **Fix:** later replace with BGE-M3 + true reranker.

- **Issue:** too few chunks
  - **Cause:** strict abstract filter or short texts
  - **Fix:** lower `min_abstract_chars` or chunk min token threshold.

## Notebook sequencing issues

- **Issue:** Notebook `03` says cache hit but outputs stale
  - **Cause:** old cache files
  - **Fix:** delete `1_data/embeddings/text_embeddings.parquet` and manifest; rerun.

- **Issue:** `04/05` have TODO and no metrics yet
  - **Cause:** currently scaffold stage
  - **Fix:** wire in retriever/reranker save logic (planned next iteration).

---

## 4) Exact Test Commands (Copy/Paste)

Run all commands from repo root.

## Step A: activate environment

```bash
source venv/bin/activate
python --version
```

## Step B: import smoke test

```bash
python -c "import sys; sys.path.append('2_src'); from pipeline import SciRetPipeline; print('import_ok')"
```

## Step C: build Tier 1 index from metadata

```bash
python -c "import sys; sys.path.append('2_src'); from pipeline import SciRetPipeline; p=SciRetPipeline(); p.build_from_metadata(tier_size=1000, seed=42); print('build_ok')"
```

## Step D: query test

```bash
python -c "import sys; sys.path.append('2_src'); from pipeline import SciRetPipeline; p=SciRetPipeline(); p.load_index(); r=p.query('What are common symptoms of COVID-19?'); print(r.answer); print(r.sources); print(r.debug)"
```

## Step E: verify output files exist

```bash
python -c "from pathlib import Path; root=Path('.'); paths=[root/'1_data/processed/papers_clean.parquet', root/'1_data/processed/chunks.parquet', root/'1_data/embeddings/text_embeddings.parquet']; print({str(p): p.exists() for p in paths})"
```

---

## 5) Notebook-by-Notebook Validation Checklist

## `01_data_exploration.ipynb`
- Confirm printed count near target tier size.
- Confirm `papers_clean.parquet` and `tier_manifest.json` exist.

## `02_text_chunking.ipynb`
- Confirm chunk count > paper count.
- Confirm `chunks.parquet` and `chunk_config.json` exist.

## `03_embedding_baseline.ipynb`
- First run: writes embedding files.
- Second run: prints cache hit.

## `04_hybrid_retrieval.ipynb`
- Confirms chunks + embeddings loaded.
- Creates/uses `4_results/retrieval/`.

## `05_reranking.ipynb`
- Confirms retrieval folder exists.
- Ready for cross-encoder integration.

## `06_figure_extraction.ipynb`
- Creates `figures_manifest.parquet`.

## `07_clip_embeddings.ipynb`
- Reads figure manifest.
- Creates `figure_embeddings.parquet`.

## `08_multimodal_pipeline.ipynb`
- Creates `4_results/multimodal/sample_trace.json`.

## `09_evaluation.ipynb`
- Creates `4_results/comparison_table.md`.

---

## 6) What to Replace Next (Production Upgrade Path)

1. Replace hash text embedding with `SentenceTransformer` BGE-M3.
2. Replace overlap reranker with cross-encoder model.
3. Add persistence of retrieval runs (`query_id`, ranked IDs, scores).
4. Add real figure extraction and CLIP embedding.
5. Add true RAGAS/faithfulness evaluation over fixed query set.

This lets you move from educational baseline -> publishable experimental stack while keeping architecture stable.
