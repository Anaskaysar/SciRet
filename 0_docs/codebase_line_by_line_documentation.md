# SciRet v2 Code Documentation (Line-by-Line)

This document explains every code file in `2_src` and every notebook in `3_notebooks`.
For Python files, line numbers match the file exactly.  
For notebooks, code is explained line-by-line inside each code cell (cell-local numbering).

---

## `2_src/__init__.py`

- `L1`: Adds a package docstring so `2_src` can be treated as a Python package namespace.

---

## `2_src/data/loader.py`

- `L1`: Enables postponed evaluation of type hints.
- `L3`: Imports `Path` for filesystem path handling.
- `L4`: Imports `Optional` for nullable type annotations.
- `L6`: Imports `pandas` for tabular data I/O and transformations.
- `L9`: Declares `CORDDataLoader` class.
- `L10`: Class docstring describing role (load/save CORD-19 tabular artifacts).
- `L12`: Constructor starts; accepts optional project root.
- `L13`: Sets root to provided path; otherwise infers repo root from file location.
- `L14`: Builds raw data directory path (`1_data/raw`).
- `L15`: Builds processed data directory path (`1_data/processed`).
- `L16`: Ensures processed directory exists.
- `L18`: `load_metadata` method signature with default filename.
- `L19`: Builds full metadata path.
- `L20`: Checks if metadata file exists.
- `L21`: Raises clear error if missing.
- `L22`: Reads CSV into DataFrame.
- `L24-L30`: `build_tier_subset` method signature and defaults (size, seed, min abstract chars).
- `L31`: Creates copy to avoid modifying original input DataFrame.
- `L32`: Checks if `cord_uid` column exists.
- `L33`: Drops duplicate papers by unique CORD id.
- `L34`: Checks if abstract column exists.
- `L35`: Builds filter mask for minimum abstract length.
- `L36`: Applies mask.
- `L37`: Computes actual sample size bounded by available rows.
- `L38`: Returns deterministic random sample (or empty frame if no rows).
- `L40`: `save_clean_subset` method signature.
- `L41`: Builds output path.
- `L42`: Saves DataFrame to parquet.
- `L43`: Returns saved path for downstream use.

---

## `2_src/data/chunker.py`

- `L1`: Enables postponed evaluation of annotations.
- `L3`: Imports `dataclass` for config object.
- `L4`: Imports typing helpers.
- `L6`: Imports `pandas`.
- `L9`: Declares `ChunkConfig` dataclass.
- `L11`: Default chunk size in tokens/words surrogate.
- `L12`: Default overlap between adjacent chunks.
- `L13`: Minimum chunk length threshold.
- `L16`: Internal generator function to slice text tokens into windows.
- `L17`: Computes sliding step (`chunk_size - overlap`) with floor protection.
- `L18`: Iterates token start offsets.
- `L19`: Yields token slice for each window.
- `L22`: Public helper that chunks one text string.
- `L23`: Tokenizes by whitespace.
- `L24`: Initializes output list.
- `L25`: Iterates generated token windows.
- `L26`: Enforces minimum chunk length.
- `L27`: Joins token list back into chunk text and appends.
- `L28`: Returns final chunk list.
- `L31`: Converts full papers DataFrame into chunk-level DataFrame.
- `L32`: Initializes row accumulator.
- `L33`: Iterates each paper row.
- `L34`: Reads `cord_uid` with safe fallback and explicit string cast.
- `L35`: Reads title.
- `L36`: Reads abstract.
- `L37`: Combines title + abstract into one chunkable text string.
- `L38`: Calls `chunk_text` with config and enumerates resulting chunks.
- `L39-L47`: Appends one chunk record dict with stable IDs and metadata.
- `L48`: Returns chunk records as DataFrame.

---

## `2_src/data/pdf_parser.py`

- `L1`: Enables postponed annotations.
- `L3`: Imports `Path`.
- `L4`: Imports typing helpers.
- `L7`: Defines PDF discovery utility.
- `L8`: Early return if root folder does not exist.
- `L10`: Recursively finds all `*.pdf` files and sorts for determinism.
- `L13`: Defines placeholder manifest extractor.
- `L14-L17`: Docstring clarifies this is a stub to be replaced by real parser logic.
- `L18`: Initializes output list.
- `L19`: Iterates PDF paths.
- `L20-L27`: Appends one synthetic manifest record per PDF.
- `L28`: Returns manifest records list.

---

## `2_src/embeddings/text_embedder.py`

- `L1`: Enables postponed annotations.
- `L3`: Imports `hashlib` for deterministic token hashing.
- `L4`: Imports `json` for manifest writing.
- `L5`: Imports `Path`.
- `L6`: Imports typing helpers.
- `L8`: Imports numpy.
- `L9`: Imports pandas.
- `L12`: Declares `TextEmbedder`.
- `L13-L16`: Docstring: deterministic local embedder, replace with BGE for final runs.
- `L18`: Constructor signature with vector dim and model label.
- `L19`: Stores dimension.
- `L20`: Stores model name string.
- `L22`: Private method to embed one text.
- `L23`: Initializes zero vector.
- `L24`: Tokenizes text and iterates tokens.
- `L25`: Hashes token string to integer.
- `L26`: Uses modulo hash to increment one vector bin (hashed bag-of-words).
- `L27`: Computes L2 norm.
- `L28`: Returns normalized vector (or zero vector if empty).
- `L30`: Public batch `encode`.
- `L31`: Stacks all single-text vectors into 2D matrix.
- `L33-L38`: `cache_or_build` method signature.
- `L39`: Cache check.
- `L40`: Loads existing embeddings from parquet if present.
- `L42`: Builds vectors from chunk texts if cache missing.
- `L43-L48`: Constructs output DataFrame with `chunk_id` and vector list.
- `L49`: Ensures output directory exists.
- `L50`: Writes embedding parquet.
- `L52`: Optional manifest path guard.
- `L53`: Builds manifest dictionary.
- `L54`: Writes JSON manifest with indentation.
- `L55`: Returns embedding DataFrame.

---

## `2_src/embeddings/vision_embedder.py`

- `L1`: Enables postponed annotations.
- `L3`: Imports `hashlib`.
- `L4`: Imports `Iterable`.
- `L6`: Imports numpy.
- `L9`: Declares `VisionEmbedder`.
- `L10-L13`: Docstring notes deterministic stub and CLIP replacement plan.
- `L15`: Constructor signature.
- `L16`: Stores output dimension.
- `L17`: Stores model label.
- `L19`: Encodes figure/asset identifiers.
- `L20`: Initializes output row list.
- `L21`: Iterates each asset id.
- `L22`: Creates zero vector.
- `L23`: Hashes asset id.
- `L24`: Sets one hashed position to 1.0.
- `L25`: Appends vector.
- `L26`: Returns stacked matrix or empty matrix when no assets.

---

## `2_src/retrieval/bm25_retriever.py`

- `L1`: Enables postponed annotations.
- `L3`: Imports `math`.
- `L4`: Imports `Counter` and `defaultdict`.
- `L5`: Imports typing helpers.
- `L8`: Defines tokenizer helper.
- `L9`: Lowercases text, splits by whitespace, removes empty tokens.
- `L12`: Declares BM25 retriever class.
- `L13`: Constructor signature with BM25 hyperparameters.
- `L14`: Stores `k1`.
- `L15`: Stores `b`.
- `L16-L21`: Initializes corpus state containers.
- `L23`: `fit` method signature.
- `L24-L25`: Stores input IDs and docs.
- `L26-L28`: Resets cached stats.
- `L29`: Iterates each document text.
- `L30`: Tokenizes document.
- `L31`: Builds term frequency counter.
- `L32`: Saves doc term frequencies.
- `L33`: Saves document length.
- `L34`: Iterates unique terms in current doc.
- `L35`: Increments document frequency per term.
- `L36`: Computes average document length.
- `L38`: IDF helper method signature.
- `L39`: Number of documents.
- `L40`: Document frequency for term.
- `L41`: BM25-style smoothed IDF formula.
- `L43`: Query method signature.
- `L44`: Tokenizes query.
- `L45`: Initializes score list.
- `L46`: Iterates each indexed doc.
- `L47`: Reads doc length.
- `L48`: Initializes doc score.
- `L49`: Iterates query terms.
- `L50`: Reads query-term frequency in doc.
- `L51-L52`: Skips if term absent.
- `L53`: Computes term IDF.
- `L54`: Computes BM25 denominator with length normalization.
- `L55`: Adds BM25 term contribution to score.
- `L56`: Appends `(doc_id, score)`.
- `L57`: Sorts by descending score.
- `L58`: Returns top-k.

---

## `2_src/retrieval/dense_retriever.py`

- `L1`: Enables postponed annotations.
- `L3`: Imports typing helpers.
- `L5`: Imports numpy.
- `L6`: Imports pandas.
- `L8`: Imports `TextEmbedder`.
- `L11`: Declares dense retriever class.
- `L12`: Constructor signature.
- `L13`: Stores embedder instance.
- `L14`: Initializes document ID list.
- `L15`: Initializes dense matrix placeholder.
- `L17`: Fit method signature.
- `L18`: Reads `chunk_id` column into index list.
- `L19`: Converts stored vector column into float32 matrix.
- `L21`: Query method signature.
- `L22-L23`: Returns empty if index not initialized.
- `L24`: Embeds query text.
- `L25`: Dot product against all document vectors.
- `L26`: Gets descending top-k indices.
- `L27`: Returns ranked `(doc_id, score)` list.

---

## `2_src/retrieval/hybrid_retriever.py`

- `L1`: Enables postponed annotations.
- `L3`: Imports `defaultdict`.
- `L4`: Imports typing helpers.
- `L7-L11`: Defines RRF function signature.
- `L12`: Initializes fused score dictionary.
- `L13`: Iterates each retrieval run.
- `L14`: Iterates ranked docs with 1-based rank.
- `L15`: Adds reciprocal-rank contribution.
- `L16`: Sorts fused docs by score.
- `L17`: Returns top-k fused results.

---

## `2_src/retrieval/reranker.py`

- `L1`: Enables postponed annotations.
- `L3`: Imports typing helpers.
- `L6`: Helper to create lowercase token set.
- `L7`: Returns de-duplicated token set.
- `L10`: Declares overlap-based reranker.
- `L11-L14`: Docstring says this is a lightweight placeholder.
- `L16-L21`: `rerank` method signature; input includes doc_id, text, base_score.
- `L22`: Tokenizes query into set.
- `L23`: Initializes rescored list.
- `L24`: Iterates candidates.
- `L25`: Tokenizes candidate text.
- `L26`: Computes overlap ratio.
- `L27`: Blends base score and overlap score (70/30).
- `L28`: Stores rescored tuple.
- `L29`: Sorts by new score descending.
- `L30`: Returns top-k reranked.

---

## `2_src/generation/text_generator.py`

- `L1`: Enables postponed annotations.
- `L3`: Imports typing helpers.
- `L6`: Declares text generation helper class.
- `L7-L10`: Docstring clarifies this is a simple placeholder generator.
- `L12`: `generate` method signature.
- `L13`: Checks if contexts are empty.
- `L14`: Returns fallback message when no evidence exists.
- `L15`: Initializes snippet list.
- `L16`: Loops through up to 3 contexts.
- `L17`: Creates short preview from first 45 words.
- `L18`: Formats snippet with citation-like chunk ID.
- `L19`: Joins snippets by newline.
- `L20-L25`: Returns formatted answer with question, evidence summary, and citation note.

---

## `2_src/generation/visual_generator.py`

- `L1`: Enables postponed annotations.
- `L3`: Imports `List`.
- `L6`: Declares visual generator class.
- `L7`: Class docstring marking placeholder status.
- `L9`: Method signature for figure-aware answer.
- `L10`: Checks if no figure IDs were found.
- `L11`: Returns no-evidence message.
- `L12`: Joins first 5 figure IDs.
- `L13`: Returns simple figure-path response.

---

## `2_src/evaluation/ragas_eval.py`

- `L1`: Enables postponed annotations.
- `L3`: Imports `math`.
- `L4`: Imports typing helpers.
- `L7`: `recall_at_k` signature.
- `L8`: Converts relevant IDs to set.
- `L9-L10`: Returns 0 if no relevant items given.
- `L11`: Counts relevant items found in top-k retrieved.
- `L12`: Divides by number of relevant items.
- `L15`: `mrr` signature.
- `L16`: Converts relevant IDs to set.
- `L17`: Iterates retrieved list with rank.
- `L18`: Checks first relevant hit.
- `L19`: Returns reciprocal rank of first hit.
- `L20`: Returns 0 when no relevant item found.
- `L23`: `ndcg_at_k` signature.
- `L24`: Converts relevant IDs to set.
- `L25`: Initializes DCG.
- `L26`: Iterates top-k retrieved with rank.
- `L27`: Checks relevance.
- `L28`: Adds discounted gain.
- `L29`: Computes ideal number of hits in top-k.
- `L30-L31`: Returns 0 when ideal is zero.
- `L32`: Initializes IDCG.
- `L33`: Iterates ideal ranks.
- `L34`: Adds ideal discounted gain.
- `L35`: Returns normalized DCG.

---

## `2_src/pipeline.py`

- `L1`: Enables postponed annotations.
- `L3`: Imports dataclass decorator.
- `L4`: Imports `Path`.
- `L5`: Imports typing helpers.
- `L7`: Imports pandas.
- `L9-L16`: Imports all pipeline component modules.
- `L19`: Declares `QueryResult` dataclass.
- `L21`: Stores query text.
- `L22`: Stores generated answer text.
- `L23`: Stores list of source chunk IDs.
- `L24`: Stores debug payload.
- `L27`: Declares `SciRetPipeline` class.
- `L28`: Class docstring: end-to-end Tier 1 baseline.
- `L30`: Constructor signature.
- `L31`: Resolves root path from argument or inferred project root.
- `L32`: Instantiates data loader.
- `L33`: Instantiates text embedder (hash baseline).
- `L34`: Instantiates BM25 retriever.
- `L35`: Instantiates dense retriever.
- `L36`: Instantiates reranker.
- `L37`: Instantiates text generator.
- `L38`: Initializes chunk DataFrame cache.
- `L40`: Property decorator for chunk path getter.
- `L41-L42`: Returns canonical chunk parquet path.
- `L44`: Property for embedding parquet path.
- `L45-L46`: Returns embedding file path.
- `L48`: Property for embedding manifest path.
- `L49-L50`: Returns manifest file path.
- `L52`: `build_from_metadata` method signature.
- `L53`: Loads full metadata.
- `L54`: Builds filtered tier subset.
- `L55`: Saves cleaned subset.
- `L56`: Builds chunk DataFrame.
- `L57`: Ensures chunk output folder exists.
- `L58`: Saves chunks parquet.
- `L59`: Builds retrieval indexes.
- `L61`: `load_index` method signature.
- `L62`: Checks for chunk file existence.
- `L63`: Raises clear error if missing.
- `L64`: Loads chunks parquet.
- `L65`: Fits indexes from loaded chunks.
- `L67`: Private `_fit_indexes` signature.
- `L68`: Resets chunk DataFrame with clean index.
- `L69`: Fits BM25 on chunk IDs/text.
- `L70`: Loads/builds embedding cache.
- `L71`: Fits dense retriever matrix.
- `L73`: Query method signature.
- `L74`: Guard for uninitialized pipeline.
- `L75`: Raises usage error if index not loaded.
- `L77`: Runs dense retrieval.
- `L78`: Runs BM25 retrieval.
- `L79`: Fuses both runs by RRF.
- `L81`: Creates `chunk_id -> text` lookup.
- `L82`: Converts fused hits into `(id, text, base_score)` tuples.
- `L83`: Reranks candidates and keeps final top-k.
- `L85`: Creates generator context tuples.
- `L86`: Generates grounded answer.
- `L87`: Extracts source IDs.
- `L88`: Builds debug payload with top slices.
- `L89`: Returns structured `QueryResult`.

---

## `3_notebooks/01_data_exploration.ipynb`

### Code Cell 1 (line-by-line)

1. `from pathlib import Path`: import path handling.
2. `import json`: import JSON writing utility.
3. `import pandas as pd`: import pandas.
4. blank: readability.
5. `ROOT = Path('..').resolve()`: set repo root from notebook folder.
6. `RAW = ...metadata.csv`: define input metadata path.
7. `OUT_DIR = ...processed`: define processed output folder.
8. `OUT_DIR.mkdir(...)`: create folder if missing.
9. blank.
10. `TIER_NAME = 'tier1'`: label current run tier.
11. `TIER_SIZE = 1000`: target sample size.
12. `SEED = 42`: fixed seed for reproducibility.
13. blank.
14. `assert RAW.exists()...`: fail early if metadata missing.
15. `df = pd.read_csv(...)`: load metadata.
16. `drop_duplicates(...)`: keep unique papers by `cord_uid`.
17. `df = df[df['abstract']...]`: keep rows with abstract length > 100.
18. `sample = df.sample(...)`: deterministic random sample to tier size.
19. blank.
20. `clean_path = ...papers_clean.parquet`: output path.
21. `sample.to_parquet(...)`: save cleaned subset.
22. blank.
23-30. `manifest = {...}`: build metadata manifest for reproducibility.
31. write manifest JSON to processed folder.
32. print success message with paper count and path.

---

## `3_notebooks/02_text_chunking.ipynb`

### Code Cell 1 (line-by-line)

1-3. import `Path`, `json`, `pandas`.
5. define repo root.
6. define input cleaned parquet path.
7. define output folder path.
9. set chunk size.
10. set overlap size.
11. compute sliding step (`chunk_size - overlap`, protected minimum 1).
13. assert cleaned input exists.
14. load cleaned papers.
15. initialize list for chunk records.
17. iterate papers.
18. combine title and abstract into one text.
19. split text into words.
20-21. skip empty texts.
22. iterate sliding windows over text.
23. take one chunk window.
24-25. skip very short chunks (<20 words).
26. compute chunk index.
27-33. append chunk record dict (`chunk_id`, metadata, chunk text).
35. build chunk dataframe from collected records.
36. define chunk output path.
37. save chunks parquet.
38. save chunk config JSON.
39. print success summary.

---

## `3_notebooks/03_embedding_baseline.ipynb`

### Code Cell 1 (line-by-line)

1-3. import `Path`, `json`, `pandas`.
5. repo root.
6. input chunks path.
7. output embedding folder.
8. ensure output folder exists.
9. output embedding parquet path.
10. embedding manifest JSON path.
11. record intended model name (`BAAI/bge-m3`).
13. cache check (`embeddings + manifest` both exist).
14. print cache-hit message and skip compute if present.
15. else branch begins.
16. assert chunks file exists.
17. load chunks.
18. comment: placeholder until real embedding integration.
19. keep `chunk_id` as key.
20. set embedding dimension metadata.
21. set placeholder vector marker.
22. save placeholder embeddings.
23. save embedding manifest (model + count).
24. print write confirmation.

---

## `3_notebooks/04_hybrid_retrieval.ipynb`

### Code Cell 1 (line-by-line)

1-2. import `Path` and `pandas`.
4. set repo root.
5. define chunks input path.
6. define embeddings input path.
7. define retrieval result folder.
8. create retrieval folder.
10. assert chunk file exists.
11. assert embedding file exists.
13. load chunks.
14. load embeddings.
15. print loaded counts.
16. print TODO message for BM25 + dense + RRF wiring.

---

## `3_notebooks/05_reranking.ipynb`

### Code Cell 1 (line-by-line)

1-2. import `Path`, `pandas`.
4. set repo root.
5. define retrieval input directory.
6. define retrieval output directory (same path currently).
7. ensure output directory exists.
9. print TODO for reranker integration.
10. print whether input folder currently exists.

---

## `3_notebooks/06_figure_extraction.ipynb`

### Code Cell 1 (line-by-line)

1-2. import `Path`, `pandas`.
4. set repo root.
5. define figures manifest output path.
6. ensure parent folder exists.
8. initialize empty DataFrame with expected figure metadata columns.
9. save empty manifest parquet.
10. print initialized output location.
11. print TODO for real PDF extraction integration.

---

## `3_notebooks/07_clip_embeddings.ipynb`

### Code Cell 1 (line-by-line)

1-2. import `Path`, `pandas`.
4. set repo root.
5. define figure manifest input path.
6. define figure embeddings output path.
7. ensure output folder exists.
9. assert figure manifest exists.
10. load figure manifest.
12. initialize embedding table with `figure_id`.
13. assign embedding dim metadata (512).
14. assign placeholder vector marker.
15. save figure embedding parquet.
16. print output row count and path.

---

## `3_notebooks/08_multimodal_pipeline.ipynb`

### Code Cell 1 (line-by-line)

1-2. import `Path`, `json`.
4. set repo root.
5. define multimodal output folder.
6. ensure folder exists.
8-14. build sample trace dictionary structure for multimodal routing output.
15. write trace JSON file.
16. print initialized location.

---

## `3_notebooks/09_evaluation.ipynb`

### Code Cell 1 (line-by-line)

1. import `Path`.
3. set repo root.
4. define results output directory.
5. ensure output directory exists.
6. define markdown table output path.
8-12. define baseline comparison table template as markdown text.
13. write markdown table file.
14. print output path.

---

## How to Read This Codebase in the Right Order

1. `2_src/pipeline.py` (overall flow)
2. `2_src/data/loader.py` and `2_src/data/chunker.py` (data prep)
3. `2_src/embeddings/text_embedder.py` (cache/build embeddings)
4. `2_src/retrieval/*` (search + fusion + rerank)
5. `2_src/generation/*` (answer formatting)
6. `2_src/evaluation/ragas_eval.py` (metrics)
7. `3_notebooks/01` to `09` (execution staging)

---

## Important Clarification

Several modules/notebooks currently use **placeholder logic** by design (`vector_stub`, TODO reranker integration, TODO PDF parsing).  
This is to keep the pipeline runnable and teachable first, then progressively replace stubs with:

- real BGE-M3 embedding generation,
- real BM25 + dense + RRF experiment persistence,
- cross-encoder reranker,
- CLIP and figure extraction stack,
- full evaluation dataset and report outputs.
