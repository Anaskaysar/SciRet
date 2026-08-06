# SciRet Notebooks — Restructured 2026-08-05

## How This Works

One notebook, `sciret_pipeline.ipynb`, runs the entire pipeline end to end: sample → chunk →
embed (BGE-M3 + BM25) → retrieve (dense/BM25/hybrid) → rerank (MS MARCO cross-encoder) →
generate (GPT-4o-mini) → evaluate (RAGAS). Change `N_PAPERS` and `SCALE_LABEL` in the config
cell at the top to rerun at a different scale — nothing else should need to change.

It also runs the Phase 1B full-text-vs-abstract-only pilot (see `SciRet_Reboot_Plan.md`):
set `RUN_FULLTEXT = True` to build both indexing conditions side by side in the same run and
get a direct comparison table at the end, instead of writing a separate notebook per condition.

## Folder Map

```
General/                    ← Run ONCE. Shared across all runs.
  00_environment.ipynb        Environment check, package versions
  01_dataset_stats.ipynb      Full CORD-19 corpus statistics
  02_chunking_strategy.ipynb  4-strategy chunking comparison (done once)
  03_query_set.ipynb          Defines the 50-query stratified eval set (1_data/eval/queries.json)
  04_cross_tier_analysis.ipynb  Fit scale curves once multiple scales have been run

sciret_pipeline.ipynb       ← THE pipeline. Parameterized by N_PAPERS/SCALE_LABEL in its config cell.
kaggle/                     ← Earlier one-off Kaggle exports (15K/30K/50K) — kept for reference,
                               not the current pattern; superseded by sciret_pipeline.ipynb going forward.
```

Results from each run land in `4_results/<SCALE_LABEL>/`, with `_abstract` / `_fulltext`
suffixes on filenames when the full-text pilot is on.

## What changed 2026-08-05

The old structure was one folder per scale (`scale_1K/` … `scale_100K/`), each with 5 separate
notebooks (`01_sample_chunk` → `05_generation_ragas`) that had to be run in sequence and kept in
sync by hand. That's now consolidated into the single `sciret_pipeline.ipynb`. The old folders
are archived, not deleted, at `6_legacy/notebooks_scale_1K_to_100K_archived_2026-08-05/` — they
contain real run history (including saved outputs) worth keeping for reference. See the Reboot
Log entry for 2026-08-05 for why, and for an important finding surfaced while archiving them.

## Key Code Conventions (carried over from the old structure)

1. `DEVICE` is always resolved once at the top (`'cuda' if torch.cuda.is_available() else 'cpu'`)
2. Ground truth loads from `1_data/eval/ground_truth.json` if it exists; otherwise falls back to
   interim pseudo-labels built from this run's own hybrid top-3 — documented as pseudo, not silently
   treated as real ground truth
3. `RANDOM_SEED = 42` — never change this, it's what keeps runs comparable across scales
4. RRF k=60, reported explicitly wherever it's used
5. Recall@K and Precision@K are computed against the (fixed, or documented-pseudo) ground truth
   dict, never derived from the system being evaluated

## Legacy

Older archived notebooks (tier_1, tier_2, Alamin) are at
`6_legacy/notebooks_archived_2026-05-04/`. The scale_1K–100K structure superseded 2026-08-05 is at
`6_legacy/notebooks_scale_1K_to_100K_archived_2026-08-05/`.
