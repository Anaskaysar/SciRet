# SciRet — Step-by-Step Reboot Plan

Status: active
Last updated: 2026-07-31
Companion doc: [SciRet_Research_Approach.md](SciRet_Research_Approach.md) (full detail on *why* each step exists)
Log: [SciRet_Reboot_Log.md](SciRet_Reboot_Log.md) (append an entry every time a step below is completed or changed)

Check off items as they're done and add a dated log entry when you do. Keep this file itself unchanged in structure — status lives in the checkboxes, history lives in the log.

## Phase 0 — Housekeeping (today)

- [x] Consolidate all reboot-plan docs into this plan + the research approach doc + the log (2026-07-31)
- [x] Restore real author names in `SciRet_ARR_2026/main.tex` (2026-07-31)
- [ ] KB: confirm CORD-19 version/filtering used
- [ ] KB: confirm code/index release repo URL
- [ ] KB: confirm third co-author name, order, affiliation
- [ ] KB: confirm target venue (affects whether the paper needs to stay de-anonymized or be re-blinded before submission)

## Phase 1 — Expand the query set (biggest lift, do first)

- [ ] Draft ~35 new queries across the 6 committed themes: imaging, molecular/mechanistic, clinical outcomes, treatment, methodology, cross-domain synthesis (aim for a balanced ~8/theme across 44, or match whatever split makes 50 total with the original 15)
- [ ] Review new queries for phrasing skew (avoid accidentally favoring sparse or dense retrieval)
- [ ] Generate pseudo-relevance labels for the new queries using the same method as the original 15 (15K-scale hybrid top-3, held fixed across scales)
- [ ] Re-run retrieval (BM25, dense, hybrid) for all 50 queries at 1K/5K/15K scales
- [ ] Re-run generation + RAGAS for all 50 queries at all scales
- [ ] Store per-query score files (needed for Phase 3 bootstrap CIs)

Decision gate: retrieval and generation results exist for the full 50-query set at all three scales, stored in a form that supports per-query statistical analysis.

## Phase 2 — Reranker baseline

- [ ] Short model search: identify ≥1 scientific/biomedical cross-encoder reranker candidate (e.g. a SciBERT/PubMedBERT-based cross-encoder, or a BGE reranker fine-tuned on biomedical pairs)
- [ ] Run the new reranker on the same retrieval outputs used for the MS MARCO reranker comparison
- [ ] Report precision@K and recall@K (see Phase 3) for: no-rerank, MS MARCO reranker, new scientific reranker
- [ ] Update Table `tab:rerank` (or add a new table) in `main.tex`

Decision gate: paper can state whether the MS MARCO degradation was reranker-specific or a general domain-mismatch effect.

## Phase 3 — Recall@K for reranking

- [ ] Compute recall@K before/after reranking from stored run outputs (no new inference needed — should be quick)
- [ ] Add recall@K columns/table alongside the existing precision@K table

Decision gate: reranking analysis reports both precision and recall, as requested by R2.

## Phase 4 — Statistical rigor

- [ ] Compute paired bootstrap confidence intervals (or paired bootstrap significance test) for the retrieval comparisons in Table `tab:recall`
- [ ] Compute the same for the reranking precision/recall comparisons in Phase 2/3
- [ ] Decide on and document the bootstrap method (resampling procedure, number of resamples, CI level) so it's reproducible
- [ ] Add CIs to the relevant tables or a supplementary table

Decision gate: every headline retrieval/reranking claim in the paper has an associated CI or significance statement.

## Phase 5 — TREC-COVID qrel anchor check

- [ ] Pull TREC-COVID qrels
- [ ] Match our 50 queries against TREC-COVID topics (exact match or near-duplicate; note how many overlap)
- [ ] For the overlapping subset, compare our pseudo-relevance labels against the official qrels (agreement rate, or recall of our labels against qrels)
- [ ] Write up the result as an external validity check for the label circularity limitation

Decision gate: paper can cite an external anchor point for the pseudo-label methodology, not just an internal disclosure.

## Phase 6 — Writing-only fixes (no new experiments)

- [ ] Section 3.4: add the label-construction explanation (15K-scale labels held fixed; smaller corpora don't contain all 3 labeled docs, hence hybrid R@3 < 1.0 at 1K/5K)
- [ ] Add R@1 denominator explanation (reuse rebuttal text — relative signal within our labeling protocol, not an absolute benchmark)
- [ ] Add chunk-token stability explanation near Table `tab:dataset` (same chunking params applied regardless of sample size)
- [ ] Correct the context-precision explanation in Results to the citation-based framing (fraction of retrieved passages the generator actually cites — not the current "topic-related but not tightly targeted" wording)
- [ ] Add RAGAS metric definitions + LLM-judge bias discussion
- [ ] Expand Limitations: generalizability beyond CORD-19/COVID-19 untested; CORD-19's English-language/journal skew; equity (who's served, who's excluded); broader societal-impact/misinformation risk if used without expert oversight
- [ ] Make "short paper track, controlled empirical comparison by design" framing explicit in the Introduction
- [ ] Once KB confirms CORD-19 version/filtering (Phase 0): add that sentence to Section 3.1

Decision gate: every writing commitment made in the rebuttal (`SciRet_Author_Replies_v2.docx`) is reflected in `main.tex`.

## Phase 7 — Reproducibility section

- [ ] Blocked on KB confirming the repo URL / index hosting location (Phase 0)
- [ ] Once confirmed: add a dedicated Reproducibility section with direct links to the CORD-19 source (github.com/allenai/cord19), the SciRet code repo, and pre-built index artifacts

Decision gate: Reproducibility/Datasets/Software concerns from R3 (scored 2/1/1) are directly addressed with working links.

## Phase 8 — Final pass

- [ ] Re-verify typo/citation fixes are still correct after all edits above (they were confirmed done as of 2026-07-31 — re-check nothing regressed)
- [ ] Full read-through against the rebuttal to confirm every commitment was kept
- [ ] Recompile `main.tex`, check for overfull boxes / broken references after new tables
- [ ] Update abstract if headline numbers changed (50-query results may differ from the original 15-query numbers)
- [ ] Final author-block check: confirmed names present, or re-anonymized if the target venue requires blind review (Phase 0 open question)

## Stretch / optional (not committed to reviewers — do only if time allows, in this order)

- [ ] Second scientific-domain dataset for generalization
- [ ] Small human evaluation of generation quality
- [ ] 2-3 qualitative failure-case examples
- [ ] Extend chunking analysis across all scales (currently 1K only)
