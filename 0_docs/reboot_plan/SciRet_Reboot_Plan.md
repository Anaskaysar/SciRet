# SciRet — Step-by-Step Reboot Plan

Status: active
Last updated: 2026-08-04
Target: October 2026 ARR cycle, submission deadline **Oct 12, 2026** → aiming to commit to NAACL 2027 / COLING 2027
Track: **Long paper, 8-page main body** (confirmed 2026-08-04) — limitations (mandatory) and ethics (optional) go after the conclusion, before references, and don't count against the 8 pages; references and appendix are unlimited.
Team: Kaysarul Anas Apurba (lead, corresponding), Md. Hasibul Hasan, Rofiqul Alam Shehab, Asab Azad
Companion doc: [SciRet_Research_Approach.md](SciRet_Research_Approach.md) (full detail on *why* each step exists)
Log: [SciRet_Reboot_Log.md](SciRet_Reboot_Log.md) (append an entry every time a step below is completed or changed)

Check off items as they're done and add a dated log entry when you do. Keep this file itself unchanged in structure — status lives in the checkboxes, history lives in the log.

**Ownership tags below are a draft proposal, not confirmed** — to be discussed and adjusted at the 2026-08-05 kickoff meeting based on actual availability/interest, then this line removed once confirmed.

## Page Budget (main body, 8-page cap)

Measured from the compiled PDF as of 2026-08-04 (current usage) against a target allocation that leaves room for Phases 1–5. Limitations/Ethics/Acknowledgments/References/Appendix are excluded from the cap entirely — push detail tables there first before spending main-body pages.

| Section | Current | Budgeted | Driven by |
|---|---|---|---|
| Front matter + Abstract | 0.4 | 0.4 | fixed; abstract gets a light update once new numbers land |
| §1 Introduction (incl. Figure 1) | 0.9 | 0.9 | done, no further growth planned |
| §2 Related Work | 0.7 | 0.8 | small optional expansion, long papers read as more thorough here |
| §3 Method | 1.5 | 2.1 | + second reranker description (Phase 2), + stats protocol paragraph (Phase 4), + TREC-COVID protocol (Phase 5) |
| §4 Results | 1.0 | 2.3 | + second reranker table (Phase 2), + recall@K table (Phase 3), + significance markers/effect sizes (Phase 4), + TREC-COVID agreement table (Phase 5) |
| §5 Discussion | 0.35 | 0.6 | synthesize the new findings above |
| §6 Conclusion | 0.35 | 0.35 | unchanged, numbers only |
| **Total** | **~5.2** | **~7.45** | leaves ~0.55 page buffer under the 8-page cap |

Rule of thumb while doing Phases 1–5: full statistical detail (per-query p-values, bootstrap CIs) belongs in the Appendix, not the main body — Results should only carry significance markers (*, **) and one-line effect-size call-outs. Same for the TREC-COVID overlap check: the qrel-matching methodology detail can live in the Appendix, with just the headline agreement number in Results/Discussion.

## Phase 0 — Housekeeping

- [x] Consolidate all reboot-plan docs into this plan + the research approach doc + the log (2026-07-31)
- [x] Restore real author names in `SciRet_ARR_2026/main.tex` for the working/preprint copy (2026-07-31)
- [x] Target venue decided: October 2026 ARR cycle → NAACL 2027 / COLING 2027 (2026-08-01)
- [x] Deliberately did NOT commit to EMNLP 2026 by the Aug 2, 2026 deadline — preserves ARR resubmission eligibility (2026-08-01)
- [x] KB: confirm CORD-19 version/filtering used (2026-08-01, via `metadata.readme` upload)
- [x] Code/index release repo URL updated in `main.tex` to `https://github.com/anaskaysar/sciret` (2026-08-04) — **confirm this repo is public and actually contains code/indexes before Phase 7 cites it**
- [x] KB: confirm third co-author name, order, affiliation (2026-08-01) — Rofiqul Alam Shehab, North South University, third author
- [x] Fourth co-author added: Asab Azad, Laurentian University (2026-08-04)
- [ ] Confirm work assignments below with the full team (kickoff meeting 2026-08-05)

## Phase 1 — Expand the query set (biggest lift, do first)

**Proposed owner: Asab Azad** (review/vetting task, good entry point; confirm interest 2026-08-05)

**Update 2026-08-05 — the 50-query set already exists, drafting is NOT needed.** `1_data/eval/queries.json` and `3_notebooks/General/03_query_set.ipynb` already contain a stratified 50-query set matching the rebuttal's exact 6-theme commitment: Imaging/Visual (10), Molecular/Mechanistic (10), Clinical Outcomes (10), Treatment/Intervention (10), Dataset/Methodology (5), Cross-domain Synthesis (5). The notebook explicitly says "all scale experiments load this file," so it's live pipeline infrastructure, not archived leftovers. It does **not** overlap with the 15 queries actually used in the current `main.tex` results (different wording, drafted separately) — the two sets need reconciling, not merging query-by-query. **Recommendation: adopt this 50-query set wholesale as the Phase 1 evaluation set**, retiring the original ad-hoc 15, since it already matches what was promised in the rebuttal more precisely than drafting new queries would. Azad's task changes from "draft ~35 new queries" to a lighter review/vetting pass:

- [ ] Review each of the 50 queries: answerable from CORD-19-style titles/abstracts (not asking for info the corpus can't have)? Not a near-duplicate of another query in the set?
- [ ] Flag phrasing that might unfairly favor keyword-style (BM25) or paraphrase-style (dense) retrieval
- [ ] Sanity-check the 10/10/10/10/5/5 theme split — flag if the two 5-query themes (Methodology, Cross-domain Synthesis) feel too thin relative to the other four
- [ ] Confirm final go/no-go on retiring the original 15 in favor of this 50-query set
- [ ] Generate pseudo-relevance labels for the 50 queries using the same method as the original 15 (15K-scale hybrid top-3, held fixed across scales) — technical step, not Azad's
- [ ] Re-run retrieval (BM25, dense, hybrid) for all 50 queries at 1K/5K/15K scales — technical step, not Azad's
- [ ] Re-run generation + RAGAS for all 50 queries at all scales — technical step, not Azad's
- [ ] Store per-query score files (needed for Phase 3 bootstrap CIs)

**Bonus finding, relevant to Phase 5 / Rofiqul's role:** `03_query_set.ipynb` also already scopes a manual ground-truth annotation task — 50 queries × 3 passages = 150 binary relevance judgments, estimated ~2-3 hours — as the intended fix for pseudo-label circularity (stronger than the TREC-COVID anchor check alone, and directly matches the "human verification" role discussed when Rofiqul joined). The annotation CSV (`4_results/tier2/passages_for_annotation.csv`) referenced by the notebook doesn't exist yet — someone technical needs to generate candidate passages per query first before this can be handed to Rofiqul. Worth raising alongside Phase 5 at the next sync.

Decision gate: retrieval and generation results exist for the full 50-query set at all three scales, stored in a form that supports per-query statistical analysis.

## Phase 1B — Full-text indexing pilot at 1K scale (added 2026-08-05)

**Owner: Kaysarul Anas Apurba (KB)** — starting now, in parallel with Phase 1. Not gated by the 50-query expansion; it's a separate axis (evidence depth vs. corpus scale).

**Why this phase exists:** the paper currently indexes only titles and abstracts and disclosed this as a Limitation rather than attempting to fix it. KB's call (2026-08-05): stop defaulting to disclosure for self-imposed scope choices, attempt to solve them instead. See "Working principle" in `SciRet_Research_Approach.md` Section 1b.

**Reality check before starting** (confirmed by reading `2_src/`): this is a real build, not a config change. `chunker.py` hardcodes `title + abstract` with no full-text field in the path at all. `pdf_parser.py` is an unimplemented stub. Recommend using CORD-19's own pre-parsed `document_parses/pdf_json/*.json` / `pmc_json/*.json` (`body_text` field) rather than writing a PDF parser from scratch — much smaller lift.

**Code written 2026-08-05: `3_notebooks/sciret_pipeline.ipynb`.** Single standalone notebook (replaces the old `scale_1K`…`scale_100K` folder-per-scale structure, now archived at `6_legacy/notebooks_scale_1K_to_100K_archived_2026-08-05/`) that implements the checklist below end to end, with `RUN_FULLTEXT` and `REQUIRE_FULLTEXT_FOR_SAMPLE` toggles in its config cell. Not yet run — needs Kaggle/local GPU + CORD-19 data + `OPENAI_API_KEY`.

- [x] Check full-text coverage in the metadata before sampling — implemented in the notebook, auto-detects either the 2022-06-02 release's `pdf_json_files`/`pmc_json_files` path columns or older `has_pdf_parse`/`has_pmc_xml_parse` boolean columns
- [x] Add a full-text loader (consume `document_parses` JSON body text) alongside the existing title+abstract loader — both conditions run side by side in the notebook, abstract-only path untouched
- [x] Chunk full body text using the same sentence-window strategy already used for abstracts — implemented, shared `sentence_window_chunk()` function
- [ ] **Run** the full pipeline at 1K scale: BM25 + BGE-M3 indexing, hybrid fusion, MS MARCO reranker (the new scientific reranker from Phase 2 can be swapped in once it lands), generation, RAGAS — code is ready, execution is not done
- [ ] Compare full-text 1K results against the existing abstract-only 1K results — the notebook prints/saves this comparison automatically once run

**Decision gate:** does full-text indexing at 1K meaningfully change the story (better recall/precision, different reranker behavior, resolves the evidence-depth limitation credibly)?
- **If yes and compute allows:** extend to 5K and 15K to preserve the multi-scale narrative on full text; the original abstract-only 1K/5K/15K results move to the Appendix as the development/ablation trail ("we validated the pipeline on title+abstract indexing before testing full-text depth").
- **If yes but compute/time doesn't allow scaling past 1K:** keep the multi-scale (1K/5K/15K) scale study on title+abstract as the primary Results (comparable across scales), and add the full-text-vs-abstract comparison as a separate 1K-only depth ablation — genuinely on-theme for a "compute-aware" paper (scale vs. depth tradeoff under a fixed compute budget), not a downgrade.
- **If no (full-text doesn't change the story or isn't ready in time):** keep the current abstract-only results as primary, but narrow the Limitations sentence to reflect that full-text indexing was actually tested at 1K, not just disclosed as untested — a materially stronger limitation than what's in the paper now either way.

Given the Oct 12 deadline and the team's prior stated Kaggle compute-time bottleneck (full-text bodies are typically 5–20x longer than abstracts to embed), do not commit to full-text at all three scales until the 1K pilot proves it's worth the cost.

## Phase 2 — Reranker baseline

**Proposed owner: Kaysarul Anas Apurba** (biggest technical lift, ML/engineering-heavy)

- [ ] Short model search: identify ≥1 scientific/biomedical cross-encoder reranker candidate (e.g. a SciBERT/PubMedBERT-based cross-encoder, or a BGE reranker fine-tuned on biomedical pairs)
- [ ] Run the new reranker on the same retrieval outputs used for the MS MARCO reranker comparison
- [ ] Report precision@K and recall@K (see Phase 3) for: no-rerank, MS MARCO reranker, new scientific reranker
- [ ] Update Table `tab:rerank` (or add a new table) in `main.tex`

Decision gate: paper can state whether the MS MARCO degradation was reranker-specific or a general domain-mismatch effect.

## Phase 3 — Recall@K for reranking

**Proposed owner: Md. Hasibul Hasan** (pairs naturally with Phase 4, same data)

- [ ] Compute recall@K before/after reranking from stored run outputs (no new inference needed — should be quick)
- [ ] Add recall@K columns/table alongside the existing precision@K table

Decision gate: reranking analysis reports both precision and recall, as requested by R2.

## Phase 4 — Statistical rigor

**Proposed owner: Md. Hasibul Hasan** (same person as Phase 3 — one coherent stats workstream)

Protocol (adopted from the original April 2026 working paper, `Multimodal_Retrieval_Augmented_Systems_for_Scientific_Knowledge_Access.pdf`, which had already spec'd this in more detail than "just bootstrap CIs"):

- [ ] Run Shapiro-Wilk normality test on per-query score distributions first
- [ ] Where normal: paired t-test (two-tailed, α = 0.05); where non-normal: Wilcoxon signed-rank test
- [ ] Report effect sizes as Cohen's d
- [ ] Report 95% confidence intervals via bootstrap
- [ ] Apply this to the retrieval comparisons in Table `tab:recall` and the reranking precision/recall comparisons from Phase 2/3
- [ ] Document the exact procedure (resampling count, seed) so it's reproducible
- [ ] Add significance markers/CIs to the relevant tables or a supplementary table

Decision gate: every headline retrieval/reranking claim in the paper has an associated significance test result, not just a raw number.

## Phase 5 — TREC-COVID qrel anchor check

**Proposed owner: Rofiqul Alam Shehab** (matches the "human verification" role discussed when he joined — validating labels against an external source)

- [ ] Pull TREC-COVID qrels
- [ ] Match our 50 queries against TREC-COVID topics (exact match or near-duplicate; note how many overlap)
- [ ] For the overlapping subset, compare our pseudo-relevance labels against the official qrels (agreement rate, or recall of our labels against qrels)
- [ ] Write up the result as an external validity check for the label circularity limitation

Decision gate: paper can cite an external anchor point for the pseudo-label methodology, not just an internal disclosure.

## Phase 6 — Writing-only fixes (no new experiments)

- [x] Section 3.4: add the label-construction explanation (15K-scale labels held fixed; smaller corpora don't contain all 3 labeled docs, hence hybrid R@3 < 1.0 at 1K/5K) (2026-08-01)
- [x] Add R@1 denominator explanation (2026-08-01)
- [x] Add chunk-token stability explanation near Table `tab:dataset` (2026-08-01)
- [x] Correct the context-precision explanation in Results to the citation-based framing (2026-08-01)
- [x] Add RAGAS metric definitions + LLM-judge bias discussion (2026-08-01)
- [x] Expand Limitations: generalizability beyond CORD-19/COVID-19 untested; CORD-19's English-language/journal skew; equity; broader societal-impact/misinformation risk (2026-08-01)
- [x] Make "short paper track, controlled empirical comparison by design" framing explicit in the Introduction (2026-08-01)
- [x] CORD-19 version/filtering sentence added to Section 3.1 (2026-08-01) — resolved from uploaded `metadata.readme` (release 2022-06-02, 1,056,660 rows, final release before discontinuation) + `2_src/data/loader.py` (no journal/peer-review/language filter; dedup by `cord_uid`, min. 100-char abstract, random sample with fixed seed)

Decision gate: every writing commitment made in the rebuttal (`SciRet_Author_Replies_v2.docx`) is reflected in `main.tex`. **Phase 6 fully complete as of 2026-08-01** — verified by recompiling `main.tex` with `latexmk`/`pdflatex` and reading the rendered PDF text; compiles clean, all cross-references resolve.

## Phase 7 — Reproducibility section

**Proposed owner: Kaysarul Anas Apurba** (owns the repo/release; pairs with Phase 2)

- [ ] Blocked on KB confirming the repo URL / index hosting location (Phase 0)
- [ ] Once confirmed: add a dedicated Reproducibility section with direct links to the CORD-19 source (github.com/allenai/cord19), the SciRet code repo, and pre-built index artifacts

Decision gate: Reproducibility/Datasets/Software concerns from R3 (scored 2/1/1) are directly addressed with working links.

## Phase 8 — Final pass

**Owner: whole team, coordinated by Kaysarul Anas Apurba**

- [ ] Re-verify typo/citation fixes are still correct after all edits above (they were confirmed done as of 2026-07-31 — re-check nothing regressed)
- [ ] Full read-through against the rebuttal to confirm every commitment was kept
- [ ] Recompile `main.tex`, check for overfull boxes / broken references after new tables
- [ ] Update abstract if headline numbers changed (50-query results may differ from the original 15-query numbers)
- [ ] Generate an anonymized ARR-submission copy from the working `main.tex`: revert author block to "Anonymous ACL submission," remove the "Preprint — Work in Progress" header, revert Acknowledgments to "Omitted for anonymous review." Keep the de-anonymized version as the separate working/preprint copy — don't toggle one file back and forth.
- [ ] Submit to ARR October 2026 cycle by Oct 12, 2026

## Stretch / optional (not committed to reviewers — do only if time allows, in this order)

- [ ] Second scientific-domain dataset for generalization
- [ ] Small human evaluation of generation quality — natural fit for Rofiqul Alam Shehab given the human-verification role
- [ ] 2-3 qualitative failure-case examples
- [ ] Extend chunking analysis across all scales (currently 1K only)
