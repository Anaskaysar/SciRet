# SciRet — Research Approach for Next Submission

Status: active
Last updated: 2026-08-01
Supersedes: sciret_reboot_research_plan.adoc, SciRet_Reboot_Research_Plan.docx, SciRet_Reboot_Research_Plan_Enriched.docx, SciRet_Reboot_Research_Plan_Weekend_Kaggle.docx, SciRet_Revision_Checklist.md (all removed — see [SciRet_Reboot_Log.md](SciRet_Reboot_Log.md) 2026-07-31 entry)

## 1. Where we are

SciRet (Submission 8722, working title "SciRet: A Compute-Aware Empirical Study of Retrieval and Reranking for Scientific RAG") went through ARR review in early July 2026:

- R1 (5KS8): Soundness 4, Excitement 4, Overall 2.5 (Borderline Findings)
- R2 (SPUz): Soundness 2.5, Excitement 2.5, Overall 2.5 (Borderline Findings)
- R3 (VbPC): Soundness 2, Excitement 2, Overall 2 (Resubmit next cycle)

We submitted author replies (`SciRet_Author_Replies_v2.docx`) on 2026-07-10/11 committing to specific fixes per reviewer. Rebuttal is done. This document is the single source of truth for what "closing the gaps" means and how we're doing it. It replaces every earlier reboot-plan draft.

**Target venue (decided 2026-08-01):** revise and resubmit through the **October 2026 ARR cycle** (submission deadline **Oct 12, 2026**), aiming to commit to **NAACL 2027 / COLING 2027** (ARR commitment date Dec 20, 2026). KB's stated goal is at least one paper at a top NLP venue, which is why the earlier EMNLP-as-is / arXiv-fallback plan and ICNLP 2027 (a non-ACL regional conference, IEEE Xplore proceedings, deadline Nov 30, 2026 — considered and ruled out 2026-08-01) were both set aside in favor of doing the fuller revision and resubmitting within the ACL ecosystem. See Section 1a for what this means concretely.

### 1a. Consequences of the ARR-resubmission decision

- **EMNLP 2026 commitment (Aug 2, 2026) is being let lapse on purpose.** Not committing is what preserves eligibility to revise-and-resubmit in October — this is an action of deliberate inaction, not an oversight.
- **~10 weeks of runway** (2026-08-01 to 2026-10-12) for the Section 3 gap-closing work below.
- **Re-anonymization is now required before submission.** ARR is double-blind. `main.tex` currently carries real author names (Kaysarul Anas Apurba, Md. Hasibul Hasan, Laurentian University) and a "Preprint — Work in Progress" header, restored 2026-07-31 for preprint/working purposes. A separate anonymized copy must be prepared for the actual Oct 12 ARR submission — see Section 7.

### 1b. Working principle (added 2026-08-05)

KB's directive: when a gap is addressable within reasonable time/compute, **attempt to actually fix or test it before defaulting to a disclosed Limitations sentence.** The pattern up to now — hit an issue, soften the claim, write it into Limitations — was making the paper read as weaker than it needed to be, and disclosure alone already failed to satisfy reviewers once (R3: 2/2, "resubmit next cycle"). Going forward, treat "disclose as a limitation" as the fallback for two specific cases only: (a) the reviewers themselves asked for disclosure, not a new experiment (e.g., R2's generalization/equity comment in 3.4 below was explicitly "flag as missing discussion," not "run a new experiment"), or (b) the fix is genuinely out of reach this cycle (new cross-domain dataset, full human evaluation study). Everything else defaults to "try to solve it" first.

**First application: full-text indexing (Section 3.8).** The paper currently indexes only titles and abstracts and discloses this as a limitation. That was a self-imposed scope choice, not a reviewer requirement — exactly the kind of thing this principle says to attack instead of disclose.

## 2. Scope decision

Two directions existed in the old drafts:

1. **Scoped revision** — fix exactly what R1/R2/R3 raised: statistical rigor, an added reranker baseline, recall@K reporting, reproducibility, limitations writing, and an optional TREC-COVID anchor check.
2. **Full rebuild ("SciRet-Verify")** — multimodal evidence-unit retrieval, claim-level citation verification, a new codebase, and 5 new datasets (SciFact, QASPER, PubMedQA, SPIQA, TREC-COVID).

**Decision (2026-07-31): scoped revision.** It's what the actual reviewers asked for, it's achievable without a ground-up rebuild, and every commitment we made in the rebuttal already falls inside it. The full-rebuild ideas aren't discarded — they're archived in Section 8 as future-paper material, not part of this submission's plan.

## 3. Gap inventory, organized by angle

Each row: the gap, who raised it, what we already promised in the rebuttal (if anything), and what closing it actually requires.

### 3.1 Statistical rigor

| Gap | Raised by | Rebuttal commitment | What's needed |
|---|---|---|---|
| No significance testing on retrieval/reranking comparisons, especially given only 15 queries | R2, R3 | "bootstrap confidence intervals on the retrieval tables given the small query set" | Paired bootstrap CIs (or bootstrap significance test) on Recall@K (Table `tab:recall`) and Precision@K (Table `tab:rerank`) comparisons. Re-run from existing per-query score files — no new retrieval runs needed. |
| Query set too small (15) for statistical power | R1, R2, R3 | "expanding to a stratified 50-query set balanced across six themes (imaging, molecular/mechanistic, clinical outcomes, treatment, methodology, cross-domain synthesis)" | Real work: draft ~35 new queries across the 6 themes, label them the same way as the original 15 (see 3.3), and re-run retrieval/generation/RAGAS on the full 50-query set at all three scales. This is the single biggest lift in the plan. |

### 3.2 Baselines

| Gap | Raised by | Rebuttal commitment | What's needed |
|---|---|---|---|
| Single MS MARCO reranker — can't separate "domain mismatch" from "this reranker" | R2, R3 | "add at least one scientific/biomedical reranker in the revision" | Pick and run ≥1 scientific/biomedical cross-encoder (candidates: a PubMedBERT- or SciBERT-based reranker, or a BGE reranker fine-tuned on biomedical pairs — needs a short model search, see Plan). Report precision@K and recall@K for it alongside the existing MS MARCO reranker and the no-rerank baseline. |
| Reranking: only precision reported, not recall | R2 | "recall@K alongside precision@K ... will add it in the revision" | Compute recall@K before/after reranking from the already-stored run outputs. Cheap — no new inference. |
| Single generator (GPT-4o-mini only) | R2 | Scope explicitly to "in our setup"; flag other-LLM generalization as future work | Writing only: tighten scope language in abstract/intro (already partly done — verify), add explicit future-work sentence. No new generation runs required for this submission. |

### 3.3 Data / evaluation-label integrity

| Gap | Raised by | Rebuttal commitment | What's needed |
|---|---|---|---|
| Pseudo-relevance labels come from the same hybrid system being evaluated (circularity) | R1, R2 | "add a TREC-COVID qrel overlap check for a subset of queries in the revision" | Pull TREC-COVID qrels, find query overlap (exact or near-duplicate) with our 15/50-query set, and report agreement between our pseudo-labels and the official qrels for that subset. Moderate effort — addresses the core methodological critique directly. |
| Why isn't hybrid R@3 = 1.0 at 1K/5K if top-3 hybrid defines relevance? | R1 | Explained: labels are fixed from the 15K-scale hybrid results; smaller corpora don't contain all 3 labeled docs, so the effective relevant set can be < 3 | Writing only: add this explanation explicitly to Section 3.4 (Retrieval Evaluation Note). Not yet in the current `main.tex` — needs to go in. |
| R@1 denominator (max 1/3) — why still meaningful? | R2 | Explained in rebuttal (relative signal within our labeling protocol, not an absolute benchmark) | Writing only: add the explanation sentence to the paper (text is already drafted in the rebuttal reply — reuse it). |
| Mean chunk tokens (215) constant across all scales | R2 | Explained: same chunking parameters applied regardless of corpus size, so a stable mean across random samples is expected | Writing only: add clarifying sentence near Table `tab:dataset`. |
| CORD-19 version/filtering not specified | R2 | **Resolved 2026-08-01.** Release 2022-06-02 (final CORD-19 release before discontinuation, 1,056,660 metadata rows); no journal/peer-review/language filter — only dedup by `cord_uid` and a 100-char minimum abstract length, then random sample per scale (fixed seed) | **Done** — sentence added to Section 3.1 of `main.tex`, 2026-08-01. |
| 15-query sourcing, cross-scale overlap, topical skew | R1 | Partially answered: queries developed iteratively during pipeline dev, same 15 used at all 3 scales, no formal skew check was run | Covered by the 50-query expansion (3.1) — the new set is stratified by theme specifically so coverage is checkable. |

### 3.4 Generalization

| Gap | Raised by | Rebuttal commitment | What's needed |
|---|---|---|---|
| Single dataset (CORD-19 only) | R2 | Flag as untested generalization in Limitations | Writing: add an explicit sentence to Limitations. **Optional stretch (not committed):** test on a second scientific domain if time allows — out of scope for this pass, listed as future work. |
| No discussion of generalizability beyond COVID-19, data/language bias, equity, broader societal impact | R2 (Limitations & Societal Impact) | Not directly promised, but reviewer flagged as missing | Writing only: expand Limitations/Ethics with a few sentences on CORD-19's English-language/journal skew, who is and isn't served by the system, and misinformation/over-reliance risk if used without expert oversight. No new experiments. |

### 3.5 Evaluation methodology / generation quality

| Gap | Raised by | Rebuttal commitment | What's needed |
|---|---|---|---|
| RAGAS metrics not defined/discussed; no human eval or fact verification | R3 | "add short definitions plus a discussion of known LLM-judge biases"; note the ethics statement's medical-use warning as partial mitigation | Writing: add a short RAGAS metric-definition paragraph + LLM-judge bias discussion to the paper. Small human eval is explicitly optional/future work, not committed. |
| Context precision very low (0.095–0.122), unexplained | R2 | Explained: RAGAS context precision here is citation-based (fraction of retrieved passages the generator actually cites, not relevance) | Writing: the current paper text ("retrieved contexts include topic-related but not always tightly targeted passages") does **not** match the committed explanation — needs to be corrected to the citation-based framing from the rebuttal. |
| No failure-case / qualitative analysis | R2 | Not committed in rebuttal | Optional, valuable for a fuller paper: 2–3 qualitative examples. Not required for this pass — list as stretch goal. |
| Chunking analysis only at 1K scale, brief | R2 | Not committed | Optional/low priority — defer. |

### 3.6 Reproducibility & release

| Gap | Raised by | Rebuttal commitment | What's needed |
|---|---|---|---|
| Reproducibility 2, Datasets 1, Software 1 scores — release commitments unclear | R3 | "add a dedicated Reproducibility section with direct links to the dataset, code repository, and pre-built index artifacts" | Needs KB input: confirm the actual repo URL / hosting location for code + index artifacts before this section can be written with real links. CORD-19 source link (github.com/allenai/cord19) is already known. |

### 3.7 Framing / editorial

| Gap | Raised by | Status |
|---|---|---|
| Paper reads as short report, scope/depth mismatch for a conference paper | R2 | Writing: make the "controlled empirical comparison, short-paper track by design" framing explicit up front, per rebuttal. |
| Typos: abstract spacing, "each retriever," `text[[...]]` artifact, "and 1 others" citations | R2 | **Done** — verified against current `main.tex`. Nothing further needed. |

### 3.8 Evidence depth (self-identified, not reviewer-raised)

| Gap | Raised by | Previous resolution | New plan (2026-08-05) |
|---|---|---|---|
| Only titles + abstracts indexed, not full paper text | Self-identified (Method 3.1: "we treat this as a limitation rather than a complete scientific-document solution") | Disclosed in Limitations, not attempted | **Attempt to solve, starting at 1K scale.** KB is building a full-text-indexing pilot at the 1K scale first. See Phase 1B in `SciRet_Reboot_Plan.md` for the concrete steps and decision gate. |

**Engineering reality check (confirmed 2026-08-05 by reading the current code):** full-text indexing is not a config flag away — it doesn't exist yet in `2_src/`. `chunker.py`'s `build_chunks()` hardcodes `text = f"{title} {abstract}"`; there is no full-text field anywhere in the chunking path. `pdf_parser.py` is a literal stub (`extract_figure_manifest_stub`, docstring: "Placeholder parser that returns empty records. Replace with PyMuPDF/pdfplumber extraction in multimodal phase.") — it returns empty records for every file. Two real risks worth knowing before committing time to this:
1. **Coverage**: not every CORD-19 row has full text available (only papers with `has_pdf_parse` and/or `has_pmc_xml_parse` set in `metadata.csv` do). `build_tier_subset()` in `loader.py` needs a full-text-availability filter before sampling, or the "1K full-text" sample will silently include papers with no body text to index.
2. **Compute cost**: full paper bodies are typically 5–20x longer than an abstract. The existing Compute Budget appendix table shows abstract-only embedding already at 73 minutes for 15K papers; full-text embedding at the same scale could be substantially more, which matters given the team's prior Kaggle free-tier time budget was already the stated bottleneck for the original 5K–100K attempt.
3. **Lighter-weight extraction path**: CORD-19 ships pre-parsed body text as JSON (`document_parses/pdf_json/*.json` and `pmc_json/*.json`, each with a `body_text` list of paragraph objects) for papers that have it. This is a much smaller lift than writing a real PDF parser — recommend using the pre-parsed JSON instead of building out `pdf_parser.py`'s PyMuPDF/pdfplumber path, unless the pre-parsed coverage turns out to be insufficient.

## 4. Priority order (highest leverage first)

1. Stratified 50-query expansion (3.1) — biggest lift, unlocks everything downstream (stats, TREC-COVID check all depend on having the fuller query set first, or at minimum need to be run twice: once on 15, once on 50)
2. Scientific/biomedical reranker baseline (3.2) — most-requested addition, appears in 2 reviews
3. Recall@K for reranking (3.2) — cheap, directly requested
4. Bootstrap CIs / significance tests (3.1) — cheap once per-query scores exist for the 50-query set
5. TREC-COVID qrel anchor check (3.3) — moderate effort, addresses the core circularity critique directly
6. Writing-only fixes (3.3, 3.4, 3.5, 3.7) — label-construction note, R@1 explanation, chunk-token note, context-precision correction, generalization/equity Limitations expansion, RAGAS definitions, scope framing
7. Reproducibility section (3.6) — blocked on KB confirming release links
8. CORD-19 version sentence (3.3) — blocked on KB confirming version/filtering

Item 1 gates items 2–5 in practice: reranker comparison, recall@K, bootstrap CIs, and the TREC-COVID check should all be computed on the final (50-query) evaluation set rather than redone twice, unless we decide to ship an interim result on the original 15 first. See [SciRet_Reboot_Plan.md](SciRet_Reboot_Plan.md) for the phased sequencing.

**9. Full-text indexing pilot (3.8)** — KB working this independently, starting now, in parallel with items 1–8. Not gated by the 50-query expansion since it's a separate axis (evidence depth vs. corpus scale); see Phase 1B in the Reboot Plan. Decision gate there determines whether the paper's primary Results lead with full-text or title+abstract indexing, and how much of the original 1K/5K/15K abstract-only run moves into the Appendix as supporting evidence of the exploration.

## 5. Explicitly out of scope for this pass

- New pipeline architecture, multimodal evidence units, claim-level citation verifier, adaptive retrieval router — all part of the archived full-rebuild direction (Section 8).
- Second dataset / cross-domain generalization test — optional stretch only, not committed to reviewers.
- Human evaluation study — optional stretch only, not committed to reviewers.
- Qualitative failure-case analysis — optional stretch only.

## 6. Open questions needing KB's input

These block specific writing tasks and can't be resolved by re-reading existing files:

~~1. CORD-19 version/filtering~~ — **resolved 2026-08-01**, see Section 3.3.
2. **Code/index release location** — actual repo URL and where pre-built indexes are hosted? `main.tex` currently has a bare `https://github.com` placeholder in the author block. (blocks 3.6 Reproducibility section)
~~3. Third co-author~~ — **resolved 2026-08-01**: Rofiqul Alam Shehab, North South University. See Section 7.

~~4. Target venue~~ — **resolved 2026-08-01**: October 2026 ARR cycle (Oct 12 submission) → NAACL 2027 / COLING 2027. See Section 1a.

## 7. Authors

Restored 2026-07-31 (was "Anonymous ACL submission" for blind review), finalized 2026-08-01 with the third co-author, for the de-anonymized/preprint working copy of `main.tex`:

- Kaysarul Anas Apurba$^1$ — Laurentian University — **corresponding author**, kaysarulanas2@gmail.com
- Md. Hasibul Hasan$^1$ — Laurentian University
- Rofiqul Alam Shehab$^2$ — North South University

*(Correction 2026-08-01: earlier versions of this doc/log incorrectly listed Md. Hasibul Hasan as corresponding author. The `main.tex` source has always had Kaysarul Anas Apurba as corresponding — that was a transcription error on my part, not a change to the file.)*

Header now reads "Preprint -- Work in Progress (Ongoing Revision for ARR Cycle2)" with a link to research.kaysarulanas.me. The in-body GitHub link is currently a bare `https://github.com` placeholder — needs the actual repo URL, which is also what's blocking the Reproducibility section (open question 2 above).

**Action needed before Oct 12, 2026:** ARR is double-blind, so the version actually uploaded for the October cycle must be re-anonymized — author block reverted to "Anonymous ACL submission," the "Preprint — Work in Progress" header removed, and the Acknowledgments line reverted to "Omitted for anonymous review." Recommendation: keep the current de-anonymized `main.tex` as the working/preprint copy, and generate the anonymized ARR-submission copy from it as a distinct step in Phase 8 (Final pass) rather than toggling the same file back and forth.

## 8. Archived: full-rebuild direction ("SciRet-Verify")

Kept for future reference only — not part of the current plan. If this submission is accepted or the team later decides to expand into a full paper, this is the direction that was scoped out on 2026-05-22:

**Working title:** SciRet-Verify: Evidence-Centric Multimodal Retrieval-Augmented Generation for Scientific Question Answering

**Core thesis:** Scientific RAG systems should be evaluated not just on whether they produce plausible answers, but on whether each answer claim can be traced to and verified against the correct scientific evidence unit.

**Four linked contributions envisioned:**
1. Evidence-unit indexing — typed units (abstract, section paragraph, table, figure, caption, equation, citation context) instead of title/abstract-only chunks.
2. Adaptive retrieval router — route queries to lexical, dense, late-interaction, visual-page, table-aware, or citation-graph retrieval.
3. Claim-level citation verifier — decompose answers into atomic claims, check each against cited evidence (entailment/contradiction/insufficient), track numeric mismatch and modality-mismatch errors.
4. Cross-dataset benchmark — TREC-COVID/CORD-19, PubMedQA, SciFact, QASPER, SPIQA, ChartQA instead of CORD-19 only.

**Why parked:** too large a scope change for the current review cycle; none of the three reviewers asked for a new architecture — they asked for the existing study to be more statistically sound, more thoroughly baselined, and more honest about generalization limits. That's a revision, not a rebuild.

**If revisited later:** start from Phase 0 (Research Lock) in the original plan — finalize contribution statement, dataset list, and evaluation metrics before writing any new code. Sources consulted for that direction (ColPali, VisRAG, GraphRAG, RAGAS, MIRAGE, MTRAG, mmRAG, PubMedQA, QASPER, TREC-COVID, SciFact, SPIQA, ChartQA) are listed in the git history of the removed `sciret_reboot_research_plan.adoc` (see log for the exact commit/removal record).
