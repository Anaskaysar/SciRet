# SciRet — Research Approach for Next Submission

Status: active
Last updated: 2026-07-31
Supersedes: sciret_reboot_research_plan.adoc, SciRet_Reboot_Research_Plan.docx, SciRet_Reboot_Research_Plan_Enriched.docx, SciRet_Reboot_Research_Plan_Weekend_Kaggle.docx, SciRet_Revision_Checklist.md (all removed — see [SciRet_Reboot_Log.md](SciRet_Reboot_Log.md) 2026-07-31 entry)

## 1. Where we are

SciRet (Submission 8722, working title "SciRet: A Compute-Aware Empirical Study of Retrieval and Reranking for Scientific RAG") went through ARR review in early July 2026:

- R1 (5KS8): Soundness 4, Excitement 4, Overall 2.5 (Borderline Findings)
- R2 (SPUz): Soundness 2.5, Excitement 2.5, Overall 2.5 (Borderline Findings)
- R3 (VbPC): Soundness 2, Excitement 2, Overall 2 (Resubmit next cycle)

We submitted author replies (`SciRet_Author_Replies_v2.docx`) on 2026-07-10/11 committing to specific fixes per reviewer. Rebuttal is done; we are no longer bound to review-period constraints, and target venue for the next submission is **not fixed yet**. This document is the single source of truth for what "closing the gaps" means and how we're doing it. It replaces every earlier reboot-plan draft.

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
| CORD-19 version/filtering not specified | R2 | **Not yet answered — needs KB input.** Which CORD-19 release/date was indexed, and was it filtered (e.g., peer-reviewed only)? | Once KB confirms, add one sentence to Section 3.1 (Corpus and Scale Protocol). |
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

## 5. Explicitly out of scope for this pass

- New pipeline architecture, multimodal evidence units, claim-level citation verifier, adaptive retrieval router — all part of the archived full-rebuild direction (Section 8).
- Second dataset / cross-domain generalization test — optional stretch only, not committed to reviewers.
- Human evaluation study — optional stretch only, not committed to reviewers.
- Qualitative failure-case analysis — optional stretch only.

## 6. Open questions needing KB's input

These block specific writing tasks and can't be resolved by re-reading existing files:

1. **CORD-19 version/filtering** — which release/date, filtered to peer-reviewed only or not? (blocks 3.3 CORD-19 version sentence)
2. **Code/index release location** — actual repo URL and where pre-built indexes are hosted? (blocks 3.6 Reproducibility section)
3. **Third co-author** — name, order, affiliation for the human-verification contributor mentioned 2026-07-31 (blocks author list finalization; see main.tex)
4. **Target venue** — still open as of 2026-07-31. If the next submission is another double-blind ARR cycle, the paper needs to be re-anonymized (author names removed, Acknowledgments re-blinded) before submission — flagged so it isn't missed.

## 7. Authors

Restored 2026-07-31 (was "Anonymous ACL submission" for blind review):

- Kaysarul Anas Apurba
- Md. Hasibul Hasan (corresponding)
- [third co-author — human-verification contribution — TBD]

Affiliation(s) not yet provided — placeholder needed in `main.tex` until confirmed.

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
