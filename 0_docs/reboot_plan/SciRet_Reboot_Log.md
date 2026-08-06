# SciRet — Reboot Change Log

Append-only. Newest entry at the bottom. Every change made under the reboot plan gets an entry here: what was done, why, and what it replaced — so we can always trace back what changed and reverse it if needed.

---

## 2026-07-31 — Doc consolidation

**What:** Removed 5 duplicate/legacy reboot-plan documents from `reboot_plan/` and replaced them with exactly 3 files:
- `SciRet_Research_Approach.md` (detailed research approach — the "why" and full gap inventory)
- `SciRet_Reboot_Plan.md` (step-by-step actionable plan — the "what, in what order")
- `SciRet_Reboot_Log.md` (this file)

**Removed:**
- `sciret_reboot_research_plan.adoc` — 2026-05-22 draft proposing a full rebuild ("SciRet-Verify": multimodal evidence units, claim-level citation verification, 5 new datasets, new codebase). Core ideas archived in `SciRet_Research_Approach.md` Section 8 rather than deleted outright.
- `SciRet_Reboot_Research_Plan.docx` — docx export of an earlier version of the same May 22 plan.
- `SciRet_Reboot_Research_Plan_Enriched.docx` — enriched docx variant of the same plan.
- `SciRet_Reboot_Research_Plan_Weekend_Kaggle.docx` — Kaggle-scale-run variant of the same plan.
- `SciRet_Revision_Checklist.md` — the review-driven punch list (post-rebuttal, 2026-07-11). Its content was the primary source folded into `SciRet_Research_Approach.md` Section 3 and `SciRet_Reboot_Plan.md`.

**Kept as-is (not reboot-plan docs, kept as historical/reference records):**
- `8722_SciRet_A_Compute_Aware_old_Submitted.pdf` — the actual submitted PDF.
- `Review Page.pdf`, `review.md` — the three reviewer reports (R1/R2/R3).
- `SciRet_Author_Replies.docx`, `SciRet_Author_Replies_v2.docx` — rebuttal drafts; v2 was what was actually submitted 2026-07-10/11.

**Why:** too many overlapping/stale planning docs made it unclear which one was authoritative. Per KB's instruction, consolidated to one detailed doc + one actionable md + one log.

**Decision captured:** scope for the next submission is a **scoped revision** addressing the actual R1/R2/R3 review comments, not the full "SciRet-Verify" rebuild. See `SciRet_Research_Approach.md` Section 2 for the reasoning. This supersedes the informal direction implied by the old `sciret_reboot_research_plan.adoc`.

---

## 2026-07-31 — Author names restored

**What:** `SciRet_ARR_2026/main.tex` author block changed from `Anonymous ACL submission` to:
- Kaysarul Anas Apurba
- Md. Hasibul Hasan (corresponding)

**Source:** author list confirmed by KB via screenshot of the paper title/author line, 2026-07-31.

**Not yet done:** a third co-author for the human-verification contribution was mentioned by KB but not yet named — placeholder left in the plan (`SciRet_Research_Approach.md` Section 6/7). Affiliation(s) also not yet provided.

**Open risk flagged:** if the next submission target turns out to be another double-blind venue (e.g. another ARR cycle), this file will need to be re-anonymized (author names removed, Acknowledgments re-blinded to "Omitted for anonymous review") before submission. Target venue was confirmed **not fixed yet** as of 2026-07-31 — revisit before final submission.

**Context:** ARR Submission 8722 received 3 reviews in early July 2026 (R1 2.5, R2 2.5, R3 2 — "resubmit next cycle"). Author replies (rebuttal) were submitted 2026-07-10/11. As of 2026-07-31, KB confirmed the team is no longer bound to anonymity/preprint-hold constraints and wants a full gap-closing pass before the next submission, prompting this reboot-plan consolidation.

---

## 2026-07-31 — main.tex affiliation/email filled in (external edit)

**What:** `main.tex` was updated (outside this session) with affiliation "Laurentian University" for both authors, corresponding email `kaysarulanas2@gmail.com` for Md. Hasibul Hasan, and a "Preprint — Work in Progress" running header (fancyhdr) added. Reflected in `SciRet_Research_Approach.md` Section 7.

---

## 2026-07-31 — Final Research Plan (Word) generated

**What:** Built `SciRet_Final_Research_Plan.docx` — a polished, shareable Word version combining `SciRet_Research_Approach.md` (gap inventory, scope decision, open questions) and `SciRet_Reboot_Plan.md` (phased checklist) into one document, plus the archived SciRet-Verify appendix. The two `.md` files remain the live working source of truth; the docx is a point-in-time snapshot for sharing outside this working folder — regenerate it if the plan changes materially.

---

## 2026-08-01 — reboot_plan/ relocated for cross-device sync

**What:** KB moved the entire `reboot_plan/` folder from `Sciret/reboot_plan/` (outer, non-git-tracked working folder) to `Sciret/SciRet/0_docs/reboot_plan/` (inside the inner git-tracked `SciRet/` repo) so it's accessible from a second device. No content was lost — same files, new location. All paths in this doc set now resolve relative to `SciRet/SciRet/0_docs/reboot_plan/`.

---

## 2026-08-01 — Target venue decided: October 2026 ARR cycle

**What:** KB confirmed the goal is at least one paper at a top NLP venue and chose to revise-and-resubmit through ARR rather than commit the current paper as-is to EMNLP 2026 or submit to ICNLP 2027.

**Researched:** current ACL Rolling Review cycle dates (aclrollingreview.org/dates, checked 2026-08-01):
- August 2026 cycle: submission Aug 3, 2026 — ruled out, no time to do the revision work.
- **October 2026 cycle: submission Oct 12, 2026 — adopted as the target.** Feeds NAACL 2027 / COLING 2027 (ARR commitment date Dec 20, 2026).
- EMNLP 2026 / AACL 2026 commitment date for the May 2026 cycle (this submission's cycle): Aug 2, 2026 — this is what's being let lapse.

**ICNLP 2027 considered and ruled out:** iconf.org listing, deadline Nov 30, 2026, conference Apr 16–18, 2027, Zhenjiang, China, organized by Jiangsu University of Science and Technology, proceedings in IEEE Xplore. Not ACL-affiliated, general/regional scope with no retrieval or RAG focus, lower profile than *ACL venues — doesn't match the "top NLP venue" goal.

**Decision consequences (see `SciRet_Research_Approach.md` Section 1a and `SciRet_Reboot_Plan.md` Phase 0/8):**
1. EMNLP 2026 will NOT be committed by the Aug 2, 2026 deadline — deliberate, not an oversight, since committing would jeopardize ARR resubmission eligibility.
2. ~10 weeks of runway (2026-08-01 → 2026-10-12) for the Section 3 gap-closing work in the Research Approach doc.
3. `main.tex` currently has real author names/affiliation restored (2026-07-31, for preprint/working use) — since ARR is double-blind, a **separate anonymized copy must be generated before Oct 12 submission**. Added as an explicit Phase 8 step rather than toggling the working file.

---

## 2026-08-01 — arXiv preprint policy checked; Phase 6 writing fixes applied to main.tex

**What:** Two things, same session.

1. Confirmed current ACL/ARR anonymity policy (aclrollingreview.org/anonymity, in effect since Feb 2024): authors may post and discuss non-anonymous preprints on arXiv at any time, including during double-blind review. No anonymity-period restriction. KB decided to post v1 to arXiv, but only after the writing-only fixes land, so the first public/citable version isn't the roughest draft — "the submission version will be pin perfect this time."

2. Applied all Phase 6 writing-only fixes to `SciRet_ARR_2026/main.tex`:
   - Introduction: added explicit "short paper track, controlled empirical comparison" framing.
   - Section 3.1 (Corpus and Scale Protocol): added chunk-token stability explanation near Table 1.
   - Section 3.3 (Generation and Metrics): added RAGAS metric definitions (faithfulness, answer relevancy, context precision, context recall) + LLM-judge bias caveat.
   - Section 3.4 (Retrieval Evaluation Note): added label-construction explanation (why hybrid R@3 < 1.0 at 1K/5K) and R@1 denominator justification (relative signal within the labeling protocol, not an absolute benchmark).
   - Section 4.3 (Generation Scores Improve With Scale): corrected the context-precision explanation to the citation-based framing (fraction of retrieved passages the generator actually cites) — previous text ("topic-related but not tightly targeted") didn't match what was promised in the rebuttal.
   - Section 6 (Limitations): added generalization (single dataset/generator, untested beyond CORD-19/COVID-19), CORD-19 English-language/journal skew and equity, and misinformation/over-reliance risk without expert oversight.

**Not done:** CORD-19 version/filtering sentence — still blocked on KB input (Phase 0 open question).

**Verified:** recompiled with `latexmk -pdf` in a scratch copy (temporarily commented out an unrelated missing-package line, `inconsolata.sty`, which isn't installed in this sandbox and is unrelated to the edits — not changed in the real file). Compiled clean to a 6-page PDF, only pre-existing cosmetic underfull-hbox warnings. Read the rendered PDF text back to confirm all new sentences render correctly and cross-references (Section 3.4, Table 2, Section 6) resolve.

**Next:** once these are confirmed good, post `main.tex` (de-anonymized/preprint copy) as arXiv v1. Push v2/v3 as the experimental work (50-query expansion, reranker baseline, bootstrap CIs, TREC-COVID check, reproducibility section) lands.

---

## 2026-08-01 — CORD-19 version/filtering resolved, last Phase 6 item closed

**What:** KB uploaded `metadata.readme` (the changelog bundled with the CORD-19 download). Most recent entry: release dated **2022-06-02**, 1,056,660 total metadata rows — this is the final CORD-19 release before AI2 discontinued the dataset, so it's the version the project has been using throughout (no later release exists to have used instead).

Checked `2_src/data/loader.py` (`build_tier_subset`) for the actual filtering/sampling logic: no journal, peer-review, or language filter is applied. The only cleaning steps are deduplication by `cord_uid` and a minimum abstract length of 100 characters, followed by random sampling with a fixed seed (42) into each corpus scale.

Added one sentence to Section 3.1 (Corpus and Scale Protocol) of `main.tex` capturing both facts. Recompiled (`latexmk -pdf`) and confirmed the rendered text.

**Result:** Phase 6 (writing-only fixes) is now fully complete — the CORD-19 version sentence was the last remaining item. Two open questions remain (code/index repo URL, third co-author) before the Reproducibility section and author list can be finalized.

---

## 2026-08-01 — Third co-author added; corresponding-author error corrected

**What:** KB updated `main.tex` directly (external edit) to add the third co-author and refresh the header:
- Author block now: Kaysarul Anas Apurba$^1$ (corresponding), Md. Hasibul Hasan$^1$, Rofiqul Alam Shehab$^2$; $^1$Laurentian University, $^2$North South University.
- Header changed to "Preprint -- Work in Progress (Ongoing Revision for ARR Cycle2)" with link to research.kaysarulanas.me.
- In-body link is currently a bare `https://github.com` placeholder, not the actual repo URL.

**Correction:** earlier entries in this log and in `SciRet_Research_Approach.md` (2026-07-31, 2026-08-01) incorrectly stated "Md. Hasibul Hasan (corresponding, kaysarulanas2@gmail.com)." That was wrong — the `main.tex` source has always had **Kaysarul Anas Apurba** as corresponding author (`\thanks{~Corresponding author.}` was always attached to Apurba's name, confirmed by re-reading the file). This was a transcription error when summarizing, not an actual change to the paper. Fixed in `SciRet_Research_Approach.md` Section 7, 2026-08-01.

**Resolved:** third co-author open question (Phase 0) — Rofiqul Alam Shehab, North South University.

**Still open:** code/index repo URL — the placeholder `https://github.com` in the author block is the same blocker as the Reproducibility section (Section 3.6 of the Research Approach doc).

---

## 2026-08-04 — Phase 6 fixes lost and reapplied; 4th author added

**What:** KB added a 4th author, Asab Azad (Laurentian University), and updated the GitHub link to `https://github.com/anaskaysar/sciret`. But the `main.tex` this landed on had reverted to a pre-2026-08-01 version — all seven Phase 6 writing fixes (short-paper framing, CORD-19 version sentence, chunk-token note, RAGAS definitions, label-construction/R@1 explanation, context-precision correction, expanded Limitations) were gone. Likely cause: an older Overleaf revision or local backup was used as the base when the author block was edited, rather than the file this session had last written.

**Fixed:** reapplied all seven Phase 6 edits on top of the current 4-author block. Recompiled (`latexmk -pdf`) — 0 errors, 7 pages, author line renders correctly with all four names.

**Risk flagged for the team-expansion phase (see next entry):** with KB working locally/via this session and now potentially 2-3 more collaborators editing `main.tex` (via Overleaf or their own clones), silent reverts like this can recur. Worth agreeing on a single source of truth (e.g., Overleaf as the only live-edit copy, or a clear git branch/PR flow) before collaborators start contributing text, so fixes don't get lost again.

**Author list as of 2026-08-04:** Kaysarul Anas Apurba$^1$ (corresponding), Md. Hasibul Hasan$^1$, Rofiqul Alam Shehab$^2$, Asab Azad$^1$. $^1$Laurentian University, $^2$North South University.

---

## 2026-08-04 — Two new collaborators joining; work division requested

**What:** KB has two people interested in joining the project as co-authors (consistent with the two most recent author additions, Shehab and Azad). Meeting scheduled for 2026-08-05 with one of them. KB wants: (1) a proposed work breakdown across the existing Phase 1-8 plan for 3 people, to discuss/adjust in the meeting, and (2) the plan/task list updated afterward with their input. Final goal restated: get the paper accepted (October 2026 ARR cycle → NAACL 2027 / COLING 2027).

**Context carried over:** Shehab was previously earmarked (2026-07-31) for the "human verify part" contribution — relevant to Phase 5 (TREC-COVID qrel anchor check) and the optional human-eval stretch goal, and to reviewing ground-truth answers for the expanded query set (the original April 2026 working paper had a domain-knowledgeable contact review ground truth — likely the same role).

**Done:**
1. Added "Proposed owner" tags to every phase in `SciRet_Reboot_Plan.md` (draft, pending confirmation at the 2026-08-05 meeting): Azad → Phase 1 (query expansion), Apurba → Phase 2 + Phase 7 (reranker baseline, reproducibility), Hasan → Phase 3 + Phase 4 (recall@K, statistical rigor), Shehab → Phase 5 (TREC-COVID anchor check) + optional human-eval stretch, whole team → Phase 8.
2. Also filled in Phase 4 with the actual statistical protocol from the archived April 2026 working paper (Shapiro-Wilk → paired t-test/Wilcoxon, Cohen's d, bootstrap CI) — this had been identified as a useful "hidden asset" on 2026-08-04 but not yet written into the plan; done now.
3. Built `SciRet_Team_Kickoff_Brief.docx` — 4-page meeting-ready brief: project recap, review scores, team list, condensed gap list, the work-division table, and discussion questions (repo status, time budget, compute access, author order) for the 2026-08-05 meeting.

**Next:** after the meeting, update the "proposed owner" tags to confirmed assignments and log what was decided (including any changes to the work breakdown based on their input).

---

## 2026-08-04 — Perplexity planning conversation surfaces a bigger execution scope

**What:** KB shared a Perplexity conversation about the coding rebuild. Two immediate fixes and several open scope questions came out of it.

**Immediate fixes applied to `main.tex`:** removed two triple-hyphen ("---") em-dash artifacts from the Phase 6 writing-fix edits (2026-08-01) — flagged in the Perplexity conversation as a tell for AI-generated text. Replaced with plain commas/periods. Recompiled clean, 0 errors. Will watch for this in all future edits.

**New decisions from that conversation (not yet reconciled with this plan):**
1. Rebuild the pipeline as Kaggle-only notebooks (KB's only real execution environment), one notebook family per scale, template-first (build 1K clean, then clone to larger scales) rather than writing many notebooks at once.
2. Document every parameter choice inline (markdown cells) so "why this setting" has a recorded answer instead of a reconstructed-after-the-fact rebuttal.
3. `plt.show()` in every plotting cell, not just saved to output — visible in the notebook itself.
4. Corpus: title+abstract only for now, but build a `load_documents(mode="abstract"|"full_text")`-style loader abstraction so switching to full text later doesn't require a rewrite.
5. Query set: "definitely not 15" — targeting 50 queries first, with 100 considered if time/annotation quality allow. **This is larger than the 50-query figure already committed in the rebuttal and used in the team kickoff brief sent for tomorrow's meeting.**
6. Evaluation: three labeling tracks kept separate in code — pseudo-relevance labels (internal), external benchmark anchor (TREC-COVID, already Phase 5), and human-annotation slots for a subset (fits Shehab's role). This formalizes human annotation as a real part of the plan, not just an optional stretch goal.
7. Cross-scale validation considered up to 50K papers (reviving the original April 2026 working paper's Tier 2 target), structured as: (1) data+indexing notebook, (2) retrieval+reranking notebook, (3) generation+RAGAS combined in one notebook per KB's Kaggle constraint, (4) a final cross-scale analysis notebook comparing 1K/5K/15K/50K.
8. Old codebase (`2_src/`, etc.) to be deleted once the new notebooks reproduce core outputs at one scale — not deleted immediately.

**Not yet resolved — needs a decision before the plan/brief are finalized:**
- 50 vs. 100 queries: the brief and Phase 1 currently say 50. Reviewers asked for more statistical power via more queries, not specifically 100; last time's Kaggle time budget was already the stated bottleneck. Recommend locking 50 as the committed target (matches rebuttal) and treating 100 as an explicit stretch goal, not a default plan.
- Scale ceiling: current plan is 1K/5K/15K (matches rebuttal and existing results). Going to 50K was not requested by any reviewer and reintroduces the exact compute-budget risk that limited the original April 2026 attempt. Recommend keeping 1K/5K/15K as committed and treating 50K as optional/stretch, same reasoning as the query-count question.
- These two questions directly affect the work-division brief already sent for the 2026-08-05 meeting (Phase 1 owner Azad was scoped for "~35 new queries to reach 50," not up to 85 for 100) — worth raising explicitly in that meeting rather than silently changing scope beforehand.

---

## 2026-08-04 (evening) — Scale target update + Introduction rewrite

**What:** Two things.

1. KB set an interim scale target: "at least 30K this time" — splits the difference between the committed 15K ceiling and the 50K stretch discussed earlier. Coding/notebook implementation discussion still deferred to the 2026-08-05 meeting per KB ("we will talk about the coding later tomorrow"); this is a target to raise there, not yet reflected in `main.tex`'s reported results (those still show the real completed 1K/5K/15K experiments).

2. Rewrote the Introduction in `main.tex`. KB shared three EMNLP (2025) paper excerpts as reference points — a bias/regard-classification paper, an in-context-learning ensemble-prompting paper, and a long-text-outline paper — noting all three visualize the problem via a diagram early and open with a concrete framing rather than generic background. Applied the same approach to SciRet's Introduction: opens with a concrete example query ("What is the efficacy of remdesivir...") to ground the abstraction, adds a "the choice is not neutral, and not static" paragraph making the scale-dependent-failure risk concrete (a reranker that helps at one scale, hurts at another), and moves the reference to Figure~\ref{fig:pipeline} earlier and ties it directly to the research question instead of trailing after the Contributions setup. Kept all existing framing (short-paper track, controlled comparison) and factual content unchanged — no new claims, just restructured for narrative pull. Recompiled clean, 0 errors, 7 pages.

**Approach for future sections:** KB said "we will fix one by one" — treat this as the first of a section-by-section rewrite pass, not a one-off. Related Work, Method, and Discussion are candidates for the same treatment later.

---

## 2026-08-04 (evening) — New Figure 1 design approved: "domain mismatch" reranker diagram

**What:** Following the Introduction rewrite, KB drafted two mockups (in what looks like a diagramming tool) of a new intro-level figure visualizing the reranker's domain-mismatch failure (a query's top-3 hybrid-retrieval results reordered by the MS MARCO reranker, correct paper demoted from rank 1 to rank 3) — grounded in the actual Table 4 finding, not a new claim. KB confirmed the second, more compact version is the template to rebuild cleanly in draw.io, with corrected input text and writing.

**Two issues flagged, to fix during the draw.io remake:**
1. Version 1's rank-3 citation ("Adaptive COVID-19 Treatment Trial (ACTT-1): Final Results" attributed to "Pan H, Peto R, Henao-Restrepo AM") mismatches a real title with the wrong real authors (those are WHO Solidarity trial authors, not ACTT-1/Beigel et al.). Not yet confirmed whether card content is meant to be literal logged pipeline output or illustrative — if literal, must be pulled from actual stored per-query results; if illustrative, should avoid pairing real author names with the wrong paper.
2. In both versions, the rank-3 card in the reranked panel (the correct paper, just buried) is marked with a flat red X, which reads as "this paper is wrong" rather than "this paper is right but misranked." Fix: a distinct demoted/downgraded marker, or carry over V1's bottom-panel clarifying caption line even in the more compact layout.

**Status:** design approved as the template; KB rebuilding in draw.io now. Not yet wired into `main.tex` — once the image file exists, add as a new figure early in the Introduction (ahead of or replacing the current pipeline diagram's Introduction placement), renumber the existing architecture figure into Method, and update the Introduction's figure reference.

---

## 2026-08-04 (evening) — New Figure 1 wired into main.tex as a placeholder

**What:** KB dropped the current draw.io export into the project folder as `IntroDuction_Fig_needs_to_remake.png` — filename deliberately kept as a self-reminder that this is not the final version. Wired it into `main.tex`:

- Cropped the image (`fig_domain_mismatch.png`) to remove the caption/title band baked into the PNG itself, so it doesn't duplicate or conflict with the LaTeX-generated `\caption`/numbering.
- Inserted as a `figure*` right after the first Introduction paragraph, with `\label{fig:domain_mismatch}`, and a text cross-reference added to that paragraph ("Figure~\ref{fig:domain_mismatch} shows one instance of this failure...").
- Caption corrects the table reference baked into the original image (it said "Table 4"; the actual precision-drop table is `tab:rerank`, which resolves to **Table 3**).
- Caption ends with "Placeholder figure, to be redrawn." so it's visibly marked as in-progress in the compiled PDF, not just the filename.
- Added an inline LaTeX comment (`% TODO(KB)`) directly above the figure repeating the still-unresolved citation mismatch from the previous entry (rank-3 card: "ACTT-1" title paired with WHO Solidarity trial authors Pan/Peto/Henao-Restrepo instead of the real ACTT-1 authors, Beigel et al.) and pointing at `intro_diag_improvement.txt` (KB's saved copy of the earlier "looks AI-generated" design notes) for the redraw.
- Existing pipeline diagram automatically renumbered to Figure 2 (no manual change needed — order in the .tex file determines LaTeX numbering); its own Introduction reference text ("Figure~\ref{fig:pipeline} shows the pipeline...") now correctly prints as "Figure 2" too.

**Verified:** recompiled with `latexmk` in a scratch dir, 0 errors, page count unchanged at 7. Confirmed via `pdftotext` that the new figure renders as Figure 1, the pipeline diagram as Figure 2, and the table cross-reference resolves to Table 3.

**Not yet done — carried over from the previous entry, still open:** the rank-3 citation mismatch is not fixed, only flagged (in-paper comment + this log). Fix it when redrawing in draw.io, before this stops being a placeholder.

---

## 2026-08-04 (late) — Committed to long-paper track (8-page main body) + page budget

**What:** KB pasted the venue's long-paper CFP rule (8-page main body, mandatory Limitations after the conclusion, optional Ethics, unlimited references/appendix) and asked to budget pages per section before continuing Phases 1–5. Asked KB directly whether this meant switching off the previously rebuttal-committed "short-paper track" framing — confirmed yes, switch to long paper, since Phases 1–5 (50 queries, second reranker, recall@K, significance testing, TREC-COVID anchor check) add more than a short paper's ~4-page body can hold.

**Three changes to `main.tex`:**
1. **Structural fix:** Limitations was numbered (§6) and placed *before* Conclusion (§7) — violates the pasted rule (Limitations must be after the conclusion, before references, and is conventionally unnumbered like Ethics Statement). Reordered to Conclusion → Limitations, changed `\section{Limitations}` to `\section*{Limitations}`. Downstream sections renumber automatically; Conclusion is now §6. Fixed the one dangling `\ref{sec:limitations}` (Section 3.4) to read "the Limitations section" as plain text, since an unnumbered section can't be usefully cross-referenced by number.
2. **Framing fix:** the Introduction's rebuttal-committed sentence ("...submitted to the short-paper track...") now reads "...for the long-paper track..." and names the four Phase 1–5 additions (second reranker baseline, recall-based reranking analysis, significance testing, external label-validity check) as what the extra length buys. Added a `% TODO(KB)` comment flagging that "15 evaluation queries per scale" in the same paragraph is still the pre-Phase-1 number and needs updating once the 50-query run lands. Also dropped a stray "short-paper narrative" reference in the Appendix Overview to "main narrative."
3. **Page budget:** measured current per-section page usage from the compiled PDF (front matter+abstract 0.4, Intro 0.9, Related Work 0.7, Method 1.5, Results 1.0, Discussion 0.35, Conclusion 0.35 ≈ 5.2 pages used of 8) and built a target allocation for Phases 1–5 additions, landing at ~7.45 pages with a ~0.55-page buffer. Full table now lives in `SciRet_Reboot_Plan.md` under "Page Budget." Ground rule set: full statistical detail (per-query p-values, bootstrap CIs) and the TREC-COVID qrel-matching methodology belong in the Appendix, not the main body — Results should only carry significance markers and headline numbers.

**Verified:** recompiled with `latexmk`, 0 errors, page count unchanged at 7 (structural/framing edits are text-only, no new content yet).

**Not yet done:** Phases 1–5 experimental work itself (unchanged from before) — this entry only locks the track decision and the budget they need to fit inside.

---

## 2026-08-05 — Solve-first principle adopted; new Phase 1B (full-text indexing pilot)

**What:** KB flagged a pattern across this whole revision: issues were consistently being resolved by softening the claim and writing a Limitations sentence instead of actually fixing them, and that pattern already failed once (R3 scored the original submission 2/2, "resubmit next cycle," disclosure alone wasn't enough). New standing instruction: attempt to solve a gap before defaulting to disclosure, unless (a) the reviewers themselves asked for disclosure rather than a new experiment, or (b) the fix is genuinely out of reach this cycle (new-domain dataset, full human eval). Recorded as "Working principle" in `SciRet_Research_Approach.md` Section 1b.

**First target: full-text indexing.** The paper indexes only titles/abstracts and disclosed this as a Limitation without attempting it — a self-imposed choice, not a reviewer requirement, so it's the clearest first case for the new principle. KB is starting a 1K-scale full-text pilot now.

**Checked the actual code before writing the plan** (`Sciret/SciRet/2_src/`): full-text indexing does not exist yet. `chunker.py`'s `build_chunks()` hardcodes `f"{title} {abstract}"` — no full-text field anywhere in the chunking path. `pdf_parser.py` is a literal unimplemented stub (`extract_figure_manifest_stub`, returns empty records, docstring says "Replace with PyMuPDF/pdfplumber extraction in multimodal phase"). Flagged two real risks: (1) not every CORD-19 row has full text — `loader.py`'s `build_tier_subset()` needs a `has_pdf_parse`/`has_pmc_xml_parse` filter before sampling, or the "1K full-text" sample will silently include abstract-only papers; (2) full bodies are 5–20x longer than abstracts to embed, and the team's prior Kaggle free-tier time budget was already the stated bottleneck at 15K abstract-only. Recommended using CORD-19's own pre-parsed `document_parses/pdf_json` / `pmc_json` body text instead of building a real PDF parser — much smaller lift than what `pdf_parser.py`'s stub docstring implies is needed.

**Added Phase 1B to `SciRet_Reboot_Plan.md`** (owner: KB, parallel to Phase 1, not gated by the 50-query expansion): coverage check, full-text loader, chunker extension, full pipeline re-run at 1K only, comparison against the existing abstract-only 1K numbers. Decision gate set up three ways depending on outcome: (1) if full-text changes the story and compute allows, extend to 5K/15K and move the original abstract-only 1K/5K/15K results to the Appendix as the development trail, matching KB's instruction; (2) if full-text helps but can't scale past 1K in time, keep the multi-scale study on title+abstract as primary and add full-text-vs-abstract as a separate 1K depth ablation (framed as a compute/depth tradeoff, on-theme for this paper); (3) if it doesn't pan out, keep abstract-only as primary but narrow the Limitations sentence to reflect that full-text was actually tested, not just disclosed as untested.

**Also updated `SciRet_Research_Approach.md`:** new Section 1b (working principle) and new Section 3.8 (Evidence depth gap, previously untracked since no reviewer raised it) with the same engineering-risk notes, plus a note in the priority order (item 9) that this runs in parallel with items 1–8.

**Not touched yet:** `main.tex` itself — no Method/Results/Limitations text changed, since there are no pilot numbers yet to write from. That's the next step once Phase 1B produces results.

---

## 2026-08-05 (later) — Found the 50-query set already exists; built Azad's handoff sheet

**What:** KB asked how to delegate the query task to Asab Azad. Before drafting delegation instructions, checked whether the codebase already had anything relevant — found `1_data/eval/queries.json` and `3_notebooks/General/03_query_set.ipynb` already contain a stratified 50-query set matching the rebuttal's exact 6-theme commitment (Imaging 10, Molecular/Mechanistic 10, Clinical Outcomes 10, Treatment 10, Dataset/Methodology 5, Cross-domain Synthesis 5). The notebook says "all scale experiments load this file" — this is live pipeline scaffolding, not archived leftovers, and it does not overlap with the 15 queries actually used in the current `main.tex` results (separately drafted, different wording).

**Updated Phase 1 in `SciRet_Reboot_Plan.md`:** recommended adopting this 50-query set wholesale instead of drafting ~35 new ones from scratch — closer to what was promised in the rebuttal, and much less work. Azad's task changed from drafting to reviewing/vetting: answerability-from-abstract, retrieval-method skew, duplicates, and a sanity check on the uneven 10/10/10/10/5/5 theme split.

**Bonus finding:** the same notebook scopes a manual ground-truth annotation task (50 queries × 3 passages = 150 binary judgments, ~2-3 hours) as the intended fix for pseudo-label circularity — stronger than the TREC-COVID anchor check alone, and a close match for Rofiqul's already-discussed human-verification role. The candidate-passage CSV it needs doesn't exist yet (someone technical has to generate it first). Flagged in the plan under Phase 1, to raise alongside Phase 5.

**Built `handoffs/SciRet_Query_Review_Phase1.xlsx`** for Azad: an Instructions tab explaining the task and why it matters, and a "Queries to Review" tab with all 50 queries pre-filled (theme, number, query text) plus four blank review columns and one shaded example row. Ready to send as-is.

---

## 2026-08-05 (evening) — First new code for 1K: consolidated pipeline notebook + an important finding

**What:** KB asked to write the first new code for the 1K full-text pilot, archive the old `scale_1K`…`scale_100K` folders, and replace them with a single standalone notebook (per the earlier Kaggle-notebooks-only direction from the Perplexity conversation).

**Before archiving anything, read through the actual old notebooks to reuse their real (working) logic rather than reinvent it.** Important distinction found in the process: `2_src/` (the embeddings/retrieval/generation/reranker Python modules) are placeholder stubs, not what produced the paper's numbers — `TextEmbedder` is a hash-based fake embedder, `OverlapReranker` is token-overlap not a cross-encoder, `TextGenerator` is a template, not an LLM call. Docstrings say as much ("Replace with ... for final experiments"). The **real** pipeline lives inline in `3_notebooks/scale_1K/*.ipynb`: BGE-M3 via `sentence-transformers`, real BM25Okapi, the actual MS MARCO cross-encoder, real GPT-4o-mini calls, real RAGAS. The new consolidated notebook is built from that real logic, not from `2_src/`.

**Bigger finding — flagging for KB to confirm, not yet acted on:** `scale_1K/05_generation_ragas.ipynb` has a saved execution output showing it was last run with **50 queries** ("Queries: 50 | Chunks: 1,029"), and the resulting `05b_ragas_eval.ipynb` RAGAS output — `{'faithfulness': 0.9169, 'answer_relevancy': 0.6796, 'context_precision': 0.0950, 'context_recall': 0.2600}` — matches the paper's currently published 1K RAGAS numbers (0.917, 0.680, 0.095, 0.260) almost to the decimal. That strongly suggests the generation+RAGAS numbers actually in `main.tex` right now came from a 50-query run, not the 15 queries the paper's text claims (Table `tab:dataset`, Appendix F, and the Limitations section all say 15). Two possibilities: either the paper's "15 queries" text is simply stale and should be corrected to 50 (a pure documentation fix, no new compute needed), or this 50-query run was exploratory and shouldn't be trusted as the source of the published numbers. **Not resolved — needs KB's input before touching the paper's query-count claims.** Did not find matching evidence either way for whether Table `tab:recall` (retrieval Recall@K) and Table `tab:rerank` (reranking Precision@K) were also computed on 50 queries — `03_retrieval_ablation.ipynb` and `04_reranking.ipynb` have no saved outputs in this snapshot, only `05`/`05b` do.

**Archived, not deleted:** moved `scale_1K/`, `scale_5K/`, `scale_15K/`, `scale_30K/`, `scale_50K/`, `scale_75K/`, `scale_100K/` to `6_legacy/notebooks_scale_1K_to_100K_archived_2026-08-05/` (plain filesystem move, not `git rm` — the working tree already had substantial uncommitted changes unrelated to this session on the `scale_5k` branch, so no git operations were performed on KB's behalf). Chose archive-not-delete specifically because these folders are where the 50-vs-15-query evidence above was found, and because it matches the project's own existing `6_legacy/notebooks_archived_2026-05-04/` convention.

**Built `3_notebooks/sciret_pipeline.ipynb`** — single file, parameterized by `N_PAPERS`/`SCALE_LABEL` at the top (change those two, nothing else, to rerun at a different scale — same convention as the old per-scale config cells, just no longer duplicated across 5 files × 7 scales). Configured for today's run: 1K papers, `RUN_FULLTEXT = True`. Sections: sample (stratified by year, seed 42, same as before) → full-text coverage check (detects either the 2022-06-02 release's `pdf_json_files`/`pmc_json_files` path columns or older releases' boolean `has_pdf_parse`/`has_pmc_xml_parse`, whichever the actual `metadata.csv` has) → full-text loader (reads CORD-19's own pre-parsed `document_parses/*_json/*.json` `body_text`, not a PDF parser) → chunking (same sentence-window strategy, now run for both an abstract-only and a full-text condition) → BGE-M3 + BM25 indexing per condition → Recall@K, Precision@K (reranking), and (optional, toggleable, costs API calls) generation + RAGAS, computed for both conditions → a final side-by-side comparison table mirroring the Phase 1B decision gate. `REQUIRE_FULLTEXT_FOR_SAMPLE = True` by default so the abstract and full-text conditions are evaluated on the exact same 1,000 papers.

**Not run yet** — this is code only, written and syntax-checked (all cells compile) but not executed; no GPU/CORD-19 data/API keys available in this session. KB runs it on Kaggle/local.

**Updated `3_notebooks/README.md`** to describe the new single-notebook structure and point at the archive.

**Updated same day:** KB confirmed a paid OpenAI key with budget to spare, so `RUN_GENERATION_RAGAS` stays on by default (cost estimate added: well under $1 for 50 queries × 2 conditions on `gpt-4o-mini`). Rewrote the notebook to be Kaggle-native rather than dual-purpose: reads CORD-19 from `/kaggle/input/`, writes results to `/kaggle/working/sciret_results/`, reads the API key from Kaggle Secrets (falls back to local `.env` only when not on Kaggle), and the 50 evaluation queries are now embedded directly in the notebook instead of read from `1_data/eval/queries.json` — avoids needing to upload a file to Kaggle at all. Added an environment-check cell at the top (lists attached `/kaggle/input` datasets, confirms GPU) so setup problems surface before the expensive cells run. Gave KB a full first-time step-by-step (dataset attach, GPU/internet toggle, Secrets setup, run order) outside this doc, in-chat.

**Still open:** whether `REQUIRE_FULLTEXT_FOR_SAMPLE` should default differently depending on how the coverage check comes back on a real run.

**Resolved (not a bug to chase):** KB's call on the 15-vs-50-query discrepancy — don't reconcile it. Every number in the paper gets rewritten once `sciret_pipeline.ipynb` actually runs, and most sections are getting rewritten anyway, so auditing which old run produced which currently-published number is wasted effort. Applies generally going forward: don't spend time reconciling/explaining old numbers that a fresh run will overwrite — flag genuine blockers to a fresh run, not discrepancies in numbers already scheduled for replacement.

---
