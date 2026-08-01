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
