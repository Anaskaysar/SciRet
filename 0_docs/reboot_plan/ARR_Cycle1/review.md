## Review1: Official Review of Submission8722 by Reviewer 5KS8
Official Reviewby Reviewer 5KS805 Jul 2026, 09:41 (modified: 08 Jul 2026, 18:56)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer 5KS8, AuthorsRevisions
Paper Summary:
This paper presents SciRet, a compute-aware empirical study of retrieval and reranking components within a fixed scientific RAG pipeline over subsets of the CORD-19 corpus at three scales (1K, 5K, 15K papers).

Summary Of Strengths:
The paper frames a practical, compute-aware comparison across retrieval and reranking choices for scientific RAG, emphasizing controlled design and reproducibility rather than proposing a new model.

The paper is generally clear, with a concise description of the pipeline, explicit hyperparameters for RRF, and tabulated results.

Summary Of Weaknesses:
Pseudo-relevance labels are derived from the hybrid retriever being evaluated, creating circularity that undermines the validity of both recall and precision comparisons—especially the “negative reranking” conclusion.

No evaluation against established scientific IR/QA benchmarks with human relevance labels (e.g., TREC-COVID, SciFact, BioASQ), nor use of pooled labeling, which would mitigate circularity.

Insufficient detail on the 15-query set origin, exact overlap across scales, and prompt/generation settings (e.g., number of retrieved passages, truncation, and citation mapping details).

Comments Suggestions And Typos:
How exactly were pseudo-relevance labels constructed at each scale, and are they system-specific per scale or fixed across scales? Can you reconcile why hybrid R@3 is not 1.0 at 1K/5K if the top-3 hybrid results define relevance?

Are the 15 queries identical across the 1K/5K/15K scales? How were they sourced and validated, and do they have any topical skew that could favor sparse or dense retrieval?

Confidence: 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
Soundness: 4 = Strong: This study provides sufficient support for all of its claims. Some extra experiments could be nice, but not essential.
Excitement: 4 = Exciting: I would mention this paper to others and/or make an effort to attend its presentation in a conference.
Overall Assessment: 2.5 = Borderline Findings
Ethical Concerns:
There are no concerns with this submission

Needs Ethics Review: No
Reproducibility: 4 = They could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.
Datasets: 4 = Useful: I would recommend the new datasets to other researchers or developers for their ongoing work.
Software: 4 = Useful: I would recommend the new software to other researchers or developers for their ongoing work.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Publication Ethics Policy Compliance: I did not use any generative AI tools for this review

# Review 2

Official Review of Submission8722 by Reviewer SPUz
Official Reviewby Reviewer SPUz02 Jul 2026, 18:24 (modified: 08 Jul 2026, 18:56)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer SPUz, AuthorsRevisions
Paper Summary:
This paper presents SciRet, a compute-aware empirical study of retrieval and reranking for scientific question answering using the CORD-19 dataset. Rather than proposing a new model, the authors evaluate a fixed RAG pipeline across three corpus scales: 1K papers (1,034 chunks), 5K papers (5,160 chunks), and 15K papers (15,480 chunks). The pipeline combines sentence-window chunking, BM25 sparse retrieval, BGE-M3 dense retrieval, reciprocal rank fusion (RRF), optional cross-encoder reranking (MS MARCO-trained), and GPT-4o-mini-based grounded answer generation. The key findings are: (1) hybrid retrieval (BM25 + BGE-M3) is more robust than either sparse-only or dense-only retrieval, reaching Recall@10 of 1.000 at both 1K and 15K scales; (2) an MS MARCO-trained cross-encoder reranker consistently reduces precision on scientific text, suggesting domain mismatch outweighs the benefits of stronger query-passage interaction; and (3) RAGAS faithfulness increases with corpus scale. The paper frames itself as a reproducible, controlled comparative study rather than a benchmark and releases code, indexes, and evaluation outputs.

Summary Of Strengths:
Clear, focused empirical contribution: The paper asks a well-defined question (how do retrieval and reranking components behave across corpus scales?) and provides a controlled, systematic answer. The scope is appropriately narrow for an empirical study.
Compute-aware design: The explicit focus on resource-constrained settings and the three-scale evaluation (1K, 5K, 15K papers) is a practical contribution. The compute budget table (Appendix G) provides useful guidance for practitioners.
Negative result is valuable: The finding that an MS MARCO-trained cross-encoder reduces precision on scientific text is important. It highlights the risk of blindly transferring web-trained components to scientific domains and provides empirical evidence for a common intuition.
Controlled experimental design: By holding preprocessing, chunking, embedding model, retrieval settings, and generation fixed across scales, the authors isolate the effect of corpus scale. This makes failure modes easier to inspect and results more interpretable.
Reproducibility commitment: The authors commit to releasing code, indexes, and evaluation outputs that support replication and follow-up work.
Transparent limitations: The paper is honest about its limitations: pseudo-relevance labels, small query set (15 queries), titles/abstracts only, and RAGAS limitations. This transparency is commendable.
Practical implications: The findings provide actionable guidance for practitioners building scientific RAG systems: use hybrid retrieval, test rerankers before deployment, and be cautious about web-trained components.
Summary Of Weaknesses:
Pseudo-relevance label circularity: The retrieval evaluation uses pseudo-relevance labels derived from the top-3 hybrid results. This inherently favors the hybrid system and limits the validity of retrieval comparisons. The authors acknowledge this, but it remains a major methodological limitation.

Very small query set: Fifteen evaluation queries are extremely small. This severely limits statistical power and generalizability. The authors acknowledge this, but it undermines confidence in the results.

Limited evidence depth: The paper indexes only titles and abstracts, not full text. This is a significant limitation for scientific QA, where key evidence often appears in methods or results sections. The authors acknowledge this, but it substantially reduces the practical value of the findings.

Limited methodological novelty: The paper does not propose new methods; it evaluates existing components. While empirical studies are valuable, the contribution to \*ACL venues is likely to be judged on methodological novelty, which is limited here.

Single dataset focus: The evaluation is on CORD-19 only. Scientific QA encompasses many domains (biomedical, physics, computer science). Generalizability to other scientific corpora is untested.

Single generator: The paper uses only GPT-4o-mini for generation. It is unclear whether the generation findings (faithfulness increasing with scale) hold with other LLMs.

Reranking evaluation limited: The paper reports only precision before/after reranking. It does not report whether reranking changes recall or whether there are cases where reranking helps despite the average decline. A more nuanced analysis would be valuable.

Lack of significance testing: The paper does not report statistical significance for the differences between retrieval systems or for the reranking degradation. This is important given the small query set.

Paper type mismatch: The paper reads as a short report or technical note, but the review is for a potential conference paper. The scope and depth are more appropriate for a workshop or system demonstration paper.

Comments Suggestions And Typos:
Improve retrieval evaluation: The most critical improvement would be to obtain human relevance annotations (even for a subset of queries) to validate the pseudo-label findings. This would substantially strengthen the paper.

Expand query set: Even 50-100 queries would provide more statistical power. The authors should consider augmenting the 15 queries with additional questions or using existing QA datasets (e.g., COVID-QA).

Add significance testing: Report statistical significance (e.g., paired bootstrap or t-tests) for key comparisons. This is essential given the small sample size.

Expand reranking analysis: Report recall@K alongside precision@K for reranking. Explore whether reranking helps for some query types even if it hurts on average.

Consider human evaluation: A small human evaluation of generation quality (beyond RAGAS) would provide additional credibility, given RAGAS limitations.

Analyze failure cases: Include qualitative examples of cases where hybrid retrieval succeeds and where reranking fails. This would make the findings more interpretable.

Explain R@1 denominator: The paper notes that with three pseudo-relevant documents, the maximum R@1 is 1/3 = 0.333. This is an unusual detail; clarify why this metric is still meaningful.

Add more chunking analysis: The chunking analysis in Appendix B is brief. Consider reporting retrieval results with different chunking strategies to show the impact on downstream retrieval.

Typos:

Page 1: "retrieval- augmented generation" appears twice in the abstract.

Page 2: "each retriever" should be "each retrieval system" or similar.

Page 3: The footnote reference "text[[115, 873, 486, 919], [511, 412, 881, 492]]" appears to be a placeholder or artifact.

References: Some citation formatting is inconsistent (e.g., "and 1 others" in some entries).

Clarify CORD-19 version: Specify which version of CORD-19 was used and whether it was filtered (e.g., only peer-reviewed papers).

Mean chunk tokens: Table 1 shows "Mean chunk tokens = 215" across all scales. This is surprising; explain why the mean does not change with scale (same chunking strategy applied to different random samples).

Context precision is low: Table 4 shows context precision is very low (0.095-0.122). Discuss why this is the case and whether it indicates a problem with the retrieval or the RAGAS metric.

Confidence: 3 = Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design.
Soundness: 2.5
Excitement: 2.5
Overall Assessment: 2.5 = Borderline Findings
Best Paper Justification:
N/A — My overall assessment is "Borderline Findings" (2.5), not "Consider for Award" or "Borderline Award". Therefore, this section is not applicable.

Limitations And Societal Impact:
Strengths:

The paper includes a dedicated Limitations section that honestly discusses: pseudo-relevance labels, small query set, titles/abstracts only, RAGAS limitations, and lack of human evaluation.

The ethics statement notes that generated answers should not be used for medical decision-making without expert review.

The generative AI disclosure is transparent about using Claude and ChatGPT for prose refinement.

Missing or insufficient points:

Generalizability: The paper does not discuss whether findings generalize beyond CORD-19 or beyond COVID-19 literature. Scientific QA in other domains (physics, chemistry, computer science) may behave differently.

Failure mode analysis: The paper does not analyze specific failure cases or identify systematic biases in the retrieval or generation.

Clinical implications: While the paper warns against medical decision-making, it does not discuss the broader societal implications of automated scientific QA systems (e.g., reinforcing existing biases in the literature, over-reliance on automation, misinformation risks).

Data bias: CORD-19 papers are predominantly English-language and from specific journals. The system may not generalize to non-English scientific literature or to literature from underrepresented regions.

Equity considerations: The paper does not discuss who benefits from scientific RAG systems or who might be excluded (e.g., researchers with limited access to compute, practitioners in low-resource settings).

Societal impact considerations: The system is intended to assist with scientific question answering but has clear limitations (titles/abstracts only, pseudo-labels). The authors appropriately frame it as a research prototype rather than a deployable system. However, the paper should more explicitly discuss how the limitations could lead to harmful outcomes if the system is used without expert oversight.

Ethical Concerns:
None. The paper uses public data (CORD-19) and does not involve human subjects. The authors are transparent about the limitations and include a generative AI disclosure. The paper does not appear to violate the ACL Code of Ethics.

Needs Ethics Review: No
Reproducibility: 4 = They could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.
Datasets: 4 = Useful: I would recommend the new datasets to other researchers or developers for their ongoing work.
Software: 4 = Useful: I would recommend the new software to other researchers or developers for their ongoing work.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source Other: N/A
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Additional: N/A
Knowledge Of Authors Guess: N/A
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Publication Ethics Policy Compliance: I did not use any generative AI tools for this review

# Review 3

Official Review of Submission8722 by Reviewer VbPC
Official Reviewby Reviewer VbPC30 Jun 2026, 01:21 (modified: 08 Jul 2026, 18:56)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer VbPC, AuthorsRevisions
Paper Summary:
This paper presents a controlled evaluation of retrieval strategies for scientific retrieval-augmented generation (RAG) using the CORD-19 corpus. It compares sparse, dense, and hybrid retrieval methods, examines the performance of a general-domain reranker in the scientific domain, and evaluates generation quality using RAGAS. Experimental results show that hybrid retrieval consistently outperforms individual retrieval methods, while the general-domain reranker degrades retrieval performance and larger literature collections improve response quality.

Summary Of Strengths:
By keeping the retrieval and generation pipeline fixed while varying only the literature collection size, the study clearly isolates the impact of corpus size and enables a fair comparison of retrieval methods under different corpus sizes.

Summary Of Weaknesses:
Using only 15 queries for retrieval and generation evaluation makes it difficult to draw statistically reliable conclusions or assess the significance of the reported improvements.
Only a single general-domain MS MARCO cross-encoder is evaluated, making it difficult to determine whether the observed performance degradation is caused by domain mismatch or by the specific reranker itself.
Although the paper reports several RAGAS metrics, it does not adequately explain their definitions or discuss their limitations. Moreover, relying solely on automated LLM-based evaluation without human annotation or fact verification reduces the credibility of the reported generation quality.
Comments Suggestions And Typos:
Expand the evaluation to a substantially larger query set and report statistical significance tests where appropriate. Incorporating manually annotated relevance labels would further improve the reliability of the retrieval evaluation.
Include additional reranker baselines, particularly domain-adapted or scientific rerankers, to better distinguish between the limitations of general-domain rerankers and the effects of cross-domain transfer.
Provide a clearer description of the RAGAS metrics and complement automated evaluation with human annotation or factual verification. Combining automatic and manual evaluation would improve the reliability of the reported generation quality.
Confidence: 5 = Positive that my evaluation is correct. I read the paper very carefully and am familiar with related work.
Soundness: 2 = Poor: Some of the main claims are not sufficiently supported. There are major technical/methodological problems.
Excitement: 2 = Potentially Interesting: this paper does not resonate with me, but it might with others in the \*ACL community.
Overall Assessment: 2 = Resubmit next cycle: I think this paper needs substantial revisions that can be completed by the next ARR cycle.
Ethical Concerns:
There are no concerns with this submission

Reproducibility: 2 = They would be hard pressed to reproduce the results: The contribution depends on data that are simply not available outside the author's institution or consortium and/or not enough details are provided.
Datasets: 1 = No usable datasets submitted.
Software: 1 = No usable software released.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Publication Ethics Policy Compliance: I used a privacy-preserving tool exclusively for the use case(s) approved by PEC policy, such as language edits
