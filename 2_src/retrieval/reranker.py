from __future__ import annotations

from typing import List, Tuple


def _token_set(text: str) -> set[str]:
    return set(t for t in text.lower().split() if t.strip())


class OverlapReranker:
    """
    Lightweight reranker based on token overlap.
    Replace with cross-encoder reranker in final experiments.
    """

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str, float]],
        top_k: int = 5,
    ) -> List[Tuple[str, str, float]]:
        q = _token_set(query)
        rescored = []
        for doc_id, text, base_score in candidates:
            d = _token_set(text)
            overlap = len(q.intersection(d)) / max(len(q), 1)
            final = 0.7 * float(base_score) + 0.3 * overlap
            rescored.append((doc_id, text, final))
        rescored.sort(key=lambda x: x[2], reverse=True)
        return rescored[:top_k]
