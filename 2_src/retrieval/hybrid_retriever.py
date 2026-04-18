from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple


def reciprocal_rank_fusion(
    runs: List[List[Tuple[str, float]]],
    k: int = 60,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = defaultdict(float)
    for run in runs:
        for rank, (doc_id, _) in enumerate(run, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
