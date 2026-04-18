from __future__ import annotations

import math
from typing import Iterable, List


def recall_at_k(relevant: Iterable[str], retrieved: List[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    hit = len(rel.intersection(set(retrieved[:k])))
    return hit / len(rel)


def mrr(relevant: Iterable[str], retrieved: List[str]) -> float:
    rel = set(relevant)
    for i, r in enumerate(retrieved, start=1):
        if r in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant: Iterable[str], retrieved: List[str], k: int) -> float:
    rel = set(relevant)
    dcg = 0.0
    for i, r in enumerate(retrieved[:k], start=1):
        if r in rel:
            dcg += 1.0 / (1.0 if i == 1 else math.log2(i))
    ideal_hits = min(len(rel), k)
    if ideal_hits == 0:
        return 0.0
    idcg = 0.0
    for i in range(1, ideal_hits + 1):
        idcg += 1.0 / (1.0 if i == 1 else math.log2(i))
    return dcg / idcg if idcg > 0 else 0.0
