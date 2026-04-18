from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t.strip()]


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: List[str] = []
        self.doc_ids: List[str] = []
        self.doc_len: List[int] = []
        self.term_freqs: List[Counter] = []
        self.df: Dict[str, int] = defaultdict(int)
        self.avgdl: float = 0.0

    def fit(self, doc_ids: List[str], docs: List[str]) -> None:
        self.doc_ids = doc_ids
        self.docs = docs
        self.term_freqs = []
        self.doc_len = []
        self.df = defaultdict(int)
        for d in docs:
            toks = _tokenize(d)
            tf = Counter(toks)
            self.term_freqs.append(tf)
            self.doc_len.append(len(toks))
            for t in tf:
                self.df[t] += 1
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)

    def _idf(self, term: str) -> float:
        n = len(self.docs)
        dft = self.df.get(term, 0)
        return math.log((n - dft + 0.5) / (dft + 0.5) + 1.0)

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        q_toks = _tokenize(query_text)
        scores = []
        for i, tf in enumerate(self.term_freqs):
            dl = self.doc_len[i]
            score = 0.0
            for t in q_toks:
                f = tf.get(t, 0)
                if f == 0:
                    continue
                idf = self._idf(t)
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-8))
                score += idf * (f * (self.k1 + 1) / denom)
            scores.append((self.doc_ids[i], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
