from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from embeddings.text_embedder import TextEmbedder


class DenseRetriever:
    def __init__(self, embedder: TextEmbedder) -> None:
        self.embedder = embedder
        self.doc_ids: List[str] = []
        self.matrix: np.ndarray | None = None

    def fit(self, emb_df: pd.DataFrame) -> None:
        self.doc_ids = emb_df["chunk_id"].tolist()
        self.matrix = np.asarray(emb_df["vector"].tolist(), dtype=np.float32)

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.matrix is None or len(self.doc_ids) == 0:
            return []
        q = self.embedder.encode([query_text])[0]
        scores = self.matrix @ q
        idx = np.argsort(-scores)[:top_k]
        return [(self.doc_ids[int(i)], float(scores[int(i)])) for i in idx]
