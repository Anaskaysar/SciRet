from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


class TextEmbedder:
    """
    Lightweight deterministic embedder for local testing.
    Replace internals with SentenceTransformer/BGE for final experiments.
    """

    def __init__(self, dim: int = 384, model_name: str = "hash-baseline-v1") -> None:
        self.dim = dim
        self.model_name = model_name

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return np.vstack([self._embed_one(t) for t in texts])

    def cache_or_build(
        self,
        chunks_df: pd.DataFrame,
        cache_path: Path,
        manifest_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        vectors = self.encode(chunks_df["chunk_text"].fillna("").tolist())
        out = pd.DataFrame(
            {
                "chunk_id": chunks_df["chunk_id"].tolist(),
                "vector": vectors.tolist(),
            }
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path, index=False)

        if manifest_path is not None:
            manifest = {"model_name": self.model_name, "dim": self.dim, "count": int(len(out))}
            manifest_path.write_text(json.dumps(manifest, indent=2))
        return out
