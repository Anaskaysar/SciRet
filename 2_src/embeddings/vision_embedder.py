from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


class VisionEmbedder:
    """
    Deterministic stub for figure embeddings.
    Replace with CLIP in multimodal experiments.
    """

    def __init__(self, dim: int = 512, model_name: str = "clip-stub-v1") -> None:
        self.dim = dim
        self.model_name = model_name

    def encode_asset_ids(self, asset_ids: Iterable[str]) -> np.ndarray:
        rows = []
        for aid in asset_ids:
            vec = np.zeros(self.dim, dtype=np.float32)
            h = int(hashlib.sha256(aid.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] = 1.0
            rows.append(vec)
        return np.vstack(rows) if rows else np.empty((0, self.dim), dtype=np.float32)
