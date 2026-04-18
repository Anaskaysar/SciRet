from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd


@dataclass
class ChunkConfig:
    chunk_size: int = 400
    overlap: int = 50
    min_tokens: int = 20


def _chunk_words(words: List[str], chunk_size: int, overlap: int) -> Iterable[List[str]]:
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        yield words[start : start + chunk_size]


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50, min_tokens: int = 20) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    for w in _chunk_words(words, chunk_size, overlap):
        if len(w) >= min_tokens:
            chunks.append(" ".join(w))
    return chunks


def build_chunks(df: pd.DataFrame, config: ChunkConfig = ChunkConfig()) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        cord_uid = str(row.get("cord_uid", "unknown"))
        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))
        text = f"{title} {abstract}".strip()
        for idx, chunk in enumerate(chunk_text(text, config.chunk_size, config.overlap, config.min_tokens)):
            records.append(
                {
                    "chunk_id": f"{cord_uid}_chunk_{idx:03d}",
                    "cord_uid": cord_uid,
                    "title": title,
                    "chunk_text": chunk,
                    "chunk_index": idx,
                }
            )
    return pd.DataFrame(records)
