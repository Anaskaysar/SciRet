from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from data.chunker import ChunkConfig, build_chunks
from data.loader import CORDDataLoader
from embeddings.text_embedder import TextEmbedder
from generation.text_generator import TextGenerator
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import reciprocal_rank_fusion
from retrieval.reranker import OverlapReranker


@dataclass
class QueryResult:
    query: str
    answer: str
    sources: List[str]
    debug: Dict[str, object]


class SciRetPipeline:
    """End-to-end local baseline pipeline for Tier 1 testing."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir = Path(root_dir) if root_dir else Path(__file__).resolve().parents[1]
        self.loader = CORDDataLoader(self.root_dir)
        self.embedder = TextEmbedder(dim=384, model_name="hash-baseline-v1")
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(self.embedder)
        self.reranker = OverlapReranker()
        self.generator = TextGenerator()
        self.chunks_df: pd.DataFrame | None = None

    @property
    def chunks_path(self) -> Path:
        return self.root_dir / "1_data" / "processed" / "chunks.parquet"

    @property
    def emb_path(self) -> Path:
        return self.root_dir / "1_data" / "embeddings" / "text_embeddings.parquet"

    @property
    def emb_manifest_path(self) -> Path:
        return self.root_dir / "1_data" / "embeddings" / "embed_manifest.json"

    def build_from_metadata(self, tier_size: int = 1000, seed: int = 42, chunk_cfg: ChunkConfig = ChunkConfig()) -> None:
        metadata = self.loader.load_metadata()
        subset = self.loader.build_tier_subset(metadata, tier_size=tier_size, seed=seed)
        self.loader.save_clean_subset(subset, "papers_clean.parquet")
        chunks = build_chunks(subset, config=chunk_cfg)
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
        chunks.to_parquet(self.chunks_path, index=False)
        self._fit_indexes(chunks)

    def load_index(self) -> None:
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Missing chunk file: {self.chunks_path}")
        chunks = pd.read_parquet(self.chunks_path)
        self._fit_indexes(chunks)

    def _fit_indexes(self, chunks: pd.DataFrame) -> None:
        self.chunks_df = chunks.reset_index(drop=True)
        self.bm25.fit(self.chunks_df["chunk_id"].tolist(), self.chunks_df["chunk_text"].fillna("").tolist())
        emb_df = self.embedder.cache_or_build(self.chunks_df, self.emb_path, self.emb_manifest_path)
        self.dense.fit(emb_df)

    def query(self, query_text: str, top_k_dense: int = 20, top_k_sparse: int = 20, top_k_final: int = 5) -> QueryResult:
        if self.chunks_df is None:
            raise RuntimeError("Pipeline is not initialized. Call load_index() or build_from_metadata().")

        dense_hits = self.dense.query(query_text, top_k=top_k_dense)
        sparse_hits = self.bm25.query(query_text, top_k=top_k_sparse)
        fused = reciprocal_rank_fusion([dense_hits, sparse_hits], top_k=max(top_k_final * 3, 10))

        text_lookup = dict(zip(self.chunks_df["chunk_id"], self.chunks_df["chunk_text"]))
        candidate_triplets = [(doc_id, text_lookup.get(doc_id, ""), score) for doc_id, score in fused]
        reranked = self.reranker.rerank(query_text, candidate_triplets, top_k=top_k_final)

        contexts = [(doc_id, text) for doc_id, text, _ in reranked]
        answer = self.generator.generate(query_text, contexts)
        sources = [doc_id for doc_id, _, _ in reranked]
        debug = {"dense_hits": dense_hits[:5], "sparse_hits": sparse_hits[:5], "fused_hits": fused[:10]}
        return QueryResult(query=query_text, answer=answer, sources=sources, debug=debug)
