"""A minimal in-memory vector store: a list of chunks + a numpy matrix.

Real systems use FAISS, Chroma, pgvector, Azure AI Search, ... but the core
idea is exactly this: store vectors, return the closest ones to a query.
"""
from dataclasses import dataclass, field

import numpy as np

from backend.rag.chunkers import Chunk, chunk_docs, chunk_sql
from backend.rag.embedder import get_embedder


@dataclass
class VectorStore:
    chunks: list[Chunk]
    vectors: np.ndarray          # shape (n_chunks, dim), rows are unit length
    embedder: object
    cache: dict = field(default_factory=dict)   # e.g. fitted k-means

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.embed([query])[0]

    def similarities(self, query_vec: np.ndarray) -> np.ndarray:
        """Cosine similarity of the query to every chunk (dot product of unit vectors)."""
        return self.vectors @ query_vec

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        sims = self.similarities(query_vec)
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]


def build_store(chunks: list[Chunk], embedder: str) -> VectorStore:
    """Embed a list of chunks and wrap them in a VectorStore."""
    texts = [c.text for c in chunks]
    emb = get_embedder(embedder).fit(texts)
    return VectorStore(chunks, emb.embed(texts), emb)


_INDEXES: dict[tuple, VectorStore] = {}


def get_index(doc_strategy: str = "paragraph", sql_strategy: str = "row",
              embedder: str = "minilm") -> VectorStore:
    """Build (or reuse) an index for a given chunking + embedding combination."""
    key = (doc_strategy, sql_strategy, embedder)
    if key not in _INDEXES:
        chunks: list[Chunk] = []
        if doc_strategy != "none":
            chunks += chunk_docs(doc_strategy)
        if sql_strategy != "none":
            chunks += chunk_sql(sql_strategy)
        if not chunks:
            raise ValueError("Pick at least one document or SQL chunking strategy.")
        _INDEXES[key] = build_store(chunks, embedder)
    return _INDEXES[key]
