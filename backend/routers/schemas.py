"""Request bodies shared by the routers."""
from typing import Literal

from pydantic import BaseModel

from backend import config

DocStrategy = Literal["fixed_size", "sentence", "paragraph", "none"]
SqlStrategy = Literal["row", "entity", "schema", "none"]
Embedder = Literal["tfidf", "minilm"]
Retriever = Literal["dense", "bm25", "hybrid", "kmeans"]
Reranker = Literal["none", "mmr", "kmeans", "cross_encoder"]


class IndexConfig(BaseModel):
    """Which chunking + embedding combination to search over."""
    doc_strategy: DocStrategy = "paragraph"
    sql_strategy: SqlStrategy = "row"
    embedder: Embedder = "minilm"


class QueryRequest(IndexConfig):
    query: str
    retriever: Retriever = "dense"
    top_n: int = config.TOP_N
    reranker: Reranker = "mmr"
    top_k: int = config.TOP_K
