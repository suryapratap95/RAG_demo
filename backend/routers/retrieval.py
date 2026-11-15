"""STEP 3 - retrieve the top-N candidate chunks for a question."""
from fastapi import APIRouter

from backend.rag.retrievers import retrieve
from backend.rag.vector_store import get_index
from backend.routers.schemas import QueryRequest

router = APIRouter(prefix="/retrieve", tags=["3. Retrieval"])


@router.post("")
def retrieve_chunks(req: QueryRequest):
    store = get_index(req.doc_strategy, req.sql_strategy, req.embedder)
    return retrieve(store, req.query, req.retriever, req.top_n)


@router.post("/compare")
def compare_retrievers(req: QueryRequest):
    """Run all four retrievers on the same question."""
    store = get_index(req.doc_strategy, req.sql_strategy, req.embedder)
    return {m: retrieve(store, req.query, m, req.top_n) for m in ("dense", "bm25", "hybrid", "kmeans")}
