"""STEP 4 - re-rank retrieval candidates down to top-K."""
from fastapi import APIRouter

from backend.rag.rerankers import rerank
from backend.rag.retrievers import retrieve
from backend.rag.vector_store import get_index
from backend.routers.schemas import QueryRequest

router = APIRouter(prefix="/rerank", tags=["4. Re-ranking"])


@router.post("")
def rerank_chunks(req: QueryRequest):
    """Returns the retrieval list (before) and the re-ranked list (after)."""
    store = get_index(req.doc_strategy, req.sql_strategy, req.embedder)
    before = retrieve(store, req.query, req.retriever, req.top_n)["results"]
    after = rerank(store, req.query, before, req.reranker, req.top_k)
    return {"before": before, "after": after, "reranker": req.reranker}
