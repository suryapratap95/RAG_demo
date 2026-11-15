"""STEP 5 - the full RAG pipeline in one call, returning every intermediate step."""
import time

from fastapi import APIRouter

from backend.rag.generator import azure_configured, generate
from backend.rag.rerankers import rerank
from backend.rag.retrievers import retrieve
from backend.rag.vector_store import get_index
from backend.routers.schemas import QueryRequest

router = APIRouter(tags=["5. Ask (full pipeline)"])


@router.post("/ask")
def ask(req: QueryRequest):
    timings = {}

    t = time.perf_counter()
    store = get_index(req.doc_strategy, req.sql_strategy, req.embedder)   # chunk + embed (cached)
    timings["index_ms"] = round((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    retrieved = retrieve(store, req.query, req.retriever, req.top_n)
    timings["retrieve_ms"] = round((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    reranked = rerank(store, req.query, retrieved["results"], req.reranker, req.top_k)
    timings["rerank_ms"] = round((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    generation = generate(req.query, reranked)
    timings["generate_ms"] = round((time.perf_counter() - t) * 1000)

    return {
        "index": {"chunks": len(store.chunks), "dim": int(store.vectors.shape[1])},
        "retrieved": retrieved,
        "reranked": reranked,
        "generation": generation,
        "timings": timings,
    }


@router.get("/status")
def status():
    return {"azure_openai_configured": azure_configured()}
