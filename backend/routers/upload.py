"""STEP 6 - RAG over your own document (PDF, Word, text)."""
import time
from typing import Literal

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

from backend import config
from backend.rag import uploads
from backend.rag.generator import DOCUMENT_SYSTEM_PROMPT, generate
from backend.rag.rerankers import rerank
from backend.rag.retrievers import retrieve
from backend.routers.schemas import Embedder, Reranker, Retriever

router = APIRouter(prefix="/upload", tags=["6. Your document"])


@router.post("")
async def upload_file(file: UploadFile = File(...)):
    """Upload a .pdf, .docx, .txt or .md file. Returns a doc_id to ask questions with."""
    data = await file.read()
    return uploads.add_upload(file.filename or "upload", data)


@router.get("")
def list_files():
    return uploads.list_uploads()


@router.delete("/{doc_id}")
def delete_file(doc_id: str):
    uploads.remove_upload(doc_id)
    return {"deleted": doc_id}


class DocConfig(BaseModel):
    doc_id: str
    strategy: Literal["paragraph", "sentence", "fixed_size"] = "paragraph"
    size: int = 500
    overlap: int = 100
    embedder: Embedder = "minilm"


class DocAsk(DocConfig):
    query: str
    retriever: Retriever = "hybrid"
    reranker: Reranker = "cross_encoder"
    top_n: int = config.TOP_N
    top_k: int = 4


def _store(c: DocConfig):
    return uploads.get_upload_store(c.doc_id, c.strategy, c.size, c.overlap, c.embedder)


@router.post("/chunks")
def chunks(c: DocConfig):
    """See how your document was chunked."""
    store = _store(c)
    lengths = [len(ch.text) for ch in store.chunks]
    return {"count": len(lengths), "avg_chars": round(sum(lengths) / len(lengths)),
            "chunks": [ch.to_dict() for ch in store.chunks]}


@router.post("/ask")
def ask(req: DocAsk):
    timings = {}
    t = time.perf_counter()
    store = _store(req)
    timings["index_ms"] = round((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    retrieved = retrieve(store, req.query, req.retriever, req.top_n)
    timings["retrieve_ms"] = round((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    reranked = rerank(store, req.query, retrieved["results"], req.reranker, req.top_k)
    timings["rerank_ms"] = round((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    generation = generate(req.query, reranked, DOCUMENT_SYSTEM_PROMPT)
    timings["generate_ms"] = round((time.perf_counter() - t) * 1000)

    return {"index": {"chunks": len(store.chunks), "dim": int(store.vectors.shape[1])},
            "retrieved": retrieved, "reranked": reranked,
            "generation": generation, "timings": timings}
