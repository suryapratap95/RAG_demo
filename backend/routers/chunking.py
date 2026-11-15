"""STEP 1 - see how each chunking strategy splits the same data."""
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from backend import config
from backend.rag.chunkers import chunk_docs, chunk_sql

router = APIRouter(prefix="/chunk", tags=["1. Chunking"])


class ChunkRequest(BaseModel):
    source: Literal["docs", "sql"] = "docs"
    strategy: str = "paragraph"
    size: int = config.FIXED_CHUNK_SIZE          # fixed_size only
    overlap: int = config.FIXED_CHUNK_OVERLAP    # fixed_size only
    sentences: int = config.SENTENCES_PER_CHUNK  # sentence only


@router.post("")
def chunk(req: ChunkRequest):
    if req.source == "sql":
        chunks = chunk_sql(req.strategy)
    elif req.strategy == "fixed_size":
        chunks = chunk_docs("fixed_size", size=req.size, overlap=req.overlap)
    elif req.strategy == "sentence":
        chunks = chunk_docs("sentence", n=req.sentences)
    else:
        chunks = chunk_docs(req.strategy)
    lengths = [len(c.text) for c in chunks]
    return {
        "count": len(chunks),
        "avg_chars": round(sum(lengths) / len(lengths)) if lengths else 0,
        "min_chars": min(lengths, default=0),
        "max_chars": max(lengths, default=0),
        "chunks": [c.to_dict() for c in chunks],
    }
