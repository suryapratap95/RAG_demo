"""Raw source data, before any processing."""
from fastapi import APIRouter

from backend.rag.chunkers import load_docs, load_tables

router = APIRouter(prefix="/ingest", tags=["0. Ingest"])


@router.get("/docs")
def raw_docs():
    """The markdown documents exactly as stored on disk."""
    return [{"name": n, "text": t, "chars": len(t)} for n, t in load_docs().items()]


@router.get("/sql")
def raw_tables():
    """The SQLite tables (products, customers, orders)."""
    return [{"table": t, "rows": rows} for t, rows in load_tables().items()]
