"""RAG demo - FastAPI entry point.

Run from the project folder:
    uvicorn backend.main:app --reload
Then open http://127.0.0.1:8000  (API docs: http://127.0.0.1:8000/docs)
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.data.seed_db import seed
from backend.routers import chat, chunking, embedding, ingest, rerank, retrieval, upload

if not config.DB_PATH.exists():
    seed(config.DB_PATH)

app = FastAPI(title="RAG Demo", description="Chunking, embedding, retrieval, re-ranking, generation.")

for r in (ingest, chunking, embedding, retrieval, rerank, chat, upload):
    app.include_router(r.router, prefix="/api")


@app.exception_handler(ValueError)
@app.exception_handler(KeyError)
async def bad_input(_: Request, exc: Exception):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# Serve the HTML frontend at "/" (mounted last so /api and /docs win)
app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
