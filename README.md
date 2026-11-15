# RAG Demo

A small training app that shows each step of Retrieval-Augmented Generation:
**chunk → embed → retrieve → re-rank → generate**. Each step has its own API
router and its own tab in the browser. See [ARCHITECTURE.md](ARCHITECTURE.md).

## Setup

Needs Python 3.10–3.13 (PyTorch for MiniLM may not support 3.14 yet).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # optional: add Azure AI Foundry keys
uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000. The API docs are at http://127.0.0.1:8000/docs.

The first MiniLM / cross-encoder use downloads two small models (~90 MB + ~90 MB)
from Hugging Face. Run the app once before the training session so they are cached.

Without Azure keys everything works except the final answer: the Ask tab shows
the exact prompt that would have been sent.

## Suggested training walkthrough (≈45 min)

1. **Data tab** – two sources: policy documents and SQL tables.
2. **Chunking tab**
   - Docs → `fixed_size` with size 100: point out sentences cut in half. Raise overlap.
   - Docs → `paragraph`: clean, self-contained chunks with the heading as context.
   - SQL → `row` vs `entity`: "what did Asha buy?" needs the JOINed entity chunk.
   - SQL → `schema`: what a text-to-SQL system retrieves.
3. **Embedding tab**
   - Compare: in TF-IDF, "return my headphones" vs "send back something" ≈ 0.
     In MiniLM they are clearly similar. That is the point of semantic embeddings.
   - Map: draw clusters, add a question, see which cluster it lands near.
4. **Retrieval tab** – query "Can I send back headphones I don't like?"
   - Header Embedder = MiniLM: dense finds the return policy even though it never says "send back".
   - Switch to TF-IDF: dense returns payments / yoga mat; kmeans searches only the
     product cluster and misses the policy entirely. BM25 only matches the word "headphones".
   - Hybrid merges dense + BM25 rankings (see the rank columns on each card).
   - Query "What did Asha buy?": with SQL chunking `row` the answer is spread across rows
     that only hold IDs; switch to `entity` and one chunk answers it.
5. **Re-ranking tab** – "Is accidental damage covered?" with retriever `hybrid`
   - `none`: basic warranty coverage is #1 (it says accidental damage is *not* covered).
   - `cross_encoder`: moves *Extended warranty* (which *does* cover it) to #1.
   - `mmr` / `kmeans`: add variety, but can pull in less relevant chunks. Diversity is a trade-off.
6. **Ask tab** – full pipeline with timings and the final prompt.
7. **Your document tab** – upload a PDF, Word (.docx), .txt or .md file and ask questions about it.
   - Try `samples/travel_policy.pdf` ("What is the hotel limit in London?") and
     `samples/thermostat_guide.docx` ("What does error E3 mean?", "What is the price?" → "I don't know").
   - PDFs are read page by page, so citations show the page (`travel_policy.pdf p.2`).
   - Word headings become sections, so `paragraph` chunking follows the document's structure.
   - "Show chunks" shows how the file was split; change the chunking and ask again.
   - Limits: 20 MB; old `.doc` files must be re-saved as `.docx`; scanned PDFs need OCR.
     Uploads are kept in memory only and are cleared when the server restarts.

## Project layout

```
backend/
  main.py            FastAPI app, mounts routers, serves frontend/
  config.py          settings and Azure env vars
  routers/           one router per RAG step (ingest, chunking, embedding, retrieval, rerank, chat, upload)
  rag/               the actual logic, plain Python
  data/docs/         sample markdown docs
  data/seed_db.py    builds sample.db (runs automatically on first start)
frontend/            index.html, app.js, style.css (no build step)
samples/             a PDF and a Word file to try in the "Your document" tab
```

## API

| Method | Path | Step |
|---|---|---|
| GET | `/api/ingest/docs`, `/api/ingest/sql` | raw data |
| POST | `/api/chunk` | chunking |
| POST | `/api/embed/compare`, `/api/embed/map` | embedding |
| POST | `/api/retrieve`, `/api/retrieve/compare` | retrieval |
| POST | `/api/rerank` | re-ranking |
| POST | `/api/ask` | full pipeline |
| POST / GET / DELETE | `/api/upload`, `/api/upload/{doc_id}` | upload, list, remove your documents |
| POST | `/api/upload/chunks`, `/api/upload/ask` | chunk / ask your document |
| GET | `/api/status` | Azure configured? |
