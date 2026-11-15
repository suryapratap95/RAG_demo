# RAG Demo — Architecture (draft for review)

A small, readable RAG app for training. Every stage of the pipeline is its own
API endpoint, so the frontend can show what happens at each step, not just the
final answer.

## Pipeline

```
           ┌──────────── INGEST ────────────┐
 Docs (.md/.txt) ──► Doc chunkers ─┐
                                   ├─► Chunks ─► Embedder ─► Vector store (in-memory numpy)
 SQLite tables ───► SQL chunkers ──┘                               │
                                                                   ▼
 Question ─► Embed query ─► RETRIEVE (top-N) ─► RE-RANK (top-k) ─► GENERATE answer
                             • cosine (dense)     • MMR (diversity)       (with cited chunks)
                             • BM25 (keyword)     • K-means diversity
                             • hybrid (RRF)       • cross-encoder score
                             • K-means cluster
                               search (IVF-style)
```

## Folder layout

```
RAG-demo/
├── backend/
│   ├── main.py               # FastAPI app: mounts routers, serves frontend/
│   ├── config.py             # chunk sizes, top-k, model names
│   ├── routers/              # one router per RAG concept
│   │   ├── schemas.py        # shared request models
│   │   ├── ingest.py         # GET  /api/ingest/docs, /api/ingest/sql – raw data
│   │   ├── chunking.py       # POST /api/chunk          – compare chunking strategies
│   │   ├── embedding.py      # POST /api/embed/compare, /api/embed/map – similarity + 2D map
│   │   ├── retrieval.py      # POST /api/retrieve(/compare) – dense | bm25 | hybrid | kmeans
│   │   ├── rerank.py         # POST /api/rerank         – mmr | kmeans | cross_encoder | none
│   │   ├── chat.py           # POST /api/ask            – full pipeline, returns every step
│   │   └── upload.py         # /api/upload(/chunks|/ask) – RAG over an uploaded PDF / Word file
│   ├── rag/                  # plain-Python logic, no web code
│   │   ├── chunkers.py       # docs: fixed-size+overlap, sentence, paragraph/heading
│   │   │                     # sql:  row-per-chunk, entity-grouped (JOIN), schema chunk
│   │   ├── embedder.py
│   │   ├── vector_store.py
│   │   ├── retrievers.py
│   │   ├── rerankers.py
│   │   ├── generator.py
│   │   └── uploads.py        # PDF (pypdf) / Word (python-docx) text extraction, per-upload index
│   └── data/
│       ├── docs/             # 4–5 short sample documents
│       └── seed_db.py        # creates sample.db (SQLite)
├── frontend/
│   ├── index.html            # tabs: Data │ Chunking │ Embedding │ Retrieval │ Re-rank │ Ask │ Your document
│   ├── app.js                # vanilla JS, fetch() calls to the API
│   └── style.css
├── requirements.txt
└── README.md                 # setup + a suggested training walkthrough
```

## Concepts and where they live

| Concept | Approaches shown | File |
|---|---|---|
| Doc chunking | fixed-size with overlap, sentence-based, paragraph/heading-aware | `rag/chunkers.py` |
| SQL chunking | one row = one chunk (row → sentence), entity-grouped (customer + their orders via JOIN), table-schema chunk | `rag/chunkers.py` |
| Embedding | text → vector, cosine similarity, 2D scatter of chunks | `rag/embedder.py` |
| Retrieval | dense cosine, BM25 keyword, hybrid (reciprocal rank fusion), K-means cluster search (search only the nearest cluster) | `rag/retrievers.py` |
| Re-ranking | MMR (relevance vs. novelty), K-means diversity (best chunk per cluster), cross-encoder relevance score | `rag/rerankers.py` |
| Generation | prompt built from top-k chunks, answer with chunk citations | `rag/generator.py` |

## Decisions (confirmed)

- **Sample data: online store.** Docs: return policy, shipping, warranty, payments FAQ.
  SQLite tables: `products`, `customers`, `orders`.
- **Embeddings: switchable in the UI.**
  - `tfidf` — scikit-learn TF-IDF (keyword vectors, no download).
  - `minilm` — sentence-transformers `all-MiniLM-L6-v2` (semantic, ~90MB one-time download).
  - Demo moment: ask "Can I send back my headphones?" and compare — TF-IDF misses
    "return policy", MiniLM finds it.
- **Generation: Azure AI Foundry, OpenAI model deployment** (e.g. `gpt-4o-mini`)
  via the `openai` SDK's `AzureOpenAI` client. Configured in `.env`:
  `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`,
  `AZURE_OPENAI_API_VERSION`. If these are not set, `/ask` still runs and returns
  the exact prompt it would have sent, so the demo works offline.
- **Cross-encoder re-ranker** uses `cross-encoder/ms-marco-MiniLM-L-6-v2`
  (same sentence-transformers library, small download).

## Design choices (kept simple on purpose)

- **FastAPI** — routers map 1:1 to concepts; auto docs at `/docs` are useful in a demo.
- **In-memory numpy vector store** — no database to install; the cosine math is visible in ~20 lines.
- **SQLite** — ships with Python; no server.
- **scikit-learn** for K-means and the 2D projection (PCA).
- **No build step** for the frontend — open the page served by FastAPI.
