"""STEP 3 - RETRIEVAL

Find the top-N chunks most likely to answer the question.

  dense  - cosine similarity between embeddings (meaning).
  bm25   - classic keyword scoring used by search engines (exact words).
  hybrid - combine dense + bm25 rankings with Reciprocal Rank Fusion (RRF).
  kmeans - group chunks into clusters once; at query time search ONLY the
           nearest cluster. Faster on big data (this is the idea behind IVF
           indexes in FAISS), but can miss answers in other clusters.
"""
import math
import re
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans

from backend import config
from backend.rag.vector_store import VectorStore

STOPWORDS = set("a an the is are was of to in on for and or with by at it this that i my me can do does what how".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def _result(store: VectorStore, idx: int, score: float, **extra) -> dict:
    idx = int(idx)  # numpy ints are not JSON-serialisable
    return {"chunk": store.chunks[idx].to_dict(), "index": idx, "score": round(float(score), 4), **extra}


def dense(store: VectorStore, query: str, top_n: int = config.TOP_N) -> dict:
    hits = store.search(store.embed_query(query), top_n)
    return {"results": [_result(store, i, s) for i, s in hits]}


def bm25_scores(store: VectorStore, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """BM25 in ~15 lines. Rewards rare query words that appear often in a short chunk."""
    docs = [tokenize(c.text) for c in store.chunks]
    avg_len = sum(map(len, docs)) / len(docs)
    df = Counter(t for d in docs for t in set(d))          # how many chunks contain term
    n = len(docs)
    scores = np.zeros(n)
    for q in set(tokenize(query)):
        idf = math.log(1 + (n - df[q] + 0.5) / (df[q] + 0.5))
        for i, d in enumerate(docs):
            tf = d.count(q)
            if tf:
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))
    return scores


def bm25(store: VectorStore, query: str, top_n: int = config.TOP_N) -> dict:
    scores = bm25_scores(store, query)
    top = [i for i in np.argsort(-scores)[:top_n] if scores[i] > 0]
    return {"results": [_result(store, i, scores[i]) for i in top]}


def hybrid(store: VectorStore, query: str, top_n: int = config.TOP_N, rrf_k: int = 60) -> dict:
    """Reciprocal Rank Fusion: score = sum(1 / (rrf_k + rank)) across both lists.
    Uses ranks, not raw scores, so the two very different scales don't matter."""
    dense_rank = [i for i, _ in store.search(store.embed_query(query), len(store.chunks))]
    bm = bm25_scores(store, query)
    bm_rank = [i for i in np.argsort(-bm) if bm[i] > 0]
    fused: dict[int, float] = {}
    for ranking in (dense_rank, bm_rank):
        for rank, i in enumerate(ranking, start=1):
            fused[i] = fused.get(i, 0) + 1 / (rrf_k + rank)
    top = sorted(fused, key=fused.get, reverse=True)[:top_n]
    return {"results": [_result(store, i, fused[i],
                                dense_rank=dense_rank.index(i) + 1,
                                bm25_rank=(bm_rank.index(i) + 1) if i in bm_rank else None)
                        for i in top]}


def fit_kmeans(store: VectorStore, n_clusters: int = config.KMEANS_CLUSTERS) -> KMeans:
    key = f"kmeans_{n_clusters}"
    if key not in store.cache:
        n_clusters = min(n_clusters, len(store.chunks))
        store.cache[key] = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(store.vectors)
    return store.cache[key]


def kmeans(store: VectorStore, query: str, top_n: int = config.TOP_N,
           n_clusters: int = config.KMEANS_CLUSTERS) -> dict:
    km = fit_kmeans(store, n_clusters)
    q = store.embed_query(query)
    centroid_sims = km.cluster_centers_ @ q
    best = int(np.argmax(centroid_sims))
    members = np.where(km.labels_ == best)[0]
    sims = store.vectors[members] @ q
    order = members[np.argsort(-sims)][:top_n]
    return {
        "results": [_result(store, i, store.vectors[i] @ q, cluster=best) for i in order],
        "info": {
            "chosen_cluster": best,
            "cluster_sizes": np.bincount(km.labels_).tolist(),
            "centroid_similarity": [round(float(s), 4) for s in centroid_sims],
            "searched_chunks": len(members),
            "total_chunks": len(store.chunks),
        },
    }


RETRIEVERS = {"dense": dense, "bm25": bm25, "hybrid": hybrid, "kmeans": kmeans}


def retrieve(store: VectorStore, query: str, method: str = "dense", top_n: int = config.TOP_N) -> dict:
    out = RETRIEVERS[method](store, query, top_n)
    out.setdefault("info", {})
    out["method"] = method
    return out
