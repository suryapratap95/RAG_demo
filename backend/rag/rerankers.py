"""STEP 4 - RE-RANKING

Retrieval is fast but rough. Re-ranking takes the top-N candidates and picks
the best top-K to send to the LLM.

  none          - keep retrieval order.
  mmr           - Maximal Marginal Relevance: balance relevance to the query
                  against similarity to chunks already picked (avoid duplicates).
  kmeans        - cluster the candidates into K groups and keep the best chunk
                  from each group, so every "topic" is represented once.
  cross_encoder - a model reads (question, chunk) TOGETHER and scores the pair.
                  Much more accurate than comparing two separate embeddings,
                  but too slow to run on the whole corpus - hence only top-N.
"""
from functools import lru_cache

import numpy as np
from sklearn.cluster import KMeans

from backend import config
from backend.rag.vector_store import VectorStore


def _with(c: dict, rank_score: float, **extra) -> dict:
    return {**c, "rerank_score": round(float(rank_score), 4), **extra}


def no_rerank(store, query, candidates, top_k):
    return [_with(c, c["score"]) for c in candidates[:top_k]]


def mmr(store: VectorStore, query: str, candidates: list[dict], top_k: int, lam: float = 0.6) -> list[dict]:
    """score = lam * sim(query, chunk) - (1 - lam) * max sim(chunk, already selected)"""
    q = store.embed_query(query)
    vecs = {c["index"]: store.vectors[c["index"]] for c in candidates}
    remaining = list(candidates)
    selected: list[dict] = []
    while remaining and len(selected) < top_k:
        def mmr_score(c):
            rel = float(vecs[c["index"]] @ q)
            red = max((float(vecs[c["index"]] @ vecs[s["index"]]) for s in selected), default=0.0)
            return lam * rel - (1 - lam) * red, rel, red
        best = max(remaining, key=lambda c: mmr_score(c)[0])
        score, rel, red = mmr_score(best)
        selected.append(_with(best, score, relevance=round(rel, 4), redundancy=round(red, 4)))
        remaining.remove(best)
    return selected


def kmeans_diversity(store: VectorStore, query: str, candidates: list[dict], top_k: int) -> list[dict]:
    if len(candidates) <= top_k:
        return no_rerank(store, query, candidates, top_k)
    q = store.embed_query(query)
    vecs = np.array([store.vectors[c["index"]] for c in candidates])
    labels = KMeans(n_clusters=top_k, n_init=10, random_state=42).fit_predict(vecs)
    sims = vecs @ q
    picked = []
    for cl in range(top_k):
        members = np.where(labels == cl)[0]
        best = members[np.argmax(sims[members])]
        picked.append(_with(candidates[best], sims[best], cluster=cl, cluster_size=len(members)))
    return sorted(picked, key=lambda c: -c["rerank_score"])


@lru_cache(maxsize=1)
def _load_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.CROSS_ENCODER_MODEL)


def cross_encoder(store: VectorStore, query: str, candidates: list[dict], top_k: int) -> list[dict]:
    if not candidates:
        return []
    scores = _load_cross_encoder().predict([(query, c["chunk"]["text"]) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])[:top_k]
    return [_with(c, s) for c, s in ranked]


RERANKERS = {"none": no_rerank, "mmr": mmr, "kmeans": kmeans_diversity, "cross_encoder": cross_encoder}


def rerank(store: VectorStore, query: str, candidates: list[dict], method: str = "mmr",
           top_k: int = config.TOP_K) -> list[dict]:
    return RERANKERS[method](store, query, candidates, top_k)
