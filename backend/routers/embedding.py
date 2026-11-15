"""STEP 2 - see what embeddings look like and how similarity works."""
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from sklearn.decomposition import PCA

from backend.rag.embedder import get_embedder
from backend.rag.retrievers import fit_kmeans
from backend.rag.vector_store import get_index
from backend.routers.schemas import IndexConfig

router = APIRouter(prefix="/embed", tags=["2. Embedding"])


class CompareRequest(BaseModel):
    texts: list[str]


@router.post("/compare")
def compare(req: CompareRequest):
    """Similarity matrix of a few sentences under BOTH embedders, side by side."""
    out = {}
    for name in ("tfidf", "minilm"):
        emb = get_embedder(name).fit(req.texts)
        vecs = emb.embed(req.texts)
        out[name] = {
            "dim": emb.dim,
            "preview": [round(float(x), 3) for x in vecs[0][:8]],  # first numbers of vector 1
            "similarity": np.round(vecs @ vecs.T, 3).tolist(),
        }
    return out


class MapRequest(IndexConfig):
    query: str | None = None
    n_clusters: int = 5


@router.post("/map")
def embedding_map(req: MapRequest):
    """Project every chunk vector to 2D (PCA) so we can plot it.
    Points are coloured by k-means cluster; the query is plotted too."""
    store = get_index(req.doc_strategy, req.sql_strategy, req.embedder)
    km = fit_kmeans(store, req.n_clusters)
    pca = PCA(n_components=2, random_state=42).fit(store.vectors)
    pts = pca.transform(store.vectors)
    result = {
        "dim": store.vectors.shape[1],
        "points": [{"x": float(x), "y": float(y), "cluster": int(km.labels_[i]),
                    "id": store.chunks[i].id, "source": store.chunks[i].source,
                    "text": store.chunks[i].text}
                   for i, (x, y) in enumerate(pts)],
        "centroids": pca.transform(km.cluster_centers_).tolist(),
        "query": None,
    }
    if req.query:
        q = store.embed_query(req.query)
        qx, qy = pca.transform(q[None, :])[0]
        result["query"] = {"x": float(qx), "y": float(qy), "text": req.query}
    return result
