"""STEP 2 - EMBEDDING

Turn text into a vector of numbers so that "similar meaning" becomes
"vectors pointing in a similar direction". All vectors are L2-normalised,
so cosine similarity is simply a dot product.

  tfidf  - keyword vectors. One dimension per word in the vocabulary.
           Fast, no model, but 'send back' does NOT match 'return'.
  minilm - a small neural model (384 dimensions) that captures meaning.
"""
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from backend import config


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        """TF-IDF must learn its vocabulary from the corpus first."""
        self.vectorizer.fit(texts)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(self.vectorizer.transform(texts).toarray())

    @property
    def dim(self) -> int:
        return len(self.vectorizer.vocabulary_)


@lru_cache(maxsize=1)
def _load_minilm():
    from sentence_transformers import SentenceTransformer  # heavy import, load lazily
    return SentenceTransformer(config.MINILM_MODEL)


class MiniLMEmbedder:
    name = "minilm"

    def fit(self, texts: list[str]) -> "MiniLMEmbedder":
        return self  # pre-trained: nothing to learn

    def embed(self, texts: list[str]) -> np.ndarray:
        return _load_minilm().encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    @property
    def dim(self) -> int:
        return 384


EMBEDDERS = {"tfidf": TfidfEmbedder, "minilm": MiniLMEmbedder}


def get_embedder(name: str):
    return EMBEDDERS[name]()
