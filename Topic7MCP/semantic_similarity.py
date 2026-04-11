"""Fast semantic similarity (cosine) using an open embedding model with hash-based caching.

This module provides a single function `semantic_cosine_similarity(a, b)` which returns
the cosine similarity between two strings.

The embedding model is loaded lazily on first use, and embeddings are cached in a
fixed-size LRU cache (max 10,000 entries) keyed by a stable hash of the input string.

Dependencies:
  pip install sentence-transformers numpy

Example:
    from semantic_similarity import semantic_cosine_similarity
    score = semantic_cosine_similarity("hello world", "hi there")
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Dict

import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Choose a fast open embedding model. This is a small, fast model with good quality.
_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Cache size limit (number of unique strings cached).
_CACHE_MAX_SIZE = 10_000


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _stable_hash(text: str) -> str:
    """Return a stable hash for a string (used as cache key)."""
    # We use SHA-256 for stability and to avoid depending on Python's hash randomization.
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


class _EmbeddingCache:
    """A simple thread-safe LRU cache for string embeddings."""

    def __init__(self, max_size: int):
        self._max_size = max_size
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            value = self._data.get(key)
            if value is not None:
                # Move to end to mark as recently used
                self._data.move_to_end(key)
            return value

    def set(self, key: str, value: np.ndarray) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._data[key] = value
                return
            self._data[key] = value
            if len(self._data) > self._max_size:
                # Drop the oldest entry
                self._data.popitem(last=False)


# Single shared cache instance.
_embedding_cache = _EmbeddingCache(max_size=_CACHE_MAX_SIZE)

# Lazy-loaded model (initialized on first call).
_model_lock = threading.Lock()
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _model


def _get_embedding(text: str) -> np.ndarray:
    """Get an embedding for a string, using the cache if available."""
    key = _stable_hash(text)
    cached = _embedding_cache.get(key)
    if cached is not None:
        return cached

    model = _get_model()
    emb = model.encode([text], convert_to_numpy=True)[0]
    # Normalize to unit length to make cosine similarity computation faster.
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    _embedding_cache.set(key, emb)
    return emb


def semantic_cosine_similarity(a: str, b: str) -> float:
    """Return cosine similarity between the embeddings of two strings.

    The result is in [-1, 1].
    """
    if a is None or b is None:
        raise ValueError("Both inputs must be non-None strings")

    emb_a = _get_embedding(a)
    emb_b = _get_embedding(b)

    # Since embeddings are normalized, cosine similarity reduces to dot product.
    return float(np.dot(emb_a, emb_b))


if __name__ == "__main__":
    # Quick sanity check
    import argparse

    parser = argparse.ArgumentParser(description="Compute semantic cosine similarity between two strings.")
    parser.add_argument("a", help="First string")
    parser.add_argument("b", help="Second string")
    args = parser.parse_args()

    score = semantic_cosine_similarity(args.a, args.b)
    print(f"cosine similarity = {score:.6f}")
