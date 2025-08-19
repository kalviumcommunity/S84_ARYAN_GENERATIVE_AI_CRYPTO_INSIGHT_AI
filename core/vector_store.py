"""
core/vector_store.py
====================
Minimal in-memory vector store for demos. Stores embeddings with metadata and
returns cosine-similar results.
"""

from typing import List, Dict, Optional
import math

_store: List[Dict] = []


def clear() -> None:
    _store.clear()


def add_document_embedding(embedding: List[float], metadata: Dict) -> None:
    _store.append({"embedding": embedding, "metadata": metadata})


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search_similar(query_embedding: List[float], top_k: int = 3) -> List[Dict]:
    scored: List[Dict] = []
    for item in _store:
        score = _cosine_similarity(query_embedding, item["embedding"])  # type: ignore[index]
        scored.append({"score": float(score), "metadata": item["metadata"]})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[: max(top_k, 0)]


