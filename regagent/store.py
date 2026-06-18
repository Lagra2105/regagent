"""Document store with hybrid retrieval and provenance.

Phase 1: dense (vector) retrieval over chunked regulation text, each chunk
carrying its source location (article/recital) so every answer can be cited.
Phase 2 will add a knowledge-graph layer (Neo4j) for structured retrieval.

Embeddings come from OpenAI when OPENAI_API_KEY is set; otherwise a deterministic
local hash-embedding is used so the whole pipeline runs offline for development.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    source: str        # e.g. "AI Act — Article 5(1)(a)"
    embedding: list[float] | None = None


def _hash_embedding(text: str, dim: int = 256) -> list[float]:
    """Deterministic offline embedding: hashed bag-of-words. Good enough to wire
    the pipeline end-to-end without an API key; swapped for real embeddings later."""
    vec = [0.0] * dim
    for tok in re.findall(r"[a-zA-Z]{3,}", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch. Uses OpenAI text-embedding-3-small if a key is present."""
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [d.embedding for d in resp.data]
    return [_hash_embedding(t) for t in texts]


def _cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class DocStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        texts = [c.text for c in chunks]
        for c, e in zip(chunks, embed(texts)):
            c.embedding = e
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        qe = embed([query])[0]
        scored = [(c, _cosine(qe, c.embedding or [])) for c in self.chunks]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
