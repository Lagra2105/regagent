"""Sparse (lexical) retrieval — the third leg of hybrid search.

Dense retrieval catches meaning; sparse catches exact terms. In regulation that
matters: "Article 5", "high-risk", "conformity assessment", "biometric" must
match literally — a paraphrase-tolerant vector search can drift. A BM25-style
ranker complements dense, and fusing the two beats either alone.

Pure-stdlib BM25 so it runs anywhere; same interface as DocStore.search.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .store import Chunk


def _tok(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class BM25Index:
    """Classic Okapi BM25."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self.docs: list[list[str]] = []
        self.df: dict[str, int] = defaultdict(int)
        self.avgdl = 0.0
        self.N = 0

    def build(self, chunks: list[Chunk]) -> "BM25Index":
        self.chunks = chunks
        self.docs = [_tok(c.text + " " + c.source) for c in chunks]
        self.N = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.N) if self.N else 0.0
        self.df.clear()
        for d in self.docs:
            for term in set(d):
                self.df[term] += 1
        return self

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        # BM25+ idf, always positive
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        q = _tok(query)
        scores: list[float] = []
        for d in self.docs:
            tf = Counter(d)
            dl = len(d) or 1
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = self._idf(term)
                num = tf[term] * (self.k1 + 1)
                den = tf[term] + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += idf * num / den
            scores.append(s)
        ranked = sorted(zip(self.chunks, scores), key=lambda x: -x[1])
        return [(c, s) for c, s in ranked[:k] if s > 0]
