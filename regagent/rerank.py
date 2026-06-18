"""Reranking — the precision pass after fusion.

Hybrid fusion gives good recall (the right article is *somewhere* in the top-k).
A reranker then re-scores each candidate against the question for precision, so
the article the answer is actually built on sits at the top. In production this
is a cross-encoder (e.g. Cohere Rerank / bge-reranker); here a lexical
cross-scorer that needs no model — same interface, swap later.
"""
from __future__ import annotations

import re
from collections import Counter

from .store import Chunk

_STOP = set("the a an of to in for and or is are be on that this with as by it "
            "from at which such not no any may shall will".split())


def _terms(text: str) -> Counter:
    return Counter(w for w in re.findall(r"[a-zA-Z0-9]{3,}", text.lower())
                   if w not in _STOP)


def _relevance(question: str, chunk: Chunk) -> float:
    """Cross-score: weighted term overlap of question against chunk text+source."""
    q = _terms(question)
    if not q:
        return 0.0
    d = _terms(chunk.text + " " + chunk.source)
    overlap = sum(min(q[t], d[t]) for t in q if t in d)
    # normalise by question length; reward exact source/article hits
    score = overlap / sum(q.values())
    if any(t in chunk.source.lower() for t in q):
        score += 0.25
    return round(score, 4)


def rerank(question: str, hits: list[tuple[Chunk, float]], top: int = 4
           ) -> list[tuple[Chunk, float]]:
    """Re-score fused candidates by relevance to the question; return best `top`."""
    rescored = [(c, _relevance(question, c)) for c, _ in hits]
    rescored.sort(key=lambda x: -x[1])
    return rescored[:top]
