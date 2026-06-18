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


def _matches(qt: str, d: Counter) -> int:
    """Count occurrences of a query term in a doc, with prefix matching so
    'interact' also hits 'interacting' and 'oblig' hits 'obligations'."""
    if qt in d:
        return d[qt]
    stem = qt[:5]
    return sum(n for w, n in d.items() if w.startswith(stem) or qt.startswith(w[:5]))


def _relevance(question: str, chunk: Chunk) -> float:
    """Cross-score: weighted (prefix-aware) term overlap of question vs chunk."""
    q = _terms(question)
    if not q:
        return 0.0
    d = _terms(chunk.text + " " + chunk.source)
    overlap = sum(min(q[t], _matches(t, d)) for t in q)
    score = overlap / sum(q.values())                  # normalise by question length
    src_terms = _terms(chunk.source)                   # word-level, not substring
    if q.keys() & src_terms:                            # reward exact article/source hit
        score += 0.25
    return round(score, 4)


def rerank(question: str, hits: list[tuple[Chunk, float]], top: int = 4
           ) -> list[tuple[Chunk, float]]:
    """Re-score fused candidates by (prefix-aware) relevance; return best `top`.

    The incoming fusion score breaks ties so a strong dense+BM25 signal isn't lost
    when two candidates score equally on lexical overlap. A real cross-encoder
    (Cohere Rerank / bge-reranker) swaps in here behind the same interface.
    """
    fused = [s for _, s in hits]
    lo, hi = (min(fused), max(fused)) if fused else (0.0, 1.0)
    span = (hi - lo) or 1.0
    rescored = [(c, round(_relevance(question, c) + 0.01 * (s - lo) / span, 4))
                for c, s in hits]
    rescored.sort(key=lambda x: -x[1])
    return rescored[:top]
