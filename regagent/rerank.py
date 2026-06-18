"""Reranking — the precision pass after fusion.

Hybrid fusion gives good recall (the right article is *somewhere* in the top-k).
A reranker then re-scores each candidate against the question for precision, so
the article the answer is actually built on sits at the top.

Two backends behind one interface:
  - default: a lexical cross-scorer (no model, offline, deterministic) — keeps the
    Docker image slim and the test suite hermetic.
  - opt-in: a real cross-encoder (set REGAGENT_RERANK=cross-encoder and install
    `sentence-transformers`). Falls back to lexical if the dependency is absent,
    so enabling it never breaks a deployment that lacks the model.
"""
from __future__ import annotations

import os
import re
from collections import Counter

from .store import Chunk

_CE = None            # lazily-loaded CrossEncoder model
_CE_UNAVAILABLE = False

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


def _load_cross_encoder():
    """Lazily load the cross-encoder; cache it; degrade to None if unavailable."""
    global _CE, _CE_UNAVAILABLE
    if _CE is not None or _CE_UNAVAILABLE:
        return _CE
    try:
        from sentence_transformers import CrossEncoder
        model = os.environ.get("REGAGENT_RERANK_MODEL",
                               "cross-encoder/ms-marco-MiniLM-L-6-v2")
        _CE = CrossEncoder(model)
    except Exception:
        _CE_UNAVAILABLE = True   # missing dep / model — fall back to lexical
    return _CE


def _rerank_cross_encoder(question, hits, top):
    ce = _load_cross_encoder()
    if ce is None:
        return None
    scores = ce.predict([[question, c.text] for c, _ in hits])
    rescored = [(c, round(float(s), 4)) for (c, _), s in zip(hits, scores)]
    rescored.sort(key=lambda x: -x[1])
    return rescored[:top]


def rerank(question: str, hits: list[tuple[Chunk, float]], top: int = 4
           ) -> list[tuple[Chunk, float]]:
    """Re-score fused candidates and return the best `top`.

    With REGAGENT_RERANK=cross-encoder (and sentence-transformers installed) a
    real cross-encoder scores each (question, passage) pair. Otherwise a lexical
    cross-scorer is used; the incoming fusion score breaks ties so a strong
    dense+BM25 signal isn't lost when two candidates tie on lexical overlap.
    """
    if not hits:
        return []
    if os.environ.get("REGAGENT_RERANK", "lexical") == "cross-encoder":
        ce_ranked = _rerank_cross_encoder(question, hits, top)
        if ce_ranked is not None:
            return ce_ranked
    fused = [s for _, s in hits]
    lo, hi = min(fused), max(fused)
    span = (hi - lo) or 1.0
    rescored = [(c, round(_relevance(question, c) + 0.01 * (s - lo) / span, 4))
                for c, s in hits]
    rescored.sort(key=lambda x: -x[1])
    return rescored[:top]
