"""Provenance scoring — the trust layer.

In compliance, an answer is only as good as its grounding. This scores how well
the generated answer is supported by the retrieved sources:

  - splits the answer into claims (sentences)
  - for each claim, finds its best-matching source passage
  - flags claims with weak support (possible hallucination)
  - returns a 0-1 grounding score + per-claim evidence

Phase 3 uses a lexical-overlap baseline (no extra model call, runs offline);
it swaps cleanly for an embedding/NLI check when a key is present.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ClaimCheck:
    claim: str
    best_source: str | None
    support: float          # 0-1 overlap with the best source
    grounded: bool


@dataclass
class ProvenanceReport:
    grounding_score: float          # mean support across claims
    grounded_claims: int
    total_claims: int
    weak: list[ClaimCheck]          # claims below threshold (review these)
    checks: list[ClaimCheck]


_STOP = set("the a an of to in for and or is are be on that this with as by it "
            "from at which an their such not no any may shall will".split())


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Z]{3,}", text.lower()) if w not in _STOP}


def _support(claim: str, source_text: str) -> float:
    c, s = _terms(claim), _terms(source_text)
    if not c:
        return 0.0
    return len(c & s) / len(c)


def _claims(answer: str) -> list[str]:
    return [c.strip() for c in re.split(r"(?<=[.])\s+", answer) if len(c.strip()) > 12]


def _checks_lexical(claims, sources, threshold) -> list[ClaimCheck]:
    out = []
    for claim in claims:
        best_src, best = None, 0.0
        for label, text in sources.items():
            sup = _support(claim, text)
            if sup > best:
                best, best_src = sup, label
        out.append(ClaimCheck(claim, best_src, round(best, 3), best >= threshold))
    return out


def _checks_semantic(claims, sources, threshold) -> list[ClaimCheck]:
    """Embedding-based: a claim is grounded if it's semantically close to a
    source — robust to paraphrase, where lexical overlap fails."""
    from .store import embed, _cosine
    labels = list(sources)
    texts = [sources[l] for l in labels]
    embs = embed(claims + texts)
    claim_e, src_e = embs[:len(claims)], embs[len(claims):]
    out = []
    for claim, ce in zip(claims, claim_e):
        best_src, best = None, -1.0
        for label, se in zip(labels, src_e):
            sim = _cosine(ce, se)
            if sim > best:
                best, best_src = sim, label
        out.append(ClaimCheck(claim, best_src, round(best, 3), best >= threshold))
    return out


def score(answer: str, sources: dict[str, str], threshold: float = 0.35,
          method: str = "lexical") -> ProvenanceReport:
    """Grounding report. method='semantic' uses embeddings (paraphrase-robust);
    'lexical' uses term overlap (fast, no embeddings)."""
    claims = _claims(answer)
    if method == "semantic" and claims and sources:
        checks = _checks_semantic(claims, sources, threshold)
    else:
        checks = _checks_lexical(claims, sources, threshold)

    total = len(checks) or 1
    grounded = sum(1 for c in checks if c.grounded)
    mean = round(sum(c.support for c in checks) / total, 3)
    weak = [c for c in checks if not c.grounded]
    return ProvenanceReport(grounding_score=mean, grounded_claims=grounded,
                            total_claims=len(checks), weak=weak, checks=checks)
