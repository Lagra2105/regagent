"""Hybrid fusion — combine dense + sparse rankings with Reciprocal Rank Fusion.

RRF is the standard, robust way to merge retrievers without tuning score scales:
each result contributes 1/(k + rank) from each list it appears in. Items that
rank well in *both* dense and sparse float to the top — exactly what you want
for regulation (meaning + exact terms agree).
"""
from __future__ import annotations

from .store import Chunk


def rrf(*ranked_lists: list[tuple[Chunk, float]], k: int = 60, top: int = 5
        ) -> list[tuple[Chunk, float]]:
    """Fuse several (chunk, score) lists. Returns fused (chunk, rrf_score), best first."""
    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for lst in ranked_lists:
        for rank, (chunk, _score) in enumerate(lst):
            fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
            by_id[chunk.id] = chunk
    order = sorted(fused.items(), key=lambda kv: -kv[1])
    return [(by_id[cid], round(s, 5)) for cid, s in order[:top]]
