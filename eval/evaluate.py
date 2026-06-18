"""Evaluate RegAgent against the golden set — the metrics a buyer asks for.

Reports:
  retrieval recall@k  — did we retrieve at least one correct article?
  MRR                 — how high did the correct article rank?
  citation recall     — did the final answer cite a correct article?
  mean grounding      — how supported are the answers by their sources?

Run:  python -m eval.evaluate
      OPENAI_API_KEY=sk-... python -m eval.evaluate   (real embeddings = real numbers)
"""
from __future__ import annotations

from regagent.ingest import load_all
from regagent.store import DocStore
from regagent.sparse import BM25Index
from regagent.fusion import rrf
from regagent.rerank import rerank
from regagent.graph import KnowledgeGraph
from regagent.agent import answer_question
from eval.golden import GOLDEN


def _hit(expected: set[str], sources: list[str]) -> bool:
    return any(any(e in s for s in sources) for e in expected)


def _rank_of(expected: set[str], ranked_sources: list[str]) -> int | None:
    for i, s in enumerate(ranked_sources):
        if any(e in s for e in expected):
            return i + 1
    return None


def _regulation_of(expected: set[str]) -> str:
    """Which regulation a golden item belongs to, from its expected label."""
    joined = " ".join(expected)
    for reg in ("DORA", "GDPR", "NIS2"):
        if reg in joined:
            return reg
    return "EU AI Act"


def main() -> None:
    chunks = load_all()   # AI Act + DORA — evaluate across the full multi-reg corpus
    store = DocStore(); store.add(chunks)
    bm25 = BM25Index().build(chunks)
    graph = KnowledgeGraph().build(chunks)

    k = 4
    retr_hits = cite_hits = 0
    rr_sum = 0.0
    grounding_sum = 0.0
    rows = []
    per_reg: dict[str, list[int]] = {}   # regulation -> [hits, total]

    for q, expected in GOLDEN:
        # retrieval-only ranking (hybrid + rerank), to measure recall/MRR
        dense = store.search(q, 6)
        sparse = bm25.search(q, 6)
        ranked = rerank(q, rrf(dense, sparse, top=6), top=k)
        ranked_sources = [c.source for c, _ in ranked]

        got = _hit(expected, ranked_sources)
        rank = _rank_of(expected, ranked_sources)
        retr_hits += int(got)
        rr_sum += (1.0 / rank) if rank else 0.0

        reg = _regulation_of(expected)
        agg = per_reg.setdefault(reg, [0, 0])
        agg[0] += int(got); agg[1] += 1

        # full agent — does the cited answer reference the right article?
        a = answer_question(store, q, customer="eval", graph=graph, bm25=bm25)
        cited = _hit(expected, a.sources)
        cite_hits += int(cited)
        grounding_sum += a.grounding

        rows.append((q[:46], "✓" if got else "✗", rank or "-",
                     "✓" if cited else "✗", round(a.grounding, 2)))

    n = len(GOLDEN)
    print(f"{'question':48}{'retr':>6}{'rank':>6}{'cite':>6}{'grnd':>7}")
    print("-" * 73)
    for r in rows:
        print(f"{r[0]:48}{r[1]:>6}{str(r[2]):>6}{r[3]:>6}{r[4]:>7}")
    print("-" * 73)
    print(f"Retrieval recall@{k}: {retr_hits/n:.0%}   "
          f"MRR: {rr_sum/n:.3f}   "
          f"Citation recall: {cite_hits/n:.0%}   "
          f"Mean grounding: {grounding_sum/n:.2f}")
    print("by regulation:  " + "   ".join(
        f"{reg} {h}/{t} ({h/t:.0%})" for reg, (h, t) in sorted(per_reg.items())))


if __name__ == "__main__":
    main()
