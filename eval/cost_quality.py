"""Cost vs quality experiment — the question only RegAgent + agentcost can answer.

agentcost's savings tip says "switch the answer step gpt-4o -> gpt-4o-mini, save
~94%." But does quality hold? This runs the golden set with each answer model,
measuring grounding (quality) AND cost (agentcost) — so the model choice is made
on evidence, not vibes. This is the dogfooding payoff: the two products together
answer a question neither could alone.

Run:  OPENAI_API_KEY=sk-... python -m eval.cost_quality   (real numbers)
      python -m eval.cost_quality                          (mock, mechanics only)
"""
from __future__ import annotations

import sqlite3
import tempfile
import os

from regagent.ingest import load_sample
from regagent.store import DocStore
from regagent.sparse import BM25Index
from regagent.graph import KnowledgeGraph
from regagent import agent as agent_mod
from eval.golden import GOLDEN


def _run_with_answer_model(model: str) -> tuple[float, float, float]:
    """Run the golden set with `model` on the answer step. Returns
    (mean_grounding, total_cost_usd, cost_per_question)."""
    # isolate cost data per variant in its own sink
    db = os.path.join(tempfile.mkdtemp(), "exp.db")
    agent_mod.SINK = agent_mod.SQLiteSink(db)
    agent_mod.ANSWER_MODEL = model      # the agent reads this for the answer step

    chunks = load_sample()
    store = DocStore(); store.add(chunks)
    bm25 = BM25Index().build(chunks)
    graph = KnowledgeGraph().build(chunks)

    grounding = 0.0
    for q, _ in GOLDEN:
        a = agent_mod.answer_question(store, q, customer="exp", graph=graph, bm25=bm25)
        grounding += a.grounding
    n = len(GOLDEN)

    total = sqlite3.connect(db).execute("SELECT COALESCE(SUM(cost_usd),0) FROM runs").fetchone()[0]
    return grounding / n, total, total / n


def main() -> None:
    variants = ["gpt-4o", "gpt-4o-mini"]
    print(f"{'answer model':16}{'grounding':>12}{'total $':>12}{'$/question':>14}")
    print("-" * 54)
    results = {}
    for m in variants:
        g, total, per = _run_with_answer_model(m)
        results[m] = (g, total, per)
        print(f"{m:16}{g:>12.3f}{total:>12.5f}{per:>14.6f}")
    print("-" * 54)

    g0, t0, _ = results["gpt-4o"]
    g1, t1, _ = results["gpt-4o-mini"]
    if t0 > 0:
        savings = (t0 - t1) / t0 * 100
        dq = g0 - g1
        print(f"Switching answer gpt-4o -> gpt-4o-mini: "
              f"save {savings:.0f}% cost, grounding change {dq:+.3f}")
        verdict = "worth it" if dq <= 0.05 else "quality drop too high"
        print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
