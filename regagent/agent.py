"""RegAgent — a regulatory-compliance GraphRAG agent with provenance.

Answers "does X comply with the AI Act?" style questions over a regulation
corpus, citing the exact articles it used. Every step is wrapped in
agentcost.track so the run's cost — per step, per customer — lands on the
agentcost dashboard. This is the dogfooding loop: a real agent measured by
the tooling we're selling.

Set OPENAI_API_KEY for real answers; without it, the agent runs in mock mode
(deterministic stub LLM) so the whole flow + cost tracking works offline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from agentcost import track, SQLiteSink

from .store import DocStore, Chunk
from .sparse import BM25Index
from .fusion import rrf
from .graph import KnowledgeGraph
from .provenance import score as score_provenance, ProvenanceReport

SINK = SQLiteSink(os.environ.get("AGENTCOST_DB", "regagent.db"))


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[str] = field(default_factory=list)        # dense provenance
    graph_sources: list[str] = field(default_factory=list)  # graph-expanded provenance
    grounding: float = 0.0                                   # 0-1 how supported by sources
    provenance: ProvenanceReport | None = None
    cost_usd: float = 0.0


def _llm(messages: list[dict], model: str = "gpt-4o-mini") -> tuple[str, dict]:
    """Call the LLM, return (text, raw_response). Mock when no API key."""
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
        return resp.choices[0].message.content, resp.model_dump()
    # mock: build a grounded answer by quoting the provided excerpts, so the
    # provenance scorer has something real to check. Fake usage feeds agentcost.
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    if "Excerpts:" in user:
        import re
        body = user.split("Excerpts:", 1)[1]
        sents = re.findall(r"[A-Z][^.]{30,}\.", body)
        text = "[mock] " + " ".join(sents[:2]) if sents else "[mock] No relevant provision found."
    else:
        text = "retrieve"
    fake = {"model": model, "usage": {"prompt_tokens": 1200, "completion_tokens": 250}}
    return text, fake


def _route(question: str) -> tuple[str, dict]:
    msg = [{"role": "system", "content": "Classify if this needs the regulation corpus. Answer 'retrieve'."},
           {"role": "user", "content": question}]
    return _llm(msg, model="gpt-4o-mini")


def answer_question(store: DocStore, question: str, customer: str = "demo",
                    graph: KnowledgeGraph | None = None,
                    bm25: BM25Index | None = None) -> Answer:
    """Run the multi-step agent on one question, tracked end-to-end by agentcost."""
    with track("reg-answer", customer=customer, feature="compliance-qa",
               budget_usd=0.50, sink=SINK) as run:
        # 1) route / classify
        _, r = _route(question)
        run.record_response(r, step="classify")

        # 2) HYBRID retrieval: dense (meaning) + sparse (exact terms), fused by RRF
        dense = store.search(question, k=5)
        if bm25 is not None:
            sparse = bm25.search(question, k=5)
            hits = rrf(dense, sparse, top=4)
        else:
            hits = dense
        sources = [c.source for c, _ in hits]
        by_source = {c.source: c.text for c, _ in hits}

        # 2b) GRAPH retrieval — expand to related articles via shared concepts
        #     (a tool call, no LLM cost — graph traversal is cheap vs the model)
        graph_sources: list[str] = []
        if graph is not None:
            for node, why in graph.expand(sources):
                graph_sources.append(f"{node} ({why})")
                # pull the related article's text into context if we have it
                rel = next((c for c in store.chunks if c.source == node), None)
                if rel:
                    by_source.setdefault(node, rel.text)

        context = "\n\n".join(f"[{s}]\n{t}" for s, t in by_source.items())

        # 3) answer grounded in retrieved text, with citations
        msg = [
            {"role": "system", "content":
             "You are a compliance assistant. Answer ONLY from the provided "
             "regulation excerpts and cite the article(s) you used. If the "
             "excerpts don't cover it, say so."},
            {"role": "user", "content": f"Question: {question}\n\nExcerpts:\n{context}"},
        ]
        text, a = _llm(msg, model="gpt-4o")
        run.record_response(a, step="answer")

        # 4) verify the answer is grounded (self-check)
        vmsg = [{"role": "system", "content": "Reply 'ok' if the answer cites the excerpts."},
                {"role": "user", "content": text}]
        _, v = _llm(vmsg, model="gpt-4o-mini")
        run.record_response(v, step="verify")

        # 5) provenance scoring — how well is the answer grounded in the sources?
        #    A failed-grounding run is marked as a FAILURE so agentcost counts it
        #    as wasted spend (cost-per-trusted-answer, not cost-per-answer).
        report = score_provenance(text, by_source)
        if report.grounding_score >= 0.35:
            run.mark_success()
        else:
            run.mark_failure()
        cost = run.total_cost

    return Answer(question=question, answer=text, sources=sources,
                  graph_sources=graph_sources, grounding=report.grounding_score,
                  provenance=report, cost_usd=cost)
