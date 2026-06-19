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
import re
from dataclasses import dataclass, field

from agentcost import track, SQLiteSink

from .store import DocStore, Chunk
from .sparse import BM25Index
from .fusion import rrf
from .rerank import rerank
from .graph import KnowledgeGraph
from .provenance import score as score_provenance, ProvenanceReport

SINK = SQLiteSink(os.environ.get("AGENTCOST_DB", "regagent.db"))
ANSWER_MODEL = os.environ.get("REGAGENT_ANSWER_MODEL", "gpt-4o")  # tunable for experiments
# Below this grounding score the agent abstains instead of answering — better a
# safe "I don't know" than a confident hallucination in a compliance context.
ABSTAIN_THRESHOLD = float(os.environ.get("REGAGENT_ABSTAIN_THRESHOLD", "0.30"))
# Minimum top retrieval relevance to even attempt an answer (else abstain early).
RETRIEVAL_MIN = float(os.environ.get("REGAGENT_RETRIEVAL_MIN", "0.4"))
# 'semantic' (embedding-based, paraphrase-robust) or 'lexical' (term overlap).
GROUNDING_METHOD = os.environ.get("REGAGENT_GROUNDING", "lexical")

# Multilingual answers — the corpus stays English (multilingual embeddings handle
# cross-lingual retrieval), but the LLM replies in the user's language. Citations
# (article labels) are language-agnostic, so they stay correct.
_LANG_NAME = {"en": "English", "fr": "French"}


def _lang_instruction(lang: str) -> str:
    if not lang or lang == "auto":
        return " Respond in the same language as the question."
    return f" Respond in {_LANG_NAME.get(lang, 'English')}."


# Localised abstention messages (auto/en default to English; fr is translated).
_ABSTAIN_EARLY = {
    "en": "Out of scope — no relevant provision was found in the available "
          "regulation. I won't answer without grounding; please rephrase or "
          "consult qualified legal counsel.",
    "fr": "Hors champ — aucune disposition pertinente n'a été trouvée dans la "
          "réglementation disponible. Je ne répondrai pas sans fondement ; "
          "veuillez reformuler ou consulter un conseil juridique qualifié.",
}
_ABSTAIN_WEAK = {
    "en": "I can't answer this with confidence from the available provisions. "
          "The retrieved articles don't sufficiently cover this question — "
          "please consult qualified legal counsel.",
    "fr": "Je ne peux pas répondre à cette question avec certitude à partir des "
          "dispositions disponibles. Les articles récupérés ne couvrent pas "
          "suffisamment cette question — veuillez consulter un conseil juridique "
          "qualifié.",
}


def _msg(table: dict, lang: str) -> str:
    return table.get(lang, table["en"])


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[str] = field(default_factory=list)        # dense provenance
    graph_sources: list[str] = field(default_factory=list)  # graph-expanded provenance
    grounding: float = 0.0                                   # 0-1 how supported by sources
    provenance: ProvenanceReport | None = None
    abstained: bool = False                                  # refused (low grounding)
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
                    bm25: BM25Index | None = None, lang: str = "auto") -> Answer:
    """Run the multi-step agent on one question, tracked end-to-end by agentcost."""
    with track("reg-answer", customer=customer, feature="compliance-qa",
               budget_usd=0.50, sink=SINK) as run:
        # 1) route / classify
        _, r = _route(question)
        run.record_response(r, step="classify")

        # 2) HYBRID retrieval: dense (meaning) + sparse (exact terms), fused by RRF
        dense = store.search(question, k=6)
        if bm25 is not None:
            sparse = bm25.search(question, k=6)
            fused = rrf(dense, sparse, top=6)
        else:
            fused = dense
        # 2a) RERANK — precision pass: put the most relevant article on top
        hits = rerank(question, fused, top=4)
        sources = [c.source for c, _ in hits]
        by_source = {c.source: c.text for c, _ in hits}

        # 2c) EARLY ABSTENTION — if no article is relevant enough, refuse BEFORE
        #     paying for generation. Trust win (no hallucination) + cost win
        #     (abstained runs skip the expensive answer step).
        if not hits or hits[0][1] < RETRIEVAL_MIN:
            run.mark_failure()
            from .provenance import ProvenanceReport
            empty = ProvenanceReport(0.0, 0, 0, [], [])
            return Answer(
                question=question, answer=_msg(_ABSTAIN_EARLY, lang),
                sources=[], graph_sources=[], grounding=0.0,
                provenance=empty, abstained=True, cost_usd=run.total_cost)

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
             "excerpts don't cover it, say so." + _lang_instruction(lang)},
            {"role": "user", "content": f"Question: {question}\n\nExcerpts:\n{context}"},
        ]
        text, a = _llm(msg, model=ANSWER_MODEL)
        run.record_response(a, step="answer")

        # 4) verify the answer is grounded (self-check)
        vmsg = [{"role": "system", "content": "Reply 'ok' if the answer cites the excerpts."},
                {"role": "user", "content": text}]
        _, v = _llm(vmsg, model="gpt-4o-mini")
        run.record_response(v, step="verify")

        # 5) provenance scoring — how well is the answer grounded in the sources?
        #    A failed-grounding run is marked as a FAILURE so agentcost counts it
        #    as wasted spend (cost-per-trusted-answer, not cost-per-answer).
        report = score_provenance(text, by_source, method=GROUNDING_METHOD)
        grounded = report.grounding_score >= ABSTAIN_THRESHOLD

        # 6) ABSTENTION — in compliance, an unsupported answer is worse than none.
        #    If grounding is too weak, refuse rather than hallucinate.
        abstained = False
        if not grounded:
            text = _msg(_ABSTAIN_WEAK, lang)
            abstained = True
            run.mark_failure()
        else:
            run.mark_success()
        cost = run.total_cost

    return Answer(question=question, answer=text, sources=sources,
                  graph_sources=graph_sources, grounding=report.grounding_score,
                  provenance=report, abstained=abstained, cost_usd=cost)


# --------------------------------------------------------------------------- #
# Multi-regulation analysis: the agentic layer.
#
# A real compliance question is rarely about one article — it's "does our system
# comply?", which spans several regulations. answer_complex PLANS (decomposes the
# question into focused sub-questions), runs each through the full pipeline above,
# then SYNTHESISES a single grounded answer. The plan/execute/synthesise loop —
# with the per-sub abstention decisions — is what makes this an agent, not a bot.
# --------------------------------------------------------------------------- #

@dataclass
class ComplexAnswer:
    question: str
    answer: str                              # synthesised final answer
    sub_questions: list[str] = field(default_factory=list)
    sub_answers: list[Answer] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)   # union of cited articles
    abstained: bool = False                  # True only if every sub abstained
    cost_usd: float = 0.0


def plan_subquestions(question: str, lang: str = "auto") -> tuple[list[str], dict]:
    """Decompose a compliance question into focused sub-questions (plan step)."""
    msg = [
        {"role": "system", "content":
         "You are a compliance analyst. Break the user's question into 2 to 4 "
         "focused, self-contained sub-questions, each targeting a specific "
         "obligation under the EU AI Act, DORA, GDPR or NIS2. Return ONLY the "
         "sub-questions, one per line, no numbering or commentary." + _lang_instruction(lang)},
        {"role": "user", "content": question},
    ]
    text, raw = _llm(msg, model="gpt-4o-mini")
    subs = [ln.strip(" -•\t0123456789.") for ln in text.splitlines()
            if len(ln.strip()) > 12 and "?" in ln]
    if not subs:   # mock / empty model output → split on conjunctions
        parts = re.split(r"\band\b|;|,", question)
        subs = [p.strip().rstrip("?") + "?" for p in parts if len(p.strip()) > 15] \
            or [question]
    return subs[:4], raw


def _synthesise(question: str, sub_answers: list[Answer],
                lang: str = "auto") -> tuple[str, dict]:
    """Combine grounded sub-answers into one coherent, cited answer."""
    blocks = "\n\n".join(
        f"Sub-question: {sa.question}\nFinding: {sa.answer}\n"
        f"Articles: {', '.join(sa.sources) or 'none'}" for sa in sub_answers)
    if not os.environ.get("OPENAI_API_KEY"):   # deterministic mock synthesis
        body = "\n".join(f"- {sa.question} {' '.join(sa.sources)}" for sa in sub_answers)
        text = "[mock] Combined analysis across regulations:\n" + body
        return text, {"model": ANSWER_MODEL,
                      "usage": {"prompt_tokens": 900, "completion_tokens": 200}}
    msg = [
        {"role": "system", "content":
         "You are a compliance analyst. Using ONLY the findings below (each "
         "grounded in cited articles), write one coherent answer to the overall "
         "question. Cite the article labels you rely on. If a sub-question found "
         "no provision, state that the regulation does not cover it. Be concise."
         + _lang_instruction(lang)},
        {"role": "user", "content": f"Overall question: {question}\n\nFindings:\n{blocks}"},
    ]
    return _llm(msg, model=ANSWER_MODEL)


def answer_complex(store: DocStore, question: str, customer: str = "demo",
                   graph: KnowledgeGraph | None = None,
                   bm25: BM25Index | None = None, lang: str = "auto") -> ComplexAnswer:
    """Plan → run each sub-question through the pipeline → synthesise. Tracked."""
    with track("reg-analyze", customer=customer, feature="multi-reg-analysis",
               budget_usd=1.0, sink=SINK) as run:
        subs, plan_raw = plan_subquestions(question, lang=lang)
        run.record_response(plan_raw, step="plan")

        sub_answers = [answer_question(store, s, customer=customer,
                                       graph=graph, bm25=bm25, lang=lang) for s in subs]

        text, syn_raw = _synthesise(question, sub_answers, lang=lang)
        run.record_response(syn_raw, step="synthesize")

        all_abstained = all(sa.abstained for sa in sub_answers) if sub_answers else True
        run.mark_failure() if all_abstained else run.mark_success()
        parent_cost = run.total_cost

    sources: list[str] = []
    for sa in sub_answers:
        for s in sa.sources:
            if s not in sources:
                sources.append(s)
    cost = parent_cost + sum(sa.cost_usd for sa in sub_answers)
    return ComplexAnswer(question=question, answer=text, sub_questions=subs,
                         sub_answers=sub_answers, sources=sources,
                         abstained=all_abstained, cost_usd=cost)
