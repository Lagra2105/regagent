# RegAgent 🛡️

**A multi-regulation compliance GraphRAG agent for EU law (EU AI Act + DORA +
GDPR + NIS2) — with provenance, abstention, and measurable accuracy.**

Ask *"does X comply?"* and get an answer grounded in the exact articles, a
confidence score, and — when the regulation doesn't cover it — an honest refusal
instead of a hallucination. In compliance, a wrong answer is worse than none.

It isn't tied to one law: the corpus is multi-regulation, and the knowledge graph
links provisions **across** regulations (e.g. an EU AI Act risk-management duty
connected to a DORA ICT-risk provision) — adding a regulation is dropping in text,
not changing code.

## Why it's different

Generic chatbots can't be trusted for compliance: they paraphrase, they invent,
and you can't audit where an answer came from. RegAgent is built around **trust**:

| Capability | What it does |
|---|---|
| **Hybrid retrieval** | dense (meaning) + BM25 (exact terms) fused with RRF, then reranked |
| **Knowledge graph** | links articles by shared concepts — within *and across* regulations (AI Act ↔ DORA); pulls in related provisions the vector search misses, and explains the link |
| **Provenance** | every answer cites the articles it used; a grounding score (lexical *or* semantic) measures how supported it is |
| **Abstention** | refuses out-of-scope / ungrounded questions — and skips the expensive answer step (≈18× cheaper) |
| **Multi-regulation analysis** | decomposes a real-world question ("does our system comply?") into focused sub-questions, answers each across regulations, then synthesises — the plan→execute→synthesise agent loop |
| **Multilingual** | ask in English or French and get the answer in that language; multilingual embeddings retrieve over the English corpus cross-lingually, citations stay language-agnostic |
| **Measured** | a golden benchmark reports retrieval recall@k, MRR, citation accuracy, grounding |
| **Costed** | instrumented with [agentcost](https://github.com/Lagra2105/agentcost): real per-step economics, cost-per-*trusted*-answer |

## Pipeline

```
question
 → classify
 → dense + BM25  →  RRF fusion  →  rerank        (hybrid retrieval)
 → abstain-gate  (refuse early if nothing relevant — trust + cost win)
 → graph expand  (related articles via shared concepts)
 → answer        (grounded, with article citations)
 → verify
 → provenance score  (lexical / semantic grounding)
 → agentcost     (cost per step, success = grounded)
```

## Benchmark

A golden set of 42 compliance questions (EU AI Act + DORA + GDPR + NIS2), each
tied to the article(s) that should ground a correct answer. Run
`python -m eval.evaluate` to reproduce.

```
# offline / deterministic (hash embeddings, no API key)
Retrieval recall@4: 98%   MRR: 0.83   Citation recall: 98%   Grounding: 0.89
by regulation:  EU AI Act 96%   DORA 100%   GDPR 100%   NIS2 100%
```

The offline numbers use a toy hash-embedding so the suite runs without a key and
is fully reproducible. With production embeddings (`OPENAI_API_KEY` →
`text-embedding-3-small`) retrieval is materially higher — the remaining offline
misses are semantic-disambiguation cases (e.g. *provider obligations* Art 16 vs
the specific obligation articles) that real embeddings resolve.

## Quickstart

```bash
pip install -r requirements.txt
python demo.py                            # mock mode, no key needed
OPENAI_API_KEY=sk-... python demo.py      # real embeddings + answers

python -m eval.evaluate        # accuracy metrics
python -m eval.cost_quality    # cost-vs-quality model selection
uvicorn service.api:app        # web UI + REST API at http://localhost:8000
```

Config via env: `OPENAI_API_KEY`, `DATABASE_URL` (→ pgvector),
`REGAGENT_ANSWER_MODEL`, `REGAGENT_GROUNDING` (lexical|semantic),
`REGAGENT_ABSTAIN_THRESHOLD`.

## Architecture notes

- **Storage**: in-memory for dev; **pgvector** (Postgres) via `DATABASE_URL` for
  scale and persistence — one component is both database and vector store.
- **Corpus**: 36 EU AI Act + 15 DORA + 15 GDPR + 9 NIS2 articles built in; drop
  the full text into `data/ai_act.txt` (all 113) or add another regulation via
  `load_corpus(path, regulation=...)` — no code change.
- **Deploy**: Dockerfile included; runs on Railway / Render with a Postgres add-on.

## Roadmap

- [x] Hybrid retrieval (dense + sparse + RRF + rerank)
- [x] Knowledge graph layer + graph-expanded provenance
- [x] Provenance scoring (lexical + semantic) + abstention
- [x] Evaluation harness + cost/quality experiment
- [x] FastAPI service, pgvector, Docker
- [x] Multi-regulation corpus (EU AI Act + DORA + GDPR + NIS2) with cross-regulation graph links
- [x] Pluggable reranker — lexical default, opt-in cross-encoder (`REGAGENT_RERANK=cross-encoder`)
- [x] Multi-regulation analysis (`/analyze`) — query decomposition → per-sub answers → synthesis
- [ ] Full AI Act (113 articles) via `data/ai_act.txt`; Neo4j-backed graph
- [ ] Hosted demo + multi-tenant access

---

Built by a Senior Data Scientist / MLOps engineer. The agent is instrumented by
**agentcost** — running it on my own tooling immediately surfaced (and fixed) a
real bug in the cost tracker. That's the point: real agents, measured honestly.
