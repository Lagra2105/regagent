# RegAgent 🛡️

**A multi-regulation compliance GraphRAG agent for EU law (EU AI Act + DORA +
GDPR) — with provenance, abstention, and measurable accuracy.**

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

A golden set of 38 compliance questions (EU AI Act + DORA + GDPR), each tied to
the article(s) that should ground a correct answer. Run `python -m eval.evaluate`
to reproduce.

```
# offline / deterministic (hash embeddings, no API key)
Retrieval recall@4: 97%   MRR: 0.84   Citation recall: 97%   Grounding: 0.89
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
- **Corpus**: 36 EU AI Act + 15 DORA + 15 GDPR articles built in; drop the full
  text into `data/ai_act.txt` (all 113) or add another regulation via
  `load_corpus(path, regulation=...)` — no code change.
- **Deploy**: Dockerfile included; runs on Railway / Render with a Postgres add-on.

## Roadmap

- [x] Hybrid retrieval (dense + sparse + RRF + rerank)
- [x] Knowledge graph layer + graph-expanded provenance
- [x] Provenance scoring (lexical + semantic) + abstention
- [x] Evaluation harness + cost/quality experiment
- [x] FastAPI service, pgvector, Docker
- [x] Multi-regulation corpus (EU AI Act + DORA + GDPR) with cross-regulation graph links
- [ ] Full AI Act (113 articles); Neo4j-backed graph; cross-encoder reranker
- [ ] Hosted demo + multi-tenant access

---

Built by a Senior Data Scientist / MLOps engineer. The agent is instrumented by
**agentcost** — running it on my own tooling immediately surfaced (and fixed) a
real bug in the cost tracker. That's the point: real agents, measured honestly.
