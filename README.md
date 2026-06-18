# RegAgent

A regulatory-compliance **GraphRAG agent with provenance** — answers
"does X comply with the EU AI Act?" and **cites the exact articles** it used.

Built as a real, sellable agent *and* as the first live case study for
[agentcost](https://github.com/Lagra2105/agentcost): every step of every run
is tracked, so the agent's real economics (cost per step, per customer) land
on the agentcost dashboard. Dogfooding the tooling on a real product.

## Why provenance
In compliance you can't hallucinate. RegAgent answers **only** from retrieved
regulation text and returns the source articles — so a human can verify. That
trust requirement is exactly where generic chatbots can't compete.

## Architecture (Phase 1)
1. **Ingest** — regulation text → citeable chunks (`ingest.py`)
2. **Agent** (multi-step, wrapped in `agentcost.track`):
   - `classify` → route the question
   - `retrieve` → dense vector search (provenance captured)
   - `answer` → grounded answer with article citations
   - `verify` → self-check against sources

Runs offline in **mock mode** (no key); set `OPENAI_API_KEY` for real answers
and embeddings.

## Run
```bash
python demo.py
```

## Roadmap
- [ ] Full EU AI Act corpus (drop into data/ai_act.txt)
- [ ] Knowledge-graph layer (Neo4j): articles, cross-references, definitions → graph retrieval
- [ ] Hybrid fusion (dense + sparse + graph) with reranking
- [ ] Provenance scoring (answer-to-source grounding)
- [ ] Deploy as an API + UI
