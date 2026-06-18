"""RegAgent demo: build corpus + graph, ask compliance questions, see GraphRAG
provenance (dense + graph-expanded sources) and per-run cost.

Run:  python demo.py            (mock mode, no API key needed)
      OPENAI_API_KEY=sk-... python demo.py   (real answers)
"""
from regagent.store import DocStore
from regagent.ingest import load_sample
from regagent.graph import KnowledgeGraph
from regagent.agent import answer_question

chunks = load_sample()
store = DocStore()
store.add(chunks)
graph = KnowledgeGraph().build(chunks)
print(f"Corpus: {len(store.chunks)} chunks · graph: {len(graph.node_concepts)} nodes\n")

questions = [
    "Is an AI system that scores citizens by social behaviour allowed?",
    "What must providers of high-risk AI do about risk management?",
    "Do I need to tell users they're talking to an AI chatbot?",
]

for q in questions:
    a = answer_question(store, q, customer="acme-bank", graph=graph)
    print("Q:", q)
    print("A:", a.answer)
    print("Dense sources:", ", ".join(a.sources))
    print("Graph-expanded:", ", ".join(a.graph_sources) or "—")
    print(f"Cost: ${a.cost_usd:.5f}\n" + "-" * 72)
