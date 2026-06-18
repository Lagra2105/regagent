"""RegAgent demo: build the corpus, ask compliance questions, see provenance + cost.

Run:  python demo.py            (mock mode, no API key needed)
      OPENAI_API_KEY=sk-... python demo.py   (real answers)
"""
from regagent.store import DocStore
from regagent.ingest import load_sample
from regagent.agent import answer_question

store = DocStore()
store.add(load_sample())
print(f"Corpus loaded: {len(store.chunks)} chunks\n")

questions = [
    "Is an AI system that scores citizens by social behaviour allowed?",
    "What must providers of high-risk AI do about risk management?",
    "Do I need to tell users they're talking to an AI chatbot?",
]

for q in questions:
    a = answer_question(store, q, customer="acme-bank")
    print("Q:", q)
    print("A:", a.answer)
    print("Sources:", ", ".join(a.sources))
    print(f"Cost: ${a.cost_usd:.5f}\n" + "-" * 70)
