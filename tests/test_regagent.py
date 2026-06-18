"""Test suite for RegAgent. Run: pytest -q"""
import os
import sqlite3

from regagent.ingest import load_sample
from regagent.store import DocStore
from regagent.sparse import BM25Index
from regagent.fusion import rrf
from regagent.rerank import rerank
from regagent.graph import KnowledgeGraph
from regagent.provenance import score
from regagent.agent import answer_question


def _corpus():
    chunks = load_sample()
    store = DocStore(); store.add(chunks)
    bm25 = BM25Index().build(chunks)
    graph = KnowledgeGraph().build(chunks)
    return chunks, store, bm25, graph


def test_dense_retrieval_finds_relevant():
    _, store, _, _ = _corpus()
    hits = store.search("social scoring of citizens", k=3)
    assert any("5(1)(c)" in c.source for c, _ in hits)


def test_sparse_matches_exact_terms():
    chunks, _, bm25, _ = _corpus()
    hits = bm25.search("transparency instructions for use", k=3)
    assert any("Article 13" in c.source for c, _ in hits)


def test_rrf_fuses_and_ranks():
    chunks, store, bm25, _ = _corpus()
    q = "high-risk risk management"
    fused = rrf(store.search(q, 5), bm25.search(q, 5), top=4)
    assert len(fused) <= 4
    # items appearing in both lists should rank highly
    assert any("Article 9" in c.source for c, _ in fused)


def test_rerank_orders_by_relevance():
    chunks, store, bm25, _ = _corpus()
    q = "inform users they interact with an AI system"
    fused = rrf(store.search(q, 6), bm25.search(q, 6), top=6)
    ranked = rerank(q, fused, top=3)
    assert "Article 50" in ranked[0][0].source   # transparency-to-users article on top


def test_graph_expands_related_articles():
    chunks, _, _, graph = _corpus()
    # use the actual node label (now includes a title)
    node = next(n for n in graph.node_concepts if "Article 9" in n)
    rel = graph.expand([node])
    # Article 9 (risk mgmt, high-risk) should connect to other high-risk articles
    assert any("Article" in n for n, _ in rel)


def test_provenance_scoring():
    sources = {"Art 9": "Providers of high-risk AI systems shall maintain a risk "
                         "management system throughout the lifecycle."}
    grounded = score("Providers must maintain a risk management system.", sources)
    assert grounded.grounding_score > 0.4 and grounded.grounded_claims >= 1
    hallucination = score("The moon is made of cheese and unicorns exist.", sources)
    assert hallucination.grounding_score < 0.2


def test_agent_end_to_end_tracks_cost(tmp_path):
    os.environ["AGENTCOST_DB"] = str(tmp_path / "t.db")
    import importlib, regagent.agent as ag
    importlib.reload(ag)
    chunks, store, bm25, graph = _corpus()
    a = ag.answer_question(store, "Is social scoring allowed?", customer="acme",
                           graph=graph, bm25=bm25)
    assert a.sources and a.cost_usd > 0 and a.provenance is not None
    n = sqlite3.connect(os.environ["AGENTCOST_DB"]).execute(
        "SELECT COUNT(*) FROM runs").fetchone()[0]
    assert n == 1
