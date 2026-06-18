"""Knowledge-graph layer — what makes this GraphRAG, not plain RAG.

Articles become nodes; an edge links two articles that share a regulatory
concept (e.g. both about 'high-risk' or 'transparency'). Dense retrieval finds
seed articles; graph traversal then pulls in *related* articles the vector
search might miss — with the connection made explicit (provenance you can show).

In-memory for Phase 2; the same interface (build / neighbors) maps cleanly onto
Neo4j Cypher in Phase 3.
"""
from __future__ import annotations

import re
from collections import defaultdict

from .store import Chunk

# Regulatory concepts that connect provisions. Extend as the corpus grows.
CONCEPTS = [
    "prohibited", "high-risk", "risk management", "transparency", "training data",
    "social scoring", "subliminal", "conformity", "provider", "deployer",
    "biometric", "interact", "harm", "lifecycle", "bias", "importer",
    "distributor", "documentation", "logs", "ce marking", "declaration of conformity",
    "registration", "serious incident", "fundamental rights", "quality management",
    "value chain", "corrective", "literacy",
]


def _concepts_in(text: str) -> set[str]:
    t = text.lower()
    found = {c for c in CONCEPTS if c in t}
    # a couple of stem variants
    if re.search(r"\btrain", t):
        found.add("training data")
    if re.search(r"\brisk", t):
        found.add("risk management")
    return found


class KnowledgeGraph:
    def __init__(self) -> None:
        self.node_concepts: dict[str, set[str]] = defaultdict(set)
        self.edges: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    def build(self, chunks: list[Chunk]) -> "KnowledgeGraph":
        for c in chunks:
            self.node_concepts[c.source] |= _concepts_in(c.text)
        nodes = list(self.node_concepts)
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                shared = self.node_concepts[a] & self.node_concepts[b]
                if shared:
                    self.edges[a][b] |= shared
                    self.edges[b][a] |= shared
        return self

    def neighbors(self, source: str, limit: int = 3) -> list[tuple[str, list[str]]]:
        """Related articles, ranked by number of shared concepts."""
        rel = self.edges.get(source, {})
        ranked = sorted(rel.items(), key=lambda kv: -len(kv[1]))
        return [(node, sorted(shared)) for node, shared in ranked[:limit]]

    def expand(self, sources: list[str], limit_each: int = 2) -> list[tuple[str, str]]:
        """For seed sources, return (related_source, 'via: concept,...') pairs."""
        out: list[tuple[str, str]] = []
        seen = set(sources)
        for s in sources:
            for node, shared in self.neighbors(s, limit_each):
                if node not in seen:
                    seen.add(node)
                    out.append((node, "via " + ", ".join(shared)))
        return out
