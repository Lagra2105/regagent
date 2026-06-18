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
    "registration", "fundamental rights", "quality management",
    "value chain", "corrective", "literacy",
    # cross-regulation concepts (AI Act ↔ DORA): these connect provisions across
    # different regulations — e.g. incident reporting appears in both.
    "incident", "third-party", "resilience", "testing", "continuity",
    "detection", "governance", "oversight", "ict",
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

    @staticmethod
    def _reg(source: str) -> str:
        """Regulation a node belongs to, inferred from its label prefix."""
        return source.split(" — ", 1)[0] if " — " in source else ""

    def _best_cross(self, source: str, exclude: set[str]) -> tuple[str, list[str]] | None:
        """Best related article from a *different* regulation (most shared concepts)."""
        reg = self._reg(source)
        cross = [(n, sh) for n, sh in self.edges.get(source, {}).items()
                 if n not in exclude and self._reg(n) and self._reg(n) != reg]
        if not cross:
            return None
        node, shared = max(cross, key=lambda kv: len(kv[1]))
        return node, sorted(shared)

    def expand(self, sources: list[str], limit_each: int = 2,
               ensure_cross_regulation: bool = True) -> list[tuple[str, str]]:
        """For seed sources, return (related_source, 'via: concept,...') pairs.

        When a related article comes from a *different* regulation than the seed,
        the link is flagged. With ensure_cross_regulation, at least one
        cross-regulation link per seed is surfaced when one exists — otherwise
        same-regulation neighbours (which usually share more concepts) crowd it
        out, and the cross-regulation insight is the whole point of GraphRAG here."""
        out: list[tuple[str, str]] = []
        seen = set(sources)

        def _emit(s: str, node: str, shared: list[str]) -> None:
            seen.add(node)
            why = "via " + ", ".join(shared)
            if self._reg(s) and self._reg(node) and self._reg(s) != self._reg(node):
                why += " ↔ cross-regulation"
            out.append((node, why))

        for s in sources:
            picked = self.neighbors(s, limit_each)
            for node, shared in picked:
                if node not in seen:
                    _emit(s, node, shared)
            if ensure_cross_regulation and not any(self._reg(s) != self._reg(n)
                                                   for n, _ in picked):
                best = self._best_cross(s, seen)
                if best:
                    _emit(s, *best)
        return out
