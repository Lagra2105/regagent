"""RegAgent MCP server — expose the hosted RegAgent as tools for any MCP host
(Claude Desktop, an agent framework, the customer's own AI stack).

It is a thin client: it calls the hosted RegAgent API over HTTPS, so nothing —
no model, no corpus, no data — runs in the customer's infrastructure. That is
the integration story: plug the agent into your environment without deploying it.

Config (env):
  REGAGENT_URL      base URL of the hosted RegAgent (default: the public demo)
  REGAGENT_API_KEY  optional X-API-Key for a tenant deployment

Run:  python regagent_mcp.py        (stdio transport, for an MCP host to launch)
"""
from __future__ import annotations

import os

import httpx
from mcp.server.fastmcp import FastMCP

BASE = os.environ.get("REGAGENT_URL", "https://Lagra21-regagent.hf.space").rstrip("/")
API_KEY = os.environ.get("REGAGENT_API_KEY")
_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

mcp = FastMCP("regagent")


def _post(path: str, payload: dict) -> dict:
    with httpx.Client(timeout=120) as client:
        r = client.post(f"{BASE}{path}", json=payload, headers=_HEADERS)
        r.raise_for_status()
        return r.json()


@mcp.tool()
def ask_regulation(question: str, lang: str = "auto") -> dict:
    """Answer a compliance question grounded in EU regulation (EU AI Act, DORA,
    GDPR, NIS2).

    Returns the answer, the exact cited articles, a grounding score, and whether
    the agent abstained (it refuses rather than guess when the law doesn't cover
    the question). lang: "auto", "en" or "fr".
    """
    d = _post("/ask", {"question": question, "lang": lang})
    return {
        "answer": d.get("answer"),
        "cited_articles": d.get("sources", []),
        "related_via_graph": d.get("graph_sources", []),
        "grounding": d.get("grounding"),
        "abstained": d.get("abstained"),
    }


@mcp.tool()
def analyze_compliance(question: str, lang: str = "auto") -> dict:
    """Analyse a real-world compliance question across EU regulations.

    The agent decomposes the question into focused sub-questions, answers each
    with citations across the EU AI Act, DORA, GDPR and NIS2, and synthesises one
    grounded answer. Use for broad "does our system comply?" questions.
    lang: "auto", "en" or "fr".
    """
    d = _post("/analyze", {"question": question, "lang": lang})
    return {
        "answer": d.get("answer"),
        "sub_findings": [
            {"question": s.get("question"), "answer": s.get("answer"),
             "cited_articles": s.get("sources", []), "abstained": s.get("abstained")}
            for s in d.get("sub_answers", [])
        ],
        "all_cited_articles": d.get("sources", []),
    }


if __name__ == "__main__":
    mcp.run()
