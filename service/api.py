"""RegAgent API — ask compliance questions over HTTP, get cited, grounded answers.

This is the product surface: a client POSTs a question, gets back the answer,
the source articles (provenance), a grounding score, and the run cost. The
corpus + indexes are built once at startup.

Run:  uvicorn service.api:app --reload --port 8000
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from regagent.ingest import load_corpus
from regagent.store import DocStore
from regagent.sparse import BM25Index
from regagent.graph import KnowledgeGraph
from regagent.agent import answer_question

app = FastAPI(title="RegAgent", version="0.1.0")

# Build the knowledge base once. Use pgvector if DATABASE_URL is set (scales,
# survives restarts), otherwise the in-memory store (zero-setup dev).
_chunks = load_corpus("data/ai_act.txt")     # falls back to built-in sample
if os.environ.get("DATABASE_URL"):
    from regagent.store_pg import PgVectorStore
    _store = PgVectorStore(os.environ["DATABASE_URL"])
else:
    _store = DocStore()
_store.add(_chunks)
_bm25 = BM25Index().build(_chunks)
_graph = KnowledgeGraph().build(_chunks)


class Ask(BaseModel):
    question: str
    customer: str = "demo"


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "chunks": len(_store.chunks)}


@app.post("/ask")
def ask(body: Ask) -> dict:
    a = answer_question(_store, body.question, customer=body.customer,
                        graph=_graph, bm25=_bm25)
    return {
        "question": a.question,
        "answer": a.answer,
        "sources": a.sources,
        "graph_sources": a.graph_sources,
        "grounding": a.grounding,
        "grounded": a.grounding >= 0.35,
        "weak_claims": [c.claim for c in (a.provenance.weak if a.provenance else [])],
        "cost_usd": round(a.cost_usd, 6),
    }


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html><meta charset=utf-8><title>RegAgent</title>
<style>
 body{font-family:-apple-system,system-ui,sans-serif;max-width:720px;margin:48px auto;padding:0 20px;color:#16181d}
 h1{font-size:22px} .sub{color:#6b7280;margin-bottom:24px}
 textarea{width:100%;height:70px;padding:12px;border:1px solid #e5e7eb;border-radius:12px;font:inherit}
 button{margin-top:10px;padding:10px 18px;border:0;border-radius:10px;background:#6d4aff;color:#fff;font-weight:600;cursor:pointer}
 .card{margin-top:20px;border:1px solid #e5e7eb;border-radius:14px;padding:18px;display:none}
 .src{display:inline-block;background:#f1f0fb;color:#5b3df5;border-radius:999px;padding:2px 10px;margin:3px 4px 0 0;font-size:12px}
 .meta{color:#6b7280;font-size:13px;margin-top:10px}
 .g-ok{color:#0f9d58;font-weight:600}.g-bad{color:#e5484d;font-weight:600}
</style>
<h1>🛡️ RegAgent</h1>
<div class=sub>Ask a compliance question about the EU AI Act — answered with cited articles and a grounding score.</div>
<textarea id=q placeholder="e.g. Is social scoring of citizens allowed under the AI Act?"></textarea><br>
<button onclick=ask()>Ask</button>
<div class=card id=out></div>
<script>
async function ask(){
  const q=document.getElementById('q').value; if(!q)return;
  const out=document.getElementById('out'); out.style.display='block'; out.innerHTML='Thinking…';
  const r=await fetch('/ask',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({question:q})});
  const d=await r.json();
  const g = d.grounded ? '<span class=g-ok>grounded '+d.grounding+'</span>' : '<span class=g-bad>weak '+d.grounding+' — review</span>';
  out.innerHTML = '<div>'+d.answer+'</div>'
    + '<div style="margin-top:12px">'+d.sources.map(s=>'<span class=src>'+s+'</span>').join('')+'</div>'
    + '<div class=meta>Grounding: '+g+' · cost $'+d.cost_usd+'</div>';
}
</script>"""
