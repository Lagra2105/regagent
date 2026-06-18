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
from service.guard import GUARD

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
    return {"ok": True, "chunks": len(_store.chunks), "spend": GUARD.status()}


@app.post("/ask")
def ask(body: Ask) -> dict:
    # Public-demo budget guard: stop calling the model once the daily cap is hit.
    if not GUARD.allowed():
        return {
            "question": body.question,
            "answer": "The shared demo has reached today's budget limit. "
                      "Please try again tomorrow, or run RegAgent locally with "
                      "your own OPENAI_API_KEY (see the README).",
            "sources": [], "graph_sources": [], "grounding": 0.0,
            "grounded": False, "abstained": True, "weak_claims": [],
            "cost_usd": 0.0, "demo_limited": True,
        }
    a = answer_question(_store, body.question, customer=body.customer,
                        graph=_graph, bm25=_bm25)
    GUARD.add(a.cost_usd)
    return {
        "question": a.question,
        "answer": a.answer,
        "sources": a.sources,
        "graph_sources": a.graph_sources,
        "grounding": a.grounding,
        "grounded": a.grounding >= 0.35,
        "abstained": a.abstained,
        "weak_claims": [c.claim for c in (a.provenance.weak if a.provenance else [])],
        "cost_usd": round(a.cost_usd, 6),
    }


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html><meta charset=utf-8><title>RegAgent — EU AI Act compliance agent</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 :root{--ink:#16181d;--muted:#6b7280;--line:#e8e8ef;--brand:#6d4aff;--brandsoft:#f1f0fb}
 *{box-sizing:border-box}
 body{font-family:-apple-system,system-ui,Segoe UI,sans-serif;max-width:760px;margin:0 auto;padding:40px 20px 80px;color:var(--ink)}
 h1{font-size:24px;margin:0 0 4px} .sub{color:var(--muted);margin-bottom:8px;line-height:1.5}
 .pill{display:inline-block;font-size:11px;color:var(--brand);background:var(--brandsoft);border-radius:999px;padding:3px 10px;margin:0 6px 0 0}
 .pills{margin:14px 0 22px}
 textarea{width:100%;height:74px;padding:13px;border:1px solid var(--line);border-radius:12px;font:inherit;resize:vertical}
 .row{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}
 button{padding:11px 20px;border:0;border-radius:10px;background:var(--brand);color:#fff;font-weight:600;cursor:pointer;font-size:14px}
 button:disabled{opacity:.5;cursor:default}
 .ex{font-size:12.5px;color:var(--muted);border:1px solid var(--line);background:#fff;border-radius:999px;padding:6px 12px;cursor:pointer}
 .ex:hover{border-color:var(--brand);color:var(--brand)}
 .exwrap{margin:14px 0 4px;display:flex;gap:8px;flex-wrap:wrap}
 .exlbl{font-size:12px;color:var(--muted);margin:18px 0 2px}
 .card{margin-top:22px;border:1px solid var(--line);border-radius:16px;padding:20px;display:none}
 .ans{font-size:15.5px;line-height:1.6;white-space:pre-wrap}
 .banner{border-radius:10px;padding:10px 13px;font-size:13.5px;margin:0 0 14px;font-weight:600}
 .b-ok{background:#eafaf1;color:#0f7a43}.b-abs{background:#fff4e5;color:#b3590a}
 .lbl{font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin:16px 0 7px}
 .src{display:inline-block;background:var(--brandsoft);color:#5b3df5;border-radius:999px;padding:3px 11px;margin:0 5px 5px 0;font-size:12.5px}
 .gsrc{background:#eef6ff;color:#1666c8}
 .weak{font-size:13px;color:#b3590a;background:#fff7ed;border-left:3px solid #f0a35e;padding:7px 11px;border-radius:0 8px 8px 0;margin:5px 0}
 .meta{color:var(--muted);font-size:13px;margin-top:16px;border-top:1px solid var(--line);padding-top:12px;display:flex;gap:18px;flex-wrap:wrap}
 .meta b{color:var(--ink)} .g-ok{color:#0f9d58}.g-bad{color:#e5484d}
 a{color:var(--brand)}
</style>
<h1>🛡️ RegAgent</h1>
<div class=sub>A compliance agent for the <b>EU AI Act</b>. Every answer is grounded in the exact articles, scored for how well it's supported, and refused when the regulation doesn't cover it — because in compliance a confident wrong answer is worse than none.</div>
<div class=pills>
 <span class=pill>hybrid retrieval</span><span class=pill>knowledge graph</span><span class=pill>provenance</span><span class=pill>abstention</span><span class=pill>cost-tracked</span>
</div>
<textarea id=q placeholder="e.g. Is social scoring of citizens allowed under the AI Act?"></textarea>
<div class=row><button id=btn onclick=ask()>Ask</button></div>
<div class=exlbl>Try one:</div>
<div class=exwrap id=ex></div>
<div class=card id=out></div>
<script>
const EXAMPLES=[
 "Is social scoring of citizens allowed under the AI Act?",
 "What are the transparency obligations when users interact with an AI system?",
 "What must providers of high-risk AI systems do for risk management?",
 "Do I need to keep technical documentation for a high-risk system?",
 "What is the best recipe for a chocolate cake?"  // out-of-scope: watch it abstain
];
const exwrap=document.getElementById('ex');
EXAMPLES.forEach(t=>{const b=document.createElement('button');b.className='ex';
  b.textContent=t.length>52?t.slice(0,50)+'…':t;b.title=t;
  b.onclick=()=>{document.getElementById('q').value=t;ask();};exwrap.appendChild(b);});

async function ask(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  const out=document.getElementById('out'), btn=document.getElementById('btn');
  out.style.display='block'; out.innerHTML='<div class=ans>Retrieving, reasoning, checking grounding…</div>';
  btn.disabled=true;
  let d;
  try{ const r=await fetch('/ask',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({question:q})}); d=await r.json(); }
  catch(e){ out.innerHTML='<div class=ans>Network error — please retry.</div>'; btn.disabled=false; return; }
  btn.disabled=false;

  let banner = d.abstained
    ? '<div class="banner b-abs">⚠ Abstained — not enough grounding to answer safely</div>'
    : '<div class="banner b-ok">✓ Answer grounded in cited articles</div>';

  let srcs = (d.sources&&d.sources.length)
    ? '<div class=lbl>Cited articles (provenance)</div>'+d.sources.map(s=>'<span class=src>'+s+'</span>').join('') : '';
  let gsrcs = (d.graph_sources&&d.graph_sources.length)
    ? '<div class=lbl>Related via knowledge graph</div>'+d.graph_sources.map(s=>'<span class="src gsrc">'+s+'</span>').join('') : '';
  let weak = (d.weak_claims&&d.weak_claims.length)
    ? '<div class=lbl>Weakly-supported claims (flagged for review)</div>'+d.weak_claims.map(c=>'<div class=weak>'+c+'</div>').join('') : '';

  const g = d.grounded ? '<span class=g-ok>'+d.grounding+'</span>' : '<span class=g-bad>'+d.grounding+'</span>';
  out.innerHTML = banner
    + '<div class=ans>'+d.answer+'</div>'
    + srcs + gsrcs + weak
    + '<div class=meta><span>Grounding <b>'+g+'</b></span><span>Run cost <b>$'+d.cost_usd+'</b></span>'
    + '<span style="margin-left:auto"><a href="https://github.com/Lagra2105/regagent" target=_blank>source ↗</a></span></div>';
}
</script>"""
