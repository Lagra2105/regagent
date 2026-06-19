"""RegAgent API — ask compliance questions over HTTP, get cited, grounded answers.

This is the product surface: a client POSTs a question, gets back the answer,
the source articles (provenance), a grounding score, and the run cost. The
corpus + indexes are built once at startup.

Run:  uvicorn service.api:app --reload --port 8000
"""
from __future__ import annotations

import os

from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from regagent.ingest import load_corpus, load_dora, load_gdpr, load_nis2
from regagent.store import DocStore
from regagent.sparse import BM25Index
from regagent.graph import KnowledgeGraph
from regagent.agent import answer_question
from service.guard import GUARD

app = FastAPI(title="RegAgent", version="0.1.0")

# Embed the agentcost cost dashboard at /dashboard — it reads the same SQLite the
# agent writes its per-run costs to (AGENTCOST_DB), so the demo shows RegAgent's
# real economics live. This is the dogfooding loop made visible.
try:
    from agentcost.dashboard import app as _cost_dashboard
    app.mount("/dashboard", _cost_dashboard)
except Exception as _e:  # don't take the API down if the dashboard can't mount
    import sys
    print(f"[regagent] cost dashboard not mounted: {_e}", file=sys.stderr)

# Build the knowledge base once. Use pgvector if DATABASE_URL is set (scales,
# survives restarts), otherwise the in-memory store (zero-setup dev).
_chunks = (load_corpus("data/ai_act.txt") + load_dora()
           + load_gdpr() + load_nis2())   # multi-regulation corpus
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
    lang: str = "auto"   # "auto" | "en" | "fr" — language of the answer


# Multi-tenant API keys for integrations. Set REGAGENT_KEYS to a comma-separated
# allow-list to require an X-API-Key header (each key = one tenant, used for cost
# attribution). Unset = open mode (the public demo). The agent stays hosted —
# customers integrate via the API/MCP without deploying anything in their infra.
_KEYS = {k.strip() for k in os.environ.get("REGAGENT_KEYS", "").split(",") if k.strip()}


def tenant(x_api_key: str | None = Header(default=None)) -> str:
    """Resolve the calling tenant from its X-API-Key (used as the cost customer)."""
    if _KEYS:
        if not x_api_key or x_api_key not in _KEYS:
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        return x_api_key
    return x_api_key or "demo"   # open mode: no key required


@app.get("/healthz")
def healthz() -> dict:
    # openai_key_detected: lets us confirm the secret reached the app (the key
    # itself is never exposed — only whether it's present).
    return {"ok": True, "chunks": len(_store.chunks),
            "openai_key_detected": bool(os.environ.get("OPENAI_API_KEY")),
            "spend": GUARD.status()}


@app.post("/ask")
def ask(body: Ask, who: str = Depends(tenant)) -> dict:
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
    a = answer_question(_store, body.question, customer=who,
                        graph=_graph, bm25=_bm25, lang=body.lang)
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


@app.post("/analyze")
def analyze(body: Ask, who: str = Depends(tenant)) -> dict:
    """Multi-regulation analysis: decompose → answer each → synthesise."""
    if not GUARD.allowed():
        return {"question": body.question,
                "answer": "The shared demo has reached today's budget limit. "
                          "Try again tomorrow, or run RegAgent locally with your "
                          "own OPENAI_API_KEY (see the README).",
                "sub_answers": [], "sources": [], "abstained": True,
                "cost_usd": 0.0, "demo_limited": True}
    from regagent.agent import answer_complex
    a = answer_complex(_store, body.question, customer=who,
                       graph=_graph, bm25=_bm25, lang=body.lang)
    GUARD.add(a.cost_usd)
    return {
        "question": a.question,
        "answer": a.answer,
        "sub_answers": [{"question": s.question, "answer": s.answer,
                         "sources": s.sources, "abstained": s.abstained,
                         "grounding": s.grounding} for s in a.sub_answers],
        "sources": a.sources,
        "abstained": a.abstained,
        "cost_usd": round(a.cost_usd, 6),
    }


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html><html lang=en><head><meta charset=utf-8>
<title>RegAgent — EU compliance agent for AI Act, DORA, GDPR &amp; NIS2</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<meta name=description content="A compliance agent for EU regulation — grounded, cited answers across the EU AI Act, DORA, GDPR and NIS2, with confidence scores and honest abstention.">
<link rel=preconnect href="https://fonts.googleapis.com">
<link rel=preconnect href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel=stylesheet>
<style>
 :root{--bg:#e9edf2;--surface:#fff;--ink:#0f1827;--ink2:#33415a;--muted:#5f6c7e;
  --line:#cdd5df;--brand:#3f5d7d;--brand-ink:#2b4a68;--brand-soft:#e9eef4;
  --grad:linear-gradient(180deg,#ffffff 0,#eef1f5 100%);--bevel:inset 0 1px 0 rgba(255,255,255,.85);
  --steel:linear-gradient(180deg,#5d6e82 0,#46566a 100%)}
 *{box-sizing:border-box} html,body{margin:0}
 body{font-family:Inter,-apple-system,system-ui,Segoe UI,sans-serif;color:var(--ink);line-height:1.6;
  -webkit-font-smoothing:antialiased;background:linear-gradient(180deg,#eef2f6 0,#dde3ea 100%) fixed}
 a{color:var(--brand);text-decoration:none}
 .nav{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:18px;
  padding:13px 22px;background:rgba(238,242,246,.86);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
 .brand{display:flex;align-items:center;gap:9px;font-weight:700;font-size:16px;letter-spacing:-.01em}
 .brand .dot{width:26px;height:26px;border-radius:5px;display:flex;align-items:center;justify-content:center;font-size:15px;
  background:radial-gradient(120% 120% at 30% 20%,#93a2b3,#637184 55%,#4b5868);border:1px solid #3f4b59}
 .nav .sp{flex:1} .nav a.lnk{color:var(--ink2);font-size:13.5px;font-weight:500} .nav a.lnk:hover{color:var(--ink)}
 .btn{display:inline-block;padding:9px 16px;border-radius:5px;font-weight:600;font-size:13.5px;cursor:pointer;
  background:var(--steel);color:#fff;border:1px solid #3a4a5b;box-shadow:inset 0 1px 0 rgba(255,255,255,.22),0 1px 2px rgba(15,23,42,.22)}
 .btn:hover{filter:brightness(1.08)}
 .btn.ghost{background:var(--grad);color:var(--brand-ink);border-color:#cbd4de;box-shadow:var(--bevel)}
 .wrap{max-width:980px;margin:0 auto;padding:0 22px} section{padding:52px 0}
 .hero{text-align:center;padding:72px 0 42px}
 .eyebrow{font-size:12.5px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--brand);margin-bottom:16px}
 h1{font-size:46px;line-height:1.08;letter-spacing:-.03em;margin:0 0 18px;font-weight:700}
 .hero p{font-size:18px;color:var(--ink2);max-width:62ch;margin:0 auto 26px}
 .cta{display:flex;gap:12px;justify-content:center;flex-wrap:wrap}
 .trust{margin-top:28px;font-size:13px;color:var(--muted)}
 h2{font-size:13px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--brand);margin:0 0 10px}
 .lead{font-size:26px;letter-spacing:-.02em;font-weight:700;margin:0 0 14px;max-width:32ch}
 .muted{color:var(--ink2);font-size:16px;max-width:64ch}
 .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:24px}
 .feat{background:var(--grad);border:1px solid var(--line);border-radius:6px;padding:18px;box-shadow:var(--bevel),0 1px 2px rgba(15,23,42,.05)}
 .feat h3{font-size:15px;margin:0 0 6px;font-weight:600} .feat p{font-size:13.5px;color:var(--ink2);margin:0;line-height:1.5}
 .flow{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-top:22px}
 .step{background:var(--grad);border:1px solid var(--line);border-radius:5px;padding:9px 13px;font-size:13px;font-weight:500;box-shadow:var(--bevel)}
 .arr{color:var(--muted)}
 .bench{background:#0f1827;border-radius:8px;padding:30px;color:#fff;text-align:center}
 .bench .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:18px}
 .bench b{display:block;font-size:30px;font-weight:700;letter-spacing:-.02em} .bench .stats span{font-size:12.5px;color:#9aa6b6}
 .bench .note{margin-top:20px;font-size:13px;color:#9aa6b6}
 .two{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:22px}
 .prod{background:var(--grad);border:1px solid var(--line);border-radius:6px;padding:20px;box-shadow:var(--bevel)}
 .prod .k{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:var(--brand);margin-bottom:6px}
 .prod h3{margin:0 0 7px;font-size:18px} .prod p{margin:0;font-size:14px;color:var(--ink2);line-height:1.55}
 .endcta{text-align:center;background:var(--grad);border:1px solid var(--line);border-radius:8px;padding:40px 22px;box-shadow:var(--bevel)}
 footer{padding:30px 22px 50px;text-align:center;color:var(--muted);font-size:13px;border-top:1px solid var(--line)}
 @media(max-width:760px){h1{font-size:33px}.hero p{font-size:16px}.grid{grid-template-columns:1fr}.bench .stats{grid-template-columns:repeat(2,1fr)}.two{grid-template-columns:1fr}.nav a.lnk{display:none}}
</style></head><body>
<div class=nav>
 <span class=brand><span class=dot>🛡️</span> RegAgent</span><span class=sp></span>
 <a class=lnk href="#how">How it works</a><a class=lnk href="#benchmark">Benchmark</a>
 <a class=lnk href="https://github.com/Lagra2105/regagent" target=_blank>GitHub</a>
 <a class=btn href="/demo">Launch demo →</a>
</div>
<section class="hero wrap">
 <div class=eyebrow>EU regulatory compliance · AI Act · DORA · GDPR · NIS2</div>
 <h1>Compliance answers you can audit.</h1>
 <p>RegAgent answers questions across EU regulation — grounded in the exact articles, scored for confidence, linked across laws, and honestly refused when the regulation doesn't cover them. Because in compliance, a confident wrong answer is worse than none.</p>
 <div class=cta><a class=btn href="/demo">Launch live demo →</a><a class="btn ghost" href="https://github.com/Lagra2105/regagent" target=_blank>View the source</a></div>
 <div class=trust>4 regulations · 75 articles · measured accuracy · English &amp; French</div>
</section>
<section class=wrap>
 <h2>The problem</h2>
 <div class=lead>Generic AI paraphrases, invents, and can't show its work.</div>
 <p class=muted>For legal and compliance teams that's a liability, not a tool — you can't file an answer you can't trace to a source. RegAgent is built the other way around: every answer is tied to the articles it came from, scored for how well it's supported, and withheld when the grounding isn't there.</p>
</section>
<section class=wrap>
 <h2>What it does</h2>
 <div class=grid>
  <div class=feat><h3>Provenance</h3><p>Cites the exact articles it used and scores how well the answer is supported — auditable, not a black box.</p></div>
  <div class=feat><h3>Abstention</h3><p>Refuses out-of-scope or weakly-grounded questions instead of guessing. A safe "consult counsel" beats a confident hallucination.</p></div>
  <div class=feat><h3>Hybrid retrieval</h3><p>Dense meaning search and exact-term search, fused and reranked, so the right provision lands on top.</p></div>
  <div class=feat><h3>Knowledge graph</h3><p>Links provisions across regulations — an AI Act duty connected to its DORA, GDPR or NIS2 counterpart.</p></div>
  <div class=feat><h3>Multi-regulation analysis</h3><p>Decomposes "does our system comply?" into focused sub-questions across all four laws, then synthesises one cited answer.</p></div>
  <div class=feat><h3>Measured</h3><p>A published benchmark — recall, citation accuracy, grounding — not a demo-day vibe. Reproducible from the repo.</p></div>
 </div>
</section>
<section class=wrap id=how>
 <h2>How it works</h2>
 <div class=lead>A controlled agent pipeline — every step decided, every step measured.</div>
 <div class=flow>
  <span class=step>Classify</span><span class=arr>→</span><span class=step>Hybrid retrieval</span><span class=arr>→</span>
  <span class=step>Abstain gate</span><span class=arr>→</span><span class=step>Graph expand</span><span class=arr>→</span>
  <span class=step>Answer</span><span class=arr>→</span><span class=step>Verify + grounding score</span>
 </div>
 <p class=muted style="margin-top:18px">It plans, retrieves, decides whether to answer or refuse, expands across regulations, then checks its own work. Not a chatbot — an agent that makes decisions and can say "I don't know."</p>
</section>
<section class=wrap id=benchmark>
 <h2>Measured, not claimed</h2>
 <div class=bench>
  <div class=stats>
   <div><b>98%</b><span>retrieval recall@4</span></div><div><b>98%</b><span>citation recall</span></div>
   <div><b>0.89</b><span>grounding</span></div><div><b>4</b><span>regulations</span></div>
  </div>
  <div class=note>42-question golden set across AI Act, DORA, GDPR &amp; NIS2 · reproducible offline · higher with production embeddings</div>
 </div>
</section>
<section class=wrap>
 <h2>The bigger picture</h2>
 <div class=lead>Two products, one loop.</div>
 <div class=two>
  <div class=prod><div class=k>The product</div><h3>RegAgent</h3><p>The compliance agent itself — trusted answers across EU regulation, available as a REST API and a UI.</p></div>
  <div class=prod><div class=k>The infrastructure</div><h3>agentcost</h3><p>Cost &amp; economics tracking for AI agents. RegAgent runs on it — surfacing the real cost per trusted answer, and catching real bugs along the way. The agent is the product; agentcost is the layer beneath it.</p></div>
 </div>
</section>
<section class=wrap>
 <h2>Integrations</h2>
 <div class=lead>Plug it into your environment — without deploying it.</div>
 <p class=muted>Call the REST API, or drop in the MCP server to use RegAgent as a tool inside Claude, an IDE assistant, or your own agent stack. The agent stays hosted — no model, no corpus and no data in your infrastructure.</p>
 <div class=flow style="margin-top:18px">
  <span class=step>REST API</span><span class=step>MCP server</span><span class=step>Multi-tenant API keys</span><span class=step>English &amp; French</span>
 </div>
</section>
<section class=wrap>
 <div class=endcta>
  <div class=lead style="font-size:24px;margin:0 auto 18px">See it answer a real compliance question.</div>
  <div class=cta><a class=btn href="/demo">Launch live demo →</a><a class="btn ghost" href="/dashboard/">View the cost dashboard</a></div>
 </div>
</section>
<footer>Built by a Senior Data Scientist &amp; MLOps engineer · <a href="https://github.com/Lagra2105/regagent" target=_blank>GitHub</a> · instrumented by agentcost</footer>
</body></html>"""


@app.get("/demo", response_class=HTMLResponse)
def demo() -> str:
    return """<!doctype html><html lang=en><head><meta charset=utf-8>
<title>RegAgent — live demo (AI Act · DORA · GDPR · NIS2)</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<link rel=preconnect href="https://fonts.googleapis.com">
<link rel=preconnect href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel=stylesheet>
<style>
 :root{--bg:#e9edf2;--surface:#fff;--ink:#0f1827;--ink2:#33415a;--muted:#5f6c7e;
  --line:#cdd5df;--brand:#3f5d7d;--brand-ink:#2b4a68;--brand-soft:#e9eef4;
  --ok:#0f7b52;--ok-bg:#e6f2ec;--warn:#8a571c;--warn-bg:#f7f0e3;--bad:#bb433d;
  --info:#2563eb;--info-bg:#eaf0fa;
  --surface-grad:linear-gradient(180deg,#ffffff 0,#eef1f5 100%);
  --bevel:inset 0 1px 0 rgba(255,255,255,.85);
  --shadow:0 1px 2px rgba(15,23,42,.06),0 10px 26px rgba(15,23,42,.08);--r:6px}
 *{box-sizing:border-box}
 html,body{margin:0}
 body{font-family:Inter,-apple-system,system-ui,Segoe UI,sans-serif;color:var(--ink);
  line-height:1.55;-webkit-font-smoothing:antialiased;min-height:100vh;
  background:linear-gradient(180deg,#eef2f6 0,#dde3ea 100%) fixed}
 .wrap{max-width:780px;margin:0 auto;padding:52px 22px 96px}
 .hero{display:flex;align-items:center;gap:14px;margin-bottom:18px}
 .badge{width:48px;height:48px;border-radius:7px;flex:none;
  display:flex;align-items:center;justify-content:center;font-size:24px;
  background:radial-gradient(120% 120% at 30% 20%,#93a2b3 0,#637184 55%,#4b5868 100%);
  border:1px solid #3f4b59;
  box-shadow:inset 0 1px 1px rgba(255,255,255,.4),inset 0 -2px 5px rgba(0,0,0,.18),0 2px 5px rgba(15,23,42,.22)}
 h1{font-size:27px;font-weight:700;letter-spacing:-.02em;margin:0}
 .tag{color:var(--muted);font-size:13.5px;font-weight:500;margin-top:2px}
 .lede{color:var(--ink2);font-size:15.5px;margin:0 0 18px;max-width:66ch}
 .pills{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:20px}
 .pill{font-size:11.5px;font-weight:500;color:var(--brand-ink);background:var(--brand-soft);border-radius:5px;padding:5px 11px}
 .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:11px}
 .stat{background:var(--surface-grad);border:1px solid var(--line);border-radius:6px;padding:13px 14px;box-shadow:var(--bevel),0 1px 2px rgba(15,23,42,.05)}
 .stat b{display:block;font-size:21px;font-weight:700;letter-spacing:-.01em}
 .stat span{font-size:11.5px;color:var(--muted)}
 .benchnote{font-size:12.5px;color:var(--muted);margin:0 0 26px}
 .benchnote a{color:var(--brand);font-weight:600;text-decoration:none}
 .panel{background:var(--surface-grad);border:1px solid var(--line);border-radius:var(--r);padding:14px;box-shadow:var(--bevel),var(--shadow)}
 textarea{width:100%;min-height:78px;padding:12px;border:1px solid var(--line);border-radius:6px;
  font:inherit;color:var(--ink);resize:vertical;outline:none;transition:border-color .15s,box-shadow .15s;
  box-shadow:inset 0 1px 2px rgba(15,23,42,.06)}
 textarea:focus{border-color:var(--brand);box-shadow:0 0 0 3px rgba(43,82,120,.15)}
 .toolbar{display:flex;align-items:center;gap:10px;margin-top:11px;flex-wrap:wrap}
 .btns{display:flex;gap:9px;flex-wrap:wrap}
 button{font-family:inherit;padding:11px 18px;border:1px solid #3a4a5b;border-radius:5px;
  background:linear-gradient(180deg,#5d6e82 0,#46566a 100%);color:#fff;font-weight:600;font-size:14px;
  cursor:pointer;transition:filter .12s,transform .05s,box-shadow .12s;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.22),0 1px 2px rgba(15,23,42,.22)}
 button:hover{filter:brightness(1.08)}
 button:active{transform:translateY(1px);box-shadow:inset 0 2px 4px rgba(0,0,0,.28)}
 button:disabled{opacity:.5;cursor:default;filter:none}
 button.secondary{background:linear-gradient(180deg,#fbfcfd 0,#e8edf2 100%);color:var(--brand-ink);
  border-color:#cbd4de;box-shadow:var(--bevel)}
 button.secondary:hover{border-color:var(--brand);filter:none}
 select#lang{margin-left:auto;padding:10px 12px;border:1px solid #cbd4de;border-radius:5px;font:inherit;
  background:linear-gradient(180deg,#fbfcfd 0,#e8edf2 100%);color:var(--ink);cursor:pointer;box-shadow:var(--bevel)}
 .exlbl{font-size:12px;font-weight:600;color:var(--muted);margin:22px 0 9px;text-transform:uppercase;letter-spacing:.05em}
 .exwrap{display:flex;flex-wrap:wrap;gap:8px}
 .ex{font-size:12.5px;color:var(--ink2);border:1px solid var(--line);background:var(--surface);
  border-radius:5px;padding:7px 13px;cursor:pointer;transition:.15s}
 .ex:hover{border-color:var(--brand);color:var(--brand-ink);background:var(--brand-soft)}
 .card{margin-top:24px;background:var(--surface-grad);border:1px solid var(--line);border-radius:var(--r);padding:22px;display:none;box-shadow:var(--bevel),var(--shadow)}
 .ans{font-size:15.5px;line-height:1.65;color:var(--ink);white-space:pre-wrap}
 .banner{display:inline-block;border-radius:5px;padding:8px 13px;font-size:13px;font-weight:600;margin-bottom:15px}
 .b-ok{background:var(--ok-bg);color:var(--ok)} .b-abs{background:var(--warn-bg);color:var(--warn)}
 .lbl{font-size:11px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);margin:18px 0 8px}
 .src{display:inline-block;background:var(--brand-soft);color:var(--brand-ink);border-radius:5px;padding:4px 11px;margin:0 6px 6px 0;font-size:12.5px;font-weight:500}
 .gsrc{background:var(--info-bg);color:var(--info)}
 .weak{font-size:13px;color:var(--warn);background:var(--warn-bg);border-radius:5px;padding:8px 12px;margin:6px 0}
 .subq{border:1px solid var(--line);border-radius:6px;padding:14px 15px;margin:9px 0;background:#fcfcfe}
 .subq .q{font-weight:600;font-size:14px;margin-bottom:6px}
 .subq .a{font-size:13.5px;line-height:1.55;color:var(--ink2)}
 .synth{border:1px solid var(--brand);background:var(--brand-soft);border-radius:7px;padding:17px;margin-top:8px}
 .synth .h{font-size:11px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:var(--brand-ink);margin-bottom:8px}
 .meta{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:13px;margin-top:18px;padding-top:13px;border-top:1px solid var(--line)}
 .meta b{color:var(--ink)} .g-ok{color:var(--ok);font-weight:600} .g-bad{color:var(--bad);font-weight:600}
 .foot{margin-top:30px;color:var(--muted);font-size:12.5px;text-align:center}
 a{color:var(--brand)}
 @media(max-width:560px){.stats{grid-template-columns:repeat(2,1fr)}select#lang{margin-left:0}}
</style></head><body><div class=wrap>
<header class=hero><div class=badge>🛡️</div><div>
 <h1>RegAgent</h1>
 <div class=tag>EU compliance agent · AI Act · DORA · GDPR · NIS2</div>
</div></header>
<p class=lede>Grounded answers with exact article citations, a confidence score, and an honest refusal when the regulation doesn't cover the question — because in compliance a confident wrong answer is worse than none. Ask in English or French.</p>
<div class=pills>
 <span class=pill>hybrid retrieval</span><span class=pill>knowledge graph</span><span class=pill>provenance</span><span class=pill>abstention</span><span class=pill>multi-regulation</span><span class=pill>cost-tracked</span>
</div>
<div class=stats>
 <div class=stat><b>98%</b><span>retrieval recall@4</span></div>
 <div class=stat><b>98%</b><span>citation recall</span></div>
 <div class=stat><b>0.89</b><span>grounding</span></div>
 <div class=stat><b>4</b><span>regulations</span></div>
</div>
<div class=benchnote>Reproducible offline benchmark · 42-question golden set — higher with production embeddings. <a href="/dashboard/">Live cost dashboard →</a></div>
<div class=panel>
 <textarea id=q placeholder="Ask a compliance question — e.g. Is social scoring of citizens allowed under the AI Act?"></textarea>
 <div class=toolbar>
  <div class=btns><button id=btn onclick=ask()>Ask</button><button id=btn2 class=secondary onclick=analyze()>Analyze across regulations</button></div>
  <select id=lang title="Answer language"><option value=auto>Auto</option><option value=en>English</option><option value=fr>Français</option></select>
 </div>
</div>
<div class=exlbl>Try one</div>
<div class=exwrap id=ex></div>
<div class=card id=out></div>
<div class=foot>Built by a Senior Data Scientist / MLOps engineer · <a href="https://github.com/Lagra2105/regagent" target=_blank>source on GitHub</a> · instrumented by agentcost</div>
<script>
const EXAMPLES=[
 "Is social scoring of citizens allowed under the AI Act?",
 "What are the transparency obligations when users interact with an AI system?",
 "What must providers of high-risk AI systems do for risk management?",
 "When must a major ICT incident be reported under DORA?",  // second regulation
 "Can a decision about me be made by an algorithm alone under GDPR?",  // third regulation
 "What are the incident reporting deadlines under NIS2?",  // fourth regulation
 "Does our AI credit-scoring system comply with EU law on automated decisions, incident reporting and risk management?",  // multi-reg: use Analyze
 "Le score social des citoyens est-il autorisé par le règlement IA ?",  // French — set language to Auto/Français
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
  try{ const r=await fetch('/ask',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({question:q,lang:document.getElementById('lang').value})}); d=await r.json(); }
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

async function analyze(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  const out=document.getElementById('out'), b1=document.getElementById('btn'), b2=document.getElementById('btn2');
  out.style.display='block';
  out.innerHTML='<div class=ans>Planning sub-questions, answering each across regulations, synthesising…</div>';
  b1.disabled=b2.disabled=true;
  let d;
  try{ const r=await fetch('/analyze',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({question:q,lang:document.getElementById('lang').value})}); d=await r.json(); }
  catch(e){ out.innerHTML='<div class=ans>Network error — please retry.</div>'; b1.disabled=b2.disabled=false; return; }
  b1.disabled=b2.disabled=false;

  let banner = d.abstained
    ? '<div class="banner b-abs">⚠ Out of scope across all regulations</div>'
    : '<div class="banner b-ok">✓ Analysed across regulations — '+(d.sub_answers||[]).length+' sub-questions</div>';

  let subs = (d.sub_answers||[]).map(s=>{
    const tag = s.abstained ? '<span class=g-bad>abstained</span>' : '<span class=g-ok>grounding '+s.grounding+'</span>';
    const cites = (s.sources&&s.sources.length) ? s.sources.map(x=>'<span class=src>'+x+'</span>').join('') : '';
    return '<div class=subq><div class=q>'+s.question+'</div><div class=a>'+s.answer+'</div>'
         + '<div style="margin-top:6px">'+cites+'</div><div class=meta style="border:0;padding:4px 0 0">'+tag+'</div></div>';
  }).join('');

  let synth = '<div class=synth><div class=h>Synthesised answer</div><div class=ans>'+d.answer+'</div></div>';

  out.innerHTML = banner
    + '<div class=lbl>Decomposed sub-questions</div>' + subs
    + synth
    + '<div class=meta><span>Total run cost <b>$'+d.cost_usd+'</b></span>'
    + '<span style="margin-left:auto"><a href="https://github.com/Lagra2105/regagent" target=_blank>source ↗</a></span></div>';
}
</script></div></body></html>"""
