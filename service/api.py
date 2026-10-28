"""RegAgent API — ask compliance questions over HTTP, get cited, grounded answers.

This is the product surface: a client POSTs a question, gets back the answer,
the source articles (provenance), a grounding score, and the run cost. The
corpus + indexes are built once at startup.

Run:  uvicorn service.api:app --reload --port 8000
"""
from __future__ import annotations

import os

from fastapi import FastAPI, Header, HTTPException, Depends, Request, UploadFile, File
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from regagent.ingest import load_corpus, load_dora, load_gdpr, load_nis2, load_mica
from regagent.store import DocStore
from regagent.sparse import BM25Index
from regagent.graph import KnowledgeGraph
from regagent.agent import answer_question
from regagent.assess import Profile, build_roadmap, extract_profile
from service.guard import GUARD, RATE

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
           + load_gdpr() + load_nis2() + load_mica())   # multi-regulation corpus
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


def _client_ip(request: Request) -> str:
    """Best-effort client IP, honouring Render's proxy X-Forwarded-For."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    """Per-IP burst guard for the public demo (429 when exceeded)."""
    if not RATE.allow(_client_ip(request)):
        raise HTTPException(
            status_code=429,
            detail="Too many requests from your address. Please slow down and "
                   "try again shortly.",
        )


@app.get("/healthz")
def healthz() -> dict:
    # openai_key_detected: lets us confirm the secret reached the app (the key
    # itself is never exposed — only whether it's present).
    return {"ok": True, "chunks": len(_store.chunks),
            "openai_key_detected": bool(os.environ.get("OPENAI_API_KEY")),
            "spend": GUARD.status()}


@app.post("/ask", dependencies=[Depends(rate_limit)])
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


@app.post("/analyze", dependencies=[Depends(rate_limit)])
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


class AssessIn(BaseModel):
    """A startup product profile for the intake → roadmap flow."""
    description: str = ""
    uses_ai: bool = False
    high_risk_area: str = ""      # "" | hiring | credit | biometrics | education | essential | law
    personal_data: bool = False
    special_categories: bool = False
    automated_decisions: bool = False
    financial_entity: bool = False
    critical_entity: bool = False
    crypto_assets: bool = False
    eu_market: bool = True
    based_in_monaco: bool = False
    lang: str = "en"


@app.post("/assess", dependencies=[Depends(rate_limit)])
def assess(body: AssessIn, who: str = Depends(tenant)) -> dict:
    """Startup profile → applicable EU regulations → grounded, actionable roadmap.

    Applicability is deterministic (rules); the obligations + citations come
    grounded from the corpus via the engine. Decision-support, not legal advice.
    """
    if not GUARD.allowed():
        return {"items": [], "not_applicable": [], "cost_usd": 0.0,
                "demo_limited": True,
                "message": "The shared demo has reached today's budget limit. "
                           "Run RegAgent locally with your own OPENAI_API_KEY (see the README)."}
    prof = Profile(
        description=body.description, uses_ai=body.uses_ai,
        high_risk_area=body.high_risk_area, personal_data=body.personal_data,
        special_categories=body.special_categories, automated_decisions=body.automated_decisions,
        financial_entity=body.financial_entity, critical_entity=body.critical_entity,
        crypto_assets=body.crypto_assets, eu_market=body.eu_market,
        based_in_monaco=body.based_in_monaco,
    )
    def ask(q: str, lang: str):
        return answer_question(_store, q, customer=who, graph=_graph, bm25=_bm25, lang=lang)
    rm = build_roadmap(prof, ask, lang=body.lang)
    GUARD.add(rm.cost_usd)
    return {
        "score": rm.score, "band": rm.band, "jurisdiction": rm.jurisdiction,
        "items": [{"regulation": i.regulation, "applies_reason": i.applies_reason,
                   "severity": i.severity, "obligations": i.obligations,
                   "provisions": i.provisions, "grounded": i.grounded,
                   "needs_review": i.needs_review, "actions": i.actions,
                   "meta": i.meta} for i in rm.items],
        "not_applicable": rm.not_applicable,
        "cost_usd": round(rm.cost_usd, 6),
    }


def _extract_text(name: str, raw: bytes) -> str:
    """Best-effort text extraction from an uploaded PDF / DOCX / TXT."""
    import io
    n = (name or "").lower()
    try:
        if n.endswith(".pdf"):
            from pypdf import PdfReader
            r = PdfReader(io.BytesIO(raw))
            return "\n".join((p.extract_text() or "") for p in r.pages)
        if n.endswith(".docx"):
            import docx
            d = docx.Document(io.BytesIO(raw))
            return "\n".join(p.text for p in d.paragraphs)
        return raw.decode("utf-8", "ignore")
    except Exception:
        return ""


@app.post("/assess/extract", dependencies=[Depends(rate_limit)])
async def assess_extract(file: UploadFile = File(...)) -> dict:
    """Upload a startup document (PDF/DOCX/TXT) → inferred profile to pre-fill the
    form. The user reviews and edits before running the assessment."""
    if not GUARD.allowed():
        return {"ok": False, "message": "The shared demo has reached today's budget limit."}
    raw = await file.read()
    if len(raw) > 4_000_000:
        return {"ok": False, "message": "File too large (max 4 MB)."}
    text = _extract_text(file.filename or "", raw)
    if not text.strip():
        return {"ok": False, "message": "Couldn't read text from this file — try a PDF, DOCX or TXT."}
    p = extract_profile(text)
    GUARD.add(0.001)
    return {"ok": True, "profile": {
        "description": p.description, "uses_ai": p.uses_ai, "high_risk_area": p.high_risk_area,
        "personal_data": p.personal_data, "special_categories": p.special_categories,
        "automated_decisions": p.automated_decisions, "financial_entity": p.financial_entity,
        "critical_entity": p.critical_entity, "crypto_assets": p.crypto_assets,
        "based_in_monaco": p.based_in_monaco,
    }}


class ReportIn(BaseModel):
    """An already-computed assessment result (as returned by /assess) + the
    product description, to render into a downloadable PDF report."""
    description: str = ""
    score: int | None = None
    band: str = ""
    jurisdiction: str = ""
    items: list[dict] = []
    not_applicable: list[str] = []
    cost_usd: float = 0.0


@app.post("/assess/report", dependencies=[Depends(rate_limit)])
def assess_report(body: ReportIn) -> Response:
    """Render the assessment the user already ran into a branded PDF report.

    Pure rendering — no LLM, no re-grounding (so it's free and matches what was
    shown on screen). Decision-support, not legal advice."""
    from regagent.report import build_report_pdf
    pdf = build_report_pdf(body.model_dump())
    return Response(
        content=pdf, media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="RegAgent-compliance-report.pdf"'},
    )


@app.get("/assess", response_class=HTMLResponse)
def assess_page() -> str:
    return _ASSESS_HTML


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


# --- Intake → roadmap page (served at GET /assess) ---------------------------
_ASSESS_HTML = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<title>RegAgent — Assess your startup</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f172a;--card:#fff;--ink:#0f172a;--mut:#64748b;--line:#e2e8f0;
 --accent:#2563eb;--hi:#b91c1c;--hibg:#fef2f2;--med:#b45309;--medbg:#fffbeb;--ok:#15803d}
*{box-sizing:border-box}body{margin:0;background:#f1f5f9;color:var(--ink);
 font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:820px;margin:0 auto;padding:32px 20px 64px}
.top{display:flex;align-items:center;gap:11px;margin-bottom:4px}
.top b{font-size:21px;letter-spacing:-.01em}
.top-right{margin-left:auto;display:flex;align-items:center;gap:14px}
.top a{color:var(--mut);text-decoration:none;font-size:13px}
.langtog{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#fff}
.langtog button{background:#fff;color:var(--mut);border:0;padding:5px 12px;font:13px inherit;font-weight:700;cursor:pointer}
.langtog button.active{background:var(--accent);color:#fff}
.upload{display:flex;align-items:center;gap:10px;flex-wrap:wrap;background:#f8fafc;border:1px dashed #cbd5e1;border-radius:10px;padding:11px 13px;margin:10px 0 4px}
.ghost{background:#fff;color:var(--accent);border:1px solid var(--accent);border-radius:8px;padding:8px 13px;font:13px inherit;font-weight:600;cursor:pointer}
.ghost:hover{background:#eff6ff}
.fname{font-size:13px;color:var(--mut);flex:1;min-width:120px}
.hint{color:var(--mut);font-size:12.5px;margin:2px 0 10px}
.upmsg{font-size:12px;color:var(--mut);display:block;margin:0 0 6px;min-height:14px}
h1{font-size:26px;margin:6px 0 4px}.sub{color:var(--mut);margin:0 0 14px}
.disc{background:#eef2ff;border:1px solid #c7d2fe;color:#3730a3;border-radius:10px;
 padding:10px 13px;font-size:13px;margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 18px;
 box-shadow:0 1px 2px rgba(15,23,42,.05)}
label.q{display:block;font-weight:600;margin:14px 0 6px}
textarea,select{width:100%;border:1px solid var(--line);border-radius:9px;padding:10px 12px;font:inherit;background:#fff}
textarea{min-height:76px;resize:vertical}
.chk{display:flex;gap:9px;align-items:flex-start;margin:9px 0;padding:9px 11px;border:1px solid var(--line);border-radius:9px;cursor:pointer}
.chk input{margin-top:3px}.chk .t{font-weight:600}.chk .d{color:var(--mut);font-size:13px}
.sub-opts{margin:6px 0 4px 22px;padding-left:12px;border-left:2px solid var(--line);display:none}
.row{display:flex;gap:12px;align-items:end;flex-wrap:wrap;margin-top:16px}
button{background:var(--accent);color:#fff;border:0;border-radius:9px;padding:11px 20px;font:inherit;font-weight:600;cursor:pointer}
button:disabled{opacity:.6;cursor:progress}
#out{margin-top:26px;display:none}
.item{background:#fff;border:1px solid var(--line);border-left:5px solid var(--line);border-radius:12px;padding:16px 18px;margin:12px 0}
.item.high{border-left-color:var(--hi)}.item.medium{border-left-color:var(--med)}
.item h3{margin:0 0 4px;font-size:17px;display:flex;align-items:center;gap:9px;flex-wrap:wrap}
.badge{font-size:11px;font-weight:700;padding:2px 8px;border-radius:20px;text-transform:uppercase;letter-spacing:.03em}
.badge.high{background:var(--hibg);color:var(--hi)}.badge.medium{background:var(--medbg);color:var(--med)}
.badge.rev{background:#f1f5f9;color:var(--mut)}
.reason{color:var(--mut);font-size:13.5px;margin:2px 0 10px}
.obl{font-size:14px;white-space:pre-wrap}
.prov{margin:12px 0 0;padding:0;list-style:none}
.prov li{font:13px ui-monospace,Menlo,monospace;background:#f8fafc;border:1px solid var(--line);
 border-radius:7px;padding:6px 10px;margin:5px 0;display:flex;gap:8px}
.prov li::before{content:"§";color:var(--accent);font-weight:700}
.na{color:var(--mut);font-size:13px;margin-top:16px}
.meta{color:var(--mut);font-size:12px;margin-top:8px}
.load{display:none;color:var(--mut);margin-top:16px}
.foot{color:var(--mut);font-size:12px;margin-top:26px;border-top:1px solid var(--line);padding-top:12px}
</style></head><body><div class=wrap>
<div class=top><svg width=38 height=38 viewBox="0 0 48 48" fill=none aria-label=RegAgent style="flex:none">
  <defs><linearGradient id=rgShield x1=24 y1=3 x2=24 y2=46 gradientUnits=userSpaceOnUse>
   <stop stop-color="#3b82f6"/><stop offset=1 stop-color="#2563eb"/></linearGradient></defs>
  <path d="M24 3.5 L40.5 9.5 V24 C40.5 33.8 33.4 41.8 24 44.5 C14.6 41.8 7.5 33.8 7.5 24 V9.5 Z" fill="url(#rgShield)"/>
  <path d="M24 3.5 L40.5 9.5 V24 C40.5 33.8 33.4 41.8 24 44.5 C14.6 41.8 7.5 33.8 7.5 24 V9.5 Z" fill=none stroke="#1d4ed8" stroke-width=1 opacity=".35"/>
  <path d="M16.5 24.3 L21.6 29.4 L31.8 18.2" stroke="#fff" stroke-width=3.4 stroke-linecap=round stroke-linejoin=round/></svg>
 <b>RegAgent</b>
 <div class=top-right>
  <div class=langtog><button type=button data-lang=en>EN</button><button type=button data-lang=fr>FR</button></div>
  <a href="/demo" data-i18n=ask>Ask a question →</a>
 </div></div>
<div style="display:flex;align-items:center;gap:13px;flex-wrap:wrap;margin:6px 0 4px">
<h1 data-i18n=h1 style="margin:0">Assess your startup</h1>
<svg width=34 height=23 viewBox="0 0 54 36" style="flex:none;border-radius:4px;box-shadow:0 1px 3px rgba(15,23,42,.18)" aria-label="European Union">
 <rect width=54 height=36 rx=4 fill="#039"/>
 <g fill="#FC0">
  <path d="M0 -2 L.45 -.62 L1.9 -.62 L.73 .24 L1.18 1.62 L0 .76 L-1.18 1.62 L-.73 .24 L-1.9 -.62 L-.45 -.62 Z" transform="translate(27 6)"/>
  <path d="M0 -2 L.45 -.62 L1.9 -.62 L.73 .24 L1.18 1.62 L0 .76 L-1.18 1.62 L-.73 .24 L-1.9 -.62 L-.45 -.62 Z" transform="translate(33 7.61)"/>
  <path d="M0 -2 L.45 -.62 L1.9 -.62 L.73 .24 L1.18 1.62 L0 .76 L-1.18 1.62 L-.73 .24 L-1.9 -.62 L-.45 -.62 Z" transform="translate(37.39 12)"/>
  <path d="M0 -2 L.45 -.62 L1.9 -.62 L.73 .24 L1.18 1.62 L0 .76 L-1.18 1.62 L-.73 .24 L-1.9 -.62 L-.45 -.62 Z" transform="translate(39 18)"/>
  <path d="M0 -2 L.45 -.62 L1.9 -.62 L.73 .24 L1.18 1.62 L0 .76 L-1.18 1.62 L-.73 .24 L-1.9 -.62 L-.45 -.62 Z" transform="translate(37.39 24)"/>
  <path d="M0 -2 L.45 -.62 L1.9 -.62 L.73 .24 L1.18 1.62 L0 .76 L-1.18 1.62 L-.73 .24 L-1.9 -.62 L-.45 -.62 Z" transform="translate(33 28.39)"/>
  <path d="M0 -2 L.45 -.62 L1.9 -.62 L.73 .24 L1.18 1.62 L0 .76 L-1.18 1.62 L-.73 .24 L-1.9 -.62 L-.45 -.62 Z" transform="translate(27 30)"/>
  <path d="M0 -2 L.45 -.62 L1.9 -.62 L.73 .24 L1.18 1.62 L0 .76 L-1.18 1.62 L-.73 .24 L-1.9 -.62 L-.45 -.62 Z" transform="translate(21 28.39)"/>
  <path d="M0 -2 L.45 -.62 L1.9 -.62 L.73 .24 L1.18 1.62 L0 .76 L-1.18 1.62 L-.73 .24 L-1.9 -.62 L-.45 -.62 Z" transform="translate(16.61 24)"/>
  <path d="M0 -2 L.45 -.62 L1.9 -.62 L.73 .24 L1.18 1.62 L0 .76 L-1.18 1.62 L-.73 .24 L-1.9 -.62 L-.45 -.62 Z" transform="translate(15 18)"/>
  <path d="M0 -2 L.45 -.62 L1.9 -.62 L.73 .24 L1.18 1.62 L0 .76 L-1.18 1.62 L-.73 .24 L-1.9 -.62 L-.45 -.62 Z" transform="translate(16.61 12)"/>
  <path d="M0 -2 L.45 -.62 L1.9 -.62 L.73 .24 L1.18 1.62 L0 .76 L-1.18 1.62 L-.73 .24 L-1.9 -.62 L-.45 -.62 Z" transform="translate(21 7.61)"/>
 </g>
</svg>
</div>
<p class=sub data-i18n=sub>Describe your product. RegAgent identifies which EU regulations apply and builds a grounded, actionable compliance roadmap.</p>
<div class=disc><b data-i18n=disc_b>Decision-support triage, not legal advice.</b> <span data-i18n=disc_r>Requirements are grounded in cited articles; confirm with qualified counsel before relying on them.</span></div>

<div class=card>
 <label class=q for=desc data-i18n=desclabel>Describe your product (a few sentences)</label>
 <textarea id=desc data-i18n-ph=descph placeholder="e.g. A SaaS platform that screens job candidates by scoring their CVs with a machine-learning model, for EU employers."></textarea>
 <div class=upload>
  <input type=file id=doc accept=".pdf,.docx,.txt" hidden>
  <button id=choose type=button class=ghost data-i18n=choose>📎 Choose a file</button>
  <span id=fname class=fname>No file chosen</span>
  <button id=upbtn type=button class=ghost data-i18n=prefill>Read &amp; pre-fill →</button>
 </div>
 <span id=upmsg class=upmsg></span>
 <div class=hint data-i18n=uploadhint>…or upload a pitch / one-pager (PDF, DOCX, TXT) — we read it and fill the form for you to review.</div>

 <label class="chk"><input type=checkbox id=uses_ai><span><span class=t data-i18n=ai_t>Our product uses AI / ML</span><span class="d" data-i18n=ai_d> — models that make predictions, classifications or generate content</span></span></label>
 <div class=sub-opts id=ai_opts>
  <label class=q for=hra data-i18n=hralabel>What does the AI do? (determines risk tier)</label>
  <select id=hra>
   <option value="" data-i18n=o_other>Other / general purpose</option>
   <option value=hiring data-i18n=o_hiring>Hiring, recruitment or worker management</option>
   <option value=credit data-i18n=o_credit>Credit scoring / creditworthiness</option>
   <option value=biometrics data-i18n=o_biometrics>Biometric identification</option>
   <option value=education data-i18n=o_education>Education / exam scoring</option>
   <option value=essential data-i18n=o_essential>Access to essential services (public or private)</option>
   <option value=law data-i18n=o_law>Law enforcement</option>
  </select>
 </div>

 <label class="chk"><input type=checkbox id=pd><span><span class=t data-i18n=pd_t>We process personal data</span><span class="d" data-i18n=pd_d> — any data about identifiable individuals (names, emails, behaviour…)</span></span></label>
 <div class=sub-opts id=pd_opts>
  <label class="chk"><input type=checkbox id=spec><span><span class=t data-i18n=spec_t>…including special-category data</span><span class="d" data-i18n=spec_d> — health, biometrics, ethnicity, beliefs, etc.</span></span></label>
  <label class="chk"><input type=checkbox id=auto><span><span class=t data-i18n=auto_t>…used for solely-automated decisions about people</span></span></label>
 </div>

 <label class="chk"><input type=checkbox id=fin><span><span class=t data-i18n=fin_t>We are a financial entity</span><span class="d" data-i18n=fin_d> — bank, payment, crypto-asset, investment or insurance provider (DORA)</span></span></label>
 <label class="chk"><input type=checkbox id=crit><span><span class=t data-i18n=crit_t>We are an essential / important entity</span><span class="d" data-i18n=crit_d> — energy, health, transport, digital infrastructure, etc. (NIS2)</span></span></label>
 <label class="chk"><input type=checkbox id=crypto><span><span class=t data-i18n=crypto_t>We issue crypto-assets or provide crypto services</span><span class="d" data-i18n=crypto_d> — tokens, stablecoins, exchange, custody, wallet (MiCA)</span></span></label>
 <label class="chk"><input type=checkbox id=monaco><span><span class=t data-i18n=monaco_t>We are based in Monaco</span><span class="d" data-i18n=monaco_d> — shows the Monaco + EU jurisdiction view</span></span></label>

 <div class=row>
  <button id=go data-i18n=go>Generate compliance roadmap →</button>
 </div>
 <div class=load id=load data-i18n=load>Analysing across regulations… grounding each requirement in the corpus. This can take ~20 s.</div>
</div>

<div id=out></div>
<div class=foot data-i18n=foot>RegAgent · EU AI Act · DORA · GDPR · NIS2 · MiCA — grounded, with honest abstention. Not legal advice.</div>
</div>
<script>
const $=id=>document.getElementById(id);
const T={
 en:{ask:"Ask a question →",h1:"Assess your startup",
  sub:"Describe your product. RegAgent identifies which EU regulations apply and builds a grounded, actionable compliance roadmap.",
  disc_b:"Decision-support triage, not legal advice.",
  disc_r:"Requirements are grounded in cited articles; confirm with qualified counsel before relying on them.",
  desclabel:"Describe your product (a few sentences)",
  descph:"e.g. A SaaS platform that screens job candidates by scoring their CVs with a machine-learning model, for EU employers.",
  choose:"📎 Choose a file",nofile:"No file chosen",prefill:"Read & pre-fill →",
  uploadhint:"…or upload a pitch / one-pager (PDF, DOCX, TXT) — we read it and fill the form for you to review.",
  ai_t:"Our product uses AI / ML",ai_d:" — models that make predictions, classifications or generate content",
  hralabel:"What does the AI do? (determines risk tier)",
  o_other:"Other / general purpose",o_hiring:"Hiring, recruitment or worker management",
  o_credit:"Credit scoring / creditworthiness",o_biometrics:"Biometric identification",
  o_education:"Education / exam scoring",o_essential:"Access to essential services (public or private)",
  o_law:"Law enforcement",
  pd_t:"We process personal data",pd_d:" — any data about identifiable individuals (names, emails, behaviour…)",
  spec_t:"…including special-category data",spec_d:" — health, biometrics, ethnicity, beliefs, etc.",
  auto_t:"…used for solely-automated decisions about people",
  fin_t:"We are a financial entity",fin_d:" — bank, payment, crypto-asset, investment or insurance provider (DORA)",
  crit_t:"We are an essential / important entity",crit_d:" — energy, health, transport, digital infrastructure, etc. (NIS2)",
  crypto_t:"We issue crypto-assets or provide crypto services",crypto_d:" — tokens, stablecoins, exchange, custody, wallet (MiCA)",
  monaco_t:"We are based in Monaco",monaco_d:" — shows the Monaco + EU jurisdiction view",
  go:"Generate compliance roadmap →",
  load:"Analysing across regulations… grounding each requirement in the corpus. This can take ~20 s.",
  foot:"RegAgent · EU AI Act · DORA · GDPR · NIS2 · MiCA — grounded, with honest abstention. Not legal advice.",
  exposure:"EU compliance exposure",band_green:"🟢 On track",band_amber:"🟠 Needs attention",band_red:"🔴 High-risk — act now",
  dl:"⬇ Download PDF report",dlhint:"Shareable one-page summary — score, map & remediation",
  whatapplies:"What applies — and what to do",applies1:" applicable regulation(s), highest priority first.",
  remplan:"Remediation plan",needsrev:"needs legal review",inforce:"in force",
  phasein:"Phase-in timeline",upcoming:"(not yet — upcoming)",
  notflagged1:"Not flagged from this profile: ",notflagged2:" — revisit if your product or sector changes.",
  runcost:"Run cost: $",
  scorenote:" · grounded in cited articles · preliminary exposure score, decision-support, not legal advice.",
  noreg:"No EU regulation flagged from the profile given. Add more detail (AI use, personal data, sector).",
  preparing:"Preparing…",upreading:"Reading your document…",
  upfilled:"✓ Filled from your document — review below, then generate.",
  upchoose:"Choose a file first.",upfail:"Could not read the file."},
 fr:{ask:"Poser une question →",h1:"Évaluez votre startup",
  sub:"Décrivez votre produit. RegAgent identifie les réglementations européennes applicables et construit une feuille de route de conformité, fondée et actionnable.",
  disc_b:"Aide à la décision, pas un conseil juridique.",
  disc_r:"Les exigences sont fondées sur des articles cités ; confirmez avec un conseil qualifié avant de vous y fier.",
  desclabel:"Décrivez votre produit (quelques phrases)",
  descph:"ex. une plateforme SaaS qui présélectionne des candidats en notant leurs CV avec un modèle de machine learning, pour des employeurs de l'UE.",
  choose:"📎 Choisir un fichier",nofile:"Aucun fichier choisi",prefill:"Lire & pré-remplir →",
  uploadhint:"…ou importez un pitch / une page de présentation (PDF, DOCX, TXT) — nous le lisons et remplissons le formulaire pour vous.",
  ai_t:"Notre produit utilise l'IA / le ML",ai_d:" — des modèles qui font des prédictions, des classifications ou génèrent du contenu",
  hralabel:"Que fait l'IA ? (détermine le niveau de risque)",
  o_other:"Autre / usage général",o_hiring:"Embauche, recrutement ou gestion du personnel",
  o_credit:"Scoring de crédit / solvabilité",o_biometrics:"Identification biométrique",
  o_education:"Éducation / notation d'examens",o_essential:"Accès aux services essentiels (publics ou privés)",
  o_law:"Maintien de l'ordre",
  pd_t:"Nous traitons des données personnelles",pd_d:" — toute donnée concernant des personnes identifiables (noms, e-mails, comportement…)",
  spec_t:"…dont des données sensibles",spec_d:" — santé, biométrie, origine, convictions, etc.",
  auto_t:"…utilisées pour des décisions entièrement automatisées concernant des personnes",
  fin_t:"Nous sommes une entité financière",fin_d:" — banque, paiement, crypto-actifs, investissement ou assurance (DORA)",
  crit_t:"Nous sommes une entité essentielle / importante",crit_d:" — énergie, santé, transport, infrastructure numérique, etc. (NIS2)",
  crypto_t:"Nous émettons des crypto-actifs ou fournissons des services crypto",crypto_d:" — jetons, stablecoins, échange, conservation, portefeuille (MiCA)",
  monaco_t:"Nous sommes basés à Monaco",monaco_d:" — affiche la vue de juridiction Monaco + UE",
  go:"Générer la feuille de route →",
  load:"Analyse des réglementations… chaque exigence est fondée dans le corpus. Cela peut prendre ~20 s.",
  foot:"RegAgent · AI Act · DORA · RGPD · NIS2 · MiCA — fondé, avec abstention honnête. Pas un conseil juridique.",
  exposure:"Exposition à la conformité UE",band_green:"🟢 Sur la bonne voie",band_amber:"🟠 À surveiller",band_red:"🔴 Risque élevé — agir",
  dl:"⬇ Télécharger le rapport PDF",dlhint:"Résumé partageable d'une page — score, carte & remédiation",
  whatapplies:"Ce qui s'applique — et quoi faire",applies1:" réglementation(s) applicable(s), par ordre de priorité.",
  remplan:"Plan de remédiation",needsrev:"revue juridique requise",inforce:"en vigueur",
  phasein:"Calendrier d'application",upcoming:"(pas encore — à venir)",
  notflagged1:"Non signalé pour ce profil : ",notflagged2:" — à revoir si votre produit ou secteur change.",
  runcost:"Coût d'analyse : $",
  scorenote:" · fondé sur des articles cités · score d'exposition préliminaire, aide à la décision, pas un conseil juridique.",
  noreg:"Aucune réglementation UE signalée à partir du profil. Ajoutez des détails (usage de l'IA, données personnelles, secteur).",
  preparing:"Préparation…",upreading:"Lecture de votre document…",
  upfilled:"✓ Rempli à partir de votre document — vérifiez, puis générez.",
  upchoose:"Choisissez d'abord un fichier.",upfail:"Impossible de lire le fichier."}
};
window._lang='en';
function L(k){return (T[window._lang]||T.en)[k];}
function applyLang(l){
 window._lang=l; document.documentElement.lang=l;
 document.querySelectorAll('[data-i18n]').forEach(function(el){var k=el.getAttribute('data-i18n'); if(T[l][k]!=null) el.textContent=T[l][k];});
 document.querySelectorAll('[data-i18n-ph]').forEach(function(el){var k=el.getAttribute('data-i18n-ph'); if(T[l][k]!=null) el.placeholder=T[l][k];});
 document.querySelectorAll('.langtog button').forEach(function(b){b.classList.toggle('active', b.getAttribute('data-lang')===l);});
 var f=$('doc')&&$('doc').files[0]; if($('fname')) $('fname').textContent=f?f.name:T[l].nofile;
}
$('uses_ai').onchange=e=>$('ai_opts').style.display=e.target.checked?'block':'none';
$('pd').onchange=e=>$('pd_opts').style.display=e.target.checked?'block':'none';
function applyProfile(p){
 $('desc').value=p.description||$('desc').value;
 $('uses_ai').checked=!!p.uses_ai; $('ai_opts').style.display=p.uses_ai?'block':'none';
 if(p.uses_ai&&p.high_risk_area){$('hra').value=p.high_risk_area;}
 $('pd').checked=!!p.personal_data; $('pd_opts').style.display=p.personal_data?'block':'none';
 $('spec').checked=!!p.special_categories; $('auto').checked=!!p.automated_decisions;
 $('fin').checked=!!p.financial_entity; $('crit').checked=!!p.critical_entity;
 $('crypto').checked=!!p.crypto_assets; $('monaco').checked=!!p.based_in_monaco;
}
$('upbtn').onclick=async()=>{
 const f=$('doc').files[0]; const m=$('upmsg');
 if(!f){m.textContent=L('upchoose');return;}
 m.textContent=L('upreading'); $('upbtn').disabled=true;
 try{
  const fd=new FormData(); fd.append('file',f);
  const r=await fetch('/assess/extract',{method:'POST',body:fd});
  const d=await r.json();
  if(!d.ok){m.textContent=d.message||L('upfail');}
  else{applyProfile(d.profile); m.textContent=L('upfilled');}
 }catch(e){m.textContent='Error: '+e;}
 $('upbtn').disabled=false;
};
$('go').onclick=async()=>{
 const btn=$('go'); btn.disabled=true; $('load').style.display='block'; $('out').style.display='none';
 const body={description:$('desc').value,uses_ai:$('uses_ai').checked,
  high_risk_area:$('uses_ai').checked?$('hra').value:'',personal_data:$('pd').checked,
  special_categories:$('pd').checked&&$('spec').checked,automated_decisions:$('pd').checked&&$('auto').checked,
  financial_entity:$('fin').checked,critical_entity:$('crit').checked,
  crypto_assets:$('crypto').checked,based_in_monaco:$('monaco').checked,eu_market:true,lang:window._lang};
 try{
  const r=await fetch('/assess',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d=await r.json(); render(d);
 }catch(err){$('out').innerHTML='<div class=card>Error: '+err+'</div>';$('out').style.display='block';}
 btn.disabled=false; $('load').style.display='none';
};
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function mdbold(s){return esc(s).replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>');}
function bandColor(b){return b==='green'?'#16a34a':(b==='amber'?'#d97706':'#dc2626');}
function bandLabel(b){return L('band_'+b);}
function render(d){
 const o=$('out');
 if(d.demo_limited){o.innerHTML='<div class=card>'+esc(d.message)+'</div>';o.style.display='block';return;}
 if(!d.items||!d.items.length){o.innerHTML='<div class=card>'+esc(L('noreg'))+'</div>';o.style.display='block';return;}
 const bc=bandColor(d.band);
 window._assess=d;
 let h='';
 // Score header
 h+='<div style="display:flex;align-items:center;gap:20px;margin:12px 0 2px;flex-wrap:wrap">'+
  '<div style="width:98px;height:98px;border-radius:50%;flex:none;display:flex;flex-direction:column;align-items:center;justify-content:center;border:7px solid '+bc+'">'+
   '<div style="font-size:31px;font-weight:800;line-height:1;color:'+bc+'">'+(d.score!=null?d.score:'–')+'</div>'+
   '<div style="font-size:11px;color:#64748b">/ 100</div></div>'+
  '<div><div style="font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:.08em;font-weight:700">'+L('exposure')+'</div>'+
   '<div style="font-size:22px;font-weight:800;color:'+bc+'">'+bandLabel(d.band)+'</div>'+
   (d.jurisdiction?'<div style="font-size:13px;color:#64748b;max-width:56ch;margin-top:3px">'+esc(d.jurisdiction)+'</div>':'')+
  '</div></div>';
 // Compliance map (traffic-light chips)
 h+='<div style="display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 4px">';
 for(const it of d.items){const c=it.severity==='high'?'#dc2626':'#d97706';
  h+='<span style="font-size:13px;font-weight:700;padding:6px 12px;border-radius:999px;color:#fff;background:'+c+'">'+esc(it.regulation)+' · '+(it.severity==='high'?'HIGH':'MED')+'</span>';}
 for(const n of (d.not_applicable||[]))
  h+='<span style="font-size:13px;font-weight:600;padding:6px 12px;border-radius:999px;color:#166534;background:#dcfce7">'+esc(n)+' · N/A</span>';
 h+='</div>';
 // Download report
 h+='<div style="margin:16px 0 2px;display:flex;align-items:center;gap:10px;flex-wrap:wrap">'+
    '<button id=dl style="font-size:14px">'+L('dl')+'</button>'+
    '<span style="font-size:12px;color:#64748b">'+esc(L('dlhint'))+'</span></div>';
 // Detail + remediation
 h+='<h2 style="margin:20px 0 2px">'+esc(L('whatapplies'))+'</h2>'+
   '<p class=sub>'+d.items.length+esc(L('applies1'))+'</p>';
 for(const it of d.items){
  const sev=it.severity==='high'?'high':'medium';
  h+='<div class="item '+sev+'"><h3>'+esc(it.regulation)+
     ' <span class="badge '+sev+'">'+sev+'</span>'+
     (it.needs_review?' <span class="badge rev">'+esc(L('needsrev'))+'</span>':'')+
     (it.meta&&it.meta.in_force?' <span style="font-size:11px;color:#64748b;font-weight:500">· '+esc(L('inforce'))+' '+esc(it.meta.in_force)+'</span>':'')+'</h3>'+
     '<div class=reason>'+esc(it.applies_reason)+'</div>'+
     '<div class=obl>'+mdbold(it.obligations)+'</div>';
  if(it.meta&&it.meta.timeline&&it.meta.timeline.length){
   h+='<div style="font-size:11px;font-weight:700;color:#334155;margin:13px 0 4px;text-transform:uppercase;letter-spacing:.06em">'+esc(L('phasein'))+'</div><ul style="list-style:none;margin:0;padding:0">';
   const today=new Date();
   for(const t of it.meta.timeline){const on=new Date(t.iso)<=today;
    h+='<li style="font-size:12.5px;margin:3px 0;color:'+(on?'#166534':'#64748b')+'">'+(on?'✓ ':'⏳ ')+'<b>'+esc(t.date)+'</b> — '+esc(t.what)+(on?'':' '+esc(L('upcoming')))+'</li>';}
   h+='</ul>';
  }
  if(it.actions&&it.actions.length){
   h+='<div style="font-size:11px;font-weight:700;color:#334155;margin:13px 0 4px;text-transform:uppercase;letter-spacing:.06em">'+esc(L('remplan'))+'</div>'+
      '<ol style="margin:0;padding-left:18px">';
   for(const s of it.actions)
    h+='<li style="margin:5px 0;font-size:13.5px">'+esc(s.action)+' <span style="font:12px ui-monospace,Menlo,monospace;color:#2563eb">— '+esc(s.ref)+'</span></li>';
   h+='</ol>';
  }
  if(it.provisions&&it.provisions.length)
   h+='<ul class=prov>'+it.provisions.map(p=>'<li>'+esc(p)+'</li>').join('')+'</ul>';
  h+='</div>';
 }
 if(d.not_applicable&&d.not_applicable.length)
  h+='<div class=na>'+esc(L('notflagged1'))+d.not_applicable.map(esc).join(', ')+esc(L('notflagged2'))+'</div>';
 h+='<div class=meta>'+esc(L('runcost'))+(d.cost_usd||0)+esc(L('scorenote'))+'</div>';
 o.innerHTML=h; o.style.display='block'; o.scrollIntoView({behavior:'smooth'});
 const dlb=$('dl'); if(dlb) dlb.onclick=downloadReport;
}
async function downloadReport(){
 const d=window._assess; if(!d) return;
 const btn=$('dl'); const old=btn.textContent; btn.disabled=true; btn.textContent=L('preparing');
 try{
  const body=Object.assign({description:$('desc').value}, d);
  const r=await fetch('/assess/report',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok) throw new Error('HTTP '+r.status);
  const blob=await r.blob(); const url=URL.createObjectURL(blob);
  const a=document.createElement('a'); a.href=url; a.download='RegAgent-compliance-report.pdf';
  document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
 }catch(e){alert('Could not generate report: '+e);}
 btn.textContent=old; btn.disabled=false;
}
// --- init: language toggle, custom file picker, auto-detect language ---
document.querySelectorAll('.langtog button').forEach(function(b){b.onclick=function(){applyLang(b.getAttribute('data-lang'));};});
$('choose').onclick=function(){$('doc').click();};
$('doc').onchange=function(){var f=$('doc').files[0]; $('fname').textContent=f?f.name:L('nofile');};
applyLang((navigator.language||'').toLowerCase().indexOf('fr')===0?'fr':'en');
</script></body></html>"""
