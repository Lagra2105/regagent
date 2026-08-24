"""Startup-facing layer: a product profile → applicable EU regulations → an
actionable compliance roadmap.

Built ON the RegAgent engine, not around it: applicability is a deterministic
rules triage (no LLM — reliable and auditable), while the concrete obligations
and their article citations come *grounded* from the corpus via the engine's
answer function. Where the engine can't ground an item, we surface an honest
"flag for legal review" instead of inventing requirements.

This is decision-support, not legal advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# --- Annex III high-risk AI use areas (simplified triage — not an exhaustive list) ---
HIGH_RISK_AREAS: dict[str, str] = {
    "hiring":     "Recruitment & worker management (AI Act, Annex III §4)",
    "credit":     "Creditworthiness / credit scoring (AI Act, Annex III §5b)",
    "biometrics": "Biometric identification & categorisation (AI Act, Annex III §1)",
    "education":  "Education & vocational training (AI Act, Annex III §3)",
    "essential":  "Access to essential public/private services (AI Act, Annex III §5)",
    "law":        "Law enforcement (AI Act, Annex III §6)",
}

ALL_REGS = ["EU AI Act", "GDPR", "DORA", "NIS2"]


@dataclass
class Profile:
    description: str = ""
    uses_ai: bool = False
    high_risk_area: str = ""        # a key of HIGH_RISK_AREAS, or "" (unknown / not high-risk)
    personal_data: bool = False
    special_categories: bool = False
    automated_decisions: bool = False
    financial_entity: bool = False
    critical_entity: bool = False   # essential/important entity in a NIS2 sector
    eu_market: bool = True


@dataclass
class RoadmapItem:
    regulation: str
    applies_reason: str
    severity: str            # "high" | "medium"
    obligations: str         # grounded summary from the engine (or the review note)
    provisions: list[str]    # cited articles → each becomes a checklist line
    grounded: bool
    needs_review: bool       # engine abstained / weak grounding → human/legal review


@dataclass
class Roadmap:
    profile: Profile
    items: list[RoadmapItem] = field(default_factory=list)
    not_applicable: list[str] = field(default_factory=list)
    cost_usd: float = 0.0


def applicable(p: Profile) -> list[dict]:
    """Deterministic applicability triage → which regulations apply and why.

    Rules, not the LLM: this is the part that must be predictable and auditable.
    """
    out: list[dict] = []
    if p.uses_ai and p.eu_market:
        hr = HIGH_RISK_AREAS.get(p.high_risk_area)
        out.append({
            "reg": "EU AI Act",
            "high_risk": bool(hr),
            "reason": ("You place an AI system on the EU market. "
                       + (f"This use is HIGH-RISK: {hr}."
                          if hr else "Risk tier to confirm (likely limited/minimal)."))
        })
    if p.personal_data:
        extra = ""
        if p.special_categories:
            extra += " Special-category data → Art 9 conditions apply."
        if p.automated_decisions:
            extra += " Solely-automated decisions → Art 22 safeguards apply."
        out.append({
            "reg": "GDPR",
            "high_risk": p.special_categories or p.automated_decisions,
            "reason": "You process personal data of individuals." + extra,
        })
    if p.financial_entity:
        out.append({
            "reg": "DORA",
            "high_risk": False,
            "reason": "You are a financial entity → ICT risk-management & operational-resilience obligations.",
        })
    if p.critical_entity:
        out.append({
            "reg": "NIS2",
            "high_risk": False,
            "reason": "You are an essential/important entity → cybersecurity risk-management & incident-reporting duties.",
        })
    return out


def _question_for(reg: str, p: Profile) -> str:
    """A focused question per regulation that grounds the concrete obligations in the corpus."""
    if reg == "EU AI Act":
        if p.high_risk_area:
            return ("What are the main obligations for providers of high-risk AI systems "
                    "(risk management, data governance, technical documentation, transparency, "
                    "human oversight, accuracy and robustness)?")
        return "What transparency obligations apply to AI systems that interact with natural persons?"
    if reg == "GDPR":
        if p.automated_decisions:
            return ("What are the core obligations for processing personal data, and what "
                    "safeguards apply to solely automated decision-making?")
        return ("What are the core obligations of a controller processing personal data "
                "(lawful basis, data-subject rights, records of processing, security)?")
    if reg == "DORA":
        return ("What are the ICT risk-management and major-incident reporting obligations "
                "for financial entities?")
    if reg == "NIS2":
        return ("What are the cybersecurity risk-management measures and incident-reporting "
                "obligations for essential and important entities?")
    return "What obligations apply?"


def build_roadmap(p: Profile, ask: Callable[[str, str], object], lang: str = "en") -> Roadmap:
    """Assemble the roadmap.

    `ask(question, lang)` must return an engine Answer-like object exposing
    `.answer`, `.sources`, `.grounding`, `.abstained`, `.cost_usd`.
    """
    rm = Roadmap(profile=p)
    applies = applicable(p)
    applied = {a["reg"] for a in applies}
    rm.not_applicable = [r for r in ALL_REGS if r not in applied]

    for a in applies:
        reg = a["reg"]
        ans = ask(_question_for(reg, p), lang)
        rm.cost_usd += getattr(ans, "cost_usd", 0.0) or 0.0
        grounded = (not getattr(ans, "abstained", False)) and getattr(ans, "grounding", 0.0) >= 0.35
        rm.items.append(RoadmapItem(
            regulation=reg,
            applies_reason=a["reason"],
            severity="high" if a.get("high_risk") else "medium",
            obligations=(getattr(ans, "answer", "") if grounded else
                         "The agent could not ground concrete obligations from the built-in "
                         "corpus — flag this regulation for review with qualified counsel."),
            provisions=list(getattr(ans, "sources", []) or []),
            grounded=grounded,
            needs_review=not grounded,
        ))

    rm.items.sort(key=lambda i: 0 if i.severity == "high" else 1)  # high-severity first
    return rm
