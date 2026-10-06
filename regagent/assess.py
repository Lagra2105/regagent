"""Startup-facing layer: a product profile → applicable EU regulations → an
actionable compliance roadmap, with a compliance score, a remediation plan, and
jurisdiction + version awareness.

Built ON the RegAgent engine, not around it: applicability, scoring and the
remediation steps are a deterministic, auditable rules triage (no LLM), while
the concrete obligations and their article citations come *grounded* from the
corpus via the engine's answer function. Where the engine can't ground an item,
we surface an honest "flag for legal review" instead of inventing requirements.

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

ALL_REGS = ["EU AI Act", "GDPR", "DORA", "NIS2", "MiCA"]

# --- Version / date awareness: when each regulation applies (for the report) ---
REG_META: dict[str, dict] = {
    "EU AI Act": {"full": "Regulation (EU) 2024/1689",
                  "in_force": "1 Aug 2024",
                  "note": "High-risk obligations apply from 2 Aug 2026."},
    "GDPR":      {"full": "Regulation (EU) 2016/679",
                  "in_force": "25 May 2018", "note": "In force."},
    "DORA":      {"full": "Regulation (EU) 2022/2554",
                  "in_force": "17 Jan 2025", "note": "Applies to financial entities."},
    "NIS2":      {"full": "Directive (EU) 2022/2555",
                  "in_force": "Oct 2024", "note": "National transposition ongoing."},
    "MiCA":      {"full": "Regulation (EU) 2023/1114",
                  "in_force": "30 Jun 2024 (ART/EMT) · 30 Dec 2024 (CASPs)",
                  "note": "Phased application 2024–2025."},
}


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
    crypto_assets: bool = False     # issues crypto-assets or provides crypto-asset services (MiCA)
    eu_market: bool = True
    based_in_monaco: bool = False   # jurisdiction flag for the Monaco + EU framing


@dataclass
class RoadmapItem:
    regulation: str
    applies_reason: str
    severity: str            # "high" | "medium"
    obligations: str         # grounded summary from the engine (or the review note)
    provisions: list[str]    # cited articles → each becomes a checklist line
    grounded: bool
    needs_review: bool       # engine abstained / weak grounding → human/legal review
    actions: list[dict] = field(default_factory=list)   # deterministic remediation steps
    meta: dict = field(default_factory=dict)            # REG_META entry (dates/version)


@dataclass
class Roadmap:
    profile: Profile
    items: list[RoadmapItem] = field(default_factory=list)
    not_applicable: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    score: int = 100              # EU compliance-exposure score 0–100
    band: str = "green"           # "green" | "amber" | "red"
    jurisdiction: str = ""        # human-readable jurisdiction note


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
    if p.crypto_assets and p.eu_market:
        out.append({
            "reg": "MiCA",
            "high_risk": True,
            "reason": ("You issue crypto-assets or provide crypto-asset services in the EU → "
                       "MiCA authorisation, white-paper, prudential, governance and safeguarding duties."),
        })
    return out


# --- Deterministic, cited remediation steps per regulation (auditable, no LLM) ---
def remediation(reg: str, p: Profile) -> list[dict]:
    def a(action: str, ref: str) -> dict:
        return {"action": action, "ref": ref}

    if reg == "EU AI Act":
        if p.high_risk_area:
            return [
                a("Classify the AI system and confirm its risk tier", "AI Act Art. 6 + Annex III"),
                a("Establish and document a risk-management system", "AI Act Art. 9"),
                a("Put in place data governance for training/validation data", "AI Act Art. 10"),
                a("Draw up technical documentation before market placement", "AI Act Art. 11"),
                a("Ensure effective human oversight", "AI Act Art. 14"),
                a("Register the high-risk system in the EU database", "AI Act Art. 49/71"),
            ]
        return [
            a("Tell users they are interacting with an AI system", "AI Act Art. 50"),
            a("Label AI-generated or manipulated content where required", "AI Act Art. 50"),
        ]
    if reg == "GDPR":
        acts = [a("Confirm a lawful basis for processing", "GDPR Art. 6")]
        if p.automated_decisions:
            acts.append(a("Add safeguards for solely-automated decisions (human review, contest)", "GDPR Art. 22"))
        if p.special_categories:
            acts.append(a("Establish an Art. 9 condition for special-category data", "GDPR Art. 9"))
        acts += [
            a("Provide clear privacy information to data subjects", "GDPR Art. 13–14"),
            a("Maintain records of processing activities", "GDPR Art. 30"),
            a("Implement appropriate security of processing", "GDPR Art. 32"),
            a("Run a Data Protection Impact Assessment (DPIA)", "GDPR Art. 35"),
        ]
        return acts
    if reg == "DORA":
        return [
            a("Establish an ICT risk-management framework", "DORA Art. 5–6"),
            a("Set up ICT-incident management and major-incident reporting", "DORA Art. 17–19"),
            a("Run digital operational resilience testing", "DORA Art. 24"),
            a("Manage ICT third-party risk and maintain the register", "DORA Art. 28"),
        ]
    if reg == "NIS2":
        return [
            a("Adopt cybersecurity risk-management measures", "NIS2 Art. 21"),
            a("Set up incident notification to the CSIRT / authority", "NIS2 Art. 23"),
            a("Ensure management-body oversight and staff training", "NIS2 Art. 20"),
        ]
    if reg == "MiCA":
        acts = [a("Confirm your classification (crypto-asset, ART, EMT, CASP)", "MiCA Art. 3")]
        if p.crypto_assets:
            acts += [
                a("Obtain CASP authorisation (or issuer authorisation for ART/EMT)", "MiCA Art. 59 / 16 / 48"),
                a("Publish a compliant crypto-asset white paper where required", "MiCA Art. 6"),
                a("Meet prudential and governance requirements", "MiCA Art. 67–68"),
                a("Safeguard clients' crypto-assets and funds (segregation)", "MiCA Art. 70"),
                a("Apply AML/CFT and the Transfer-of-Funds 'travel rule'", "TFR (EU) 2023/1113"),
            ]
        return acts
    return []


def compliance_score(items: list[RoadmapItem]) -> int:
    """Preliminary EU compliance-exposure score (0–100). Deterministic, auditable:
    more high-severity regimes and ungrounded items = higher exposure = lower score.
    Not a certification — an exposure indicator."""
    score = 100
    for it in items:
        score -= 22 if it.severity == "high" else 10
        if it.needs_review:
            score -= 4
    return max(score, 5)


def score_band(score: int) -> str:
    return "green" if score >= 75 else ("amber" if score >= 45 else "red")


def jurisdiction_note(p: Profile) -> str:
    if p.based_in_monaco and p.eu_market:
        return ("You operate from Monaco (outside the EU/EEA) and target the EU market. EU rules "
                "(AI Act, GDPR, MiCA) apply extraterritorially to products placed on or targeting the "
                "EU; Monaco-specific supervision (e.g. CCIN for personal data) may also apply.")
    if p.based_in_monaco:
        return "You operate from Monaco. EU rules apply where you place products on or target the EU market."
    return "EU-based / EU market: EU rules apply directly."


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
    if reg == "MiCA":
        return ("What are the authorisation, white paper, prudential, governance and safeguarding "
                "obligations for crypto-asset service providers and issuers under MiCA?")
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
            actions=remediation(reg, p),
            meta=REG_META.get(reg, {}),
        ))

    rm.items.sort(key=lambda i: 0 if i.severity == "high" else 1)  # high-severity first
    rm.score = compliance_score(rm.items)
    rm.band = score_band(rm.score)
    rm.jurisdiction = jurisdiction_note(p)
    return rm
