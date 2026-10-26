"""Branded EU Compliance Report (PDF) from an assessment result.

Renders the same score / compliance map / obligations / remediation that the
user sees in the /assess UI into a shareable, downloadable PDF. This is pure
rendering — no LLM and no re-grounding: it takes the already-computed
assessment (as returned by /assess) as data, so generating a report costs
nothing and matches exactly what was shown on screen.

Decision-support, not legal advice.
"""
from __future__ import annotations

import datetime
import html
import io
import re

# Brand palette (mirrors the /assess web UI)
_ACCENT = "#2563eb"
_INK = "#0f172a"
_MUT = "#64748b"
_LINE = "#e2e8f0"


def _band_color(band: str) -> str:
    return {"green": "#16a34a", "amber": "#d97706", "red": "#dc2626"}.get(band, "#64748b")


def _band_label(band: str) -> str:
    return {"green": "On track",
            "amber": "Needs attention",
            "red": "High-risk — act now"}.get(band, "Assessed")


def _xml(s: str) -> str:
    """Escape for reportlab Paragraph markup, but keep **bold** → <b>bold</b>."""
    s = html.escape(s or "", quote=False)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)


def build_report_pdf(data: dict) -> bytes:
    """Render an assessment result dict into PDF bytes.

    Expected keys (all optional, rendered defensively):
      description, score, band, jurisdiction, items[], not_applicable[], cost_usd
    Each item: regulation, applies_reason, severity, obligations, provisions[],
      grounded, needs_review, actions[{action, ref}], meta{full,in_force,note}.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, ListFlowable, ListItem,
                                    HRFlowable, KeepTogether)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    base = getSampleStyleSheet()
    S = {
        "title": ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=20, textColor=colors.HexColor(_INK),
                                alignment=0, spaceAfter=0, leading=23),
        "kicker": ParagraphStyle("k", parent=base["Normal"], fontName="Helvetica-Bold",
                                 fontSize=10, textColor=colors.HexColor(_ACCENT),
                                 spaceAfter=1),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontName="Helvetica",
                              fontSize=9, textColor=colors.HexColor(_MUT), spaceAfter=2),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold",
                             fontSize=13, textColor=colors.HexColor(_INK),
                             spaceBefore=14, spaceAfter=4),
        "reg": ParagraphStyle("reg", parent=base["Heading3"], fontName="Helvetica-Bold",
                              fontSize=12.5, textColor=colors.HexColor(_INK),
                              spaceBefore=2, spaceAfter=1),
        "reason": ParagraphStyle("reason", parent=base["Normal"], fontName="Helvetica-Oblique",
                                 fontSize=9.5, textColor=colors.HexColor(_MUT),
                                 spaceAfter=5, leading=13),
        "body": ParagraphStyle("body", parent=base["Normal"], fontName="Helvetica",
                               fontSize=10, textColor=colors.HexColor(_INK),
                               leading=14.5, spaceAfter=4),
        "small": ParagraphStyle("small", parent=base["Normal"], fontName="Helvetica",
                                fontSize=8.5, textColor=colors.HexColor(_MUT),
                                leading=12, spaceAfter=2),
        "li": ParagraphStyle("li", parent=base["Normal"], fontName="Helvetica",
                             fontSize=9.5, textColor=colors.HexColor(_INK),
                             leading=13, spaceAfter=1),
        "prov": ParagraphStyle("prov", parent=base["Normal"], fontName="Courier",
                               fontSize=8.5, textColor=colors.HexColor("#334155"),
                               leading=12),
        "disc": ParagraphStyle("disc", parent=base["Normal"], fontName="Helvetica",
                               fontSize=8, textColor=colors.HexColor(_MUT), leading=11),
    }

    band = (data.get("band") or "").lower()
    bc = colors.HexColor(_band_color(band))
    score = data.get("score")
    items = data.get("items") or []
    na = data.get("not_applicable") or []

    story = []

    # --- Header ---
    story.append(Paragraph("REGAGENT · GROUNDED LABS", S["kicker"]))
    story.append(Paragraph("EU Compliance Report", S["title"]))
    today = datetime.date.today().strftime("%d %B %Y")
    story.append(Paragraph("Generated " + today +
                           " · decision-support triage, not legal advice.", S["sub"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=bc,
                            spaceBefore=6, spaceAfter=10))

    # --- Assessed product ---
    desc = (data.get("description") or "").strip()
    if desc:
        story.append(Paragraph("Assessed product", S["h2"]))
        story.append(Paragraph(_xml(desc), S["body"]))

    # --- Score + band + jurisdiction ---
    score_txt = str(score) if score is not None else "–"
    score_cell = Paragraph(
        '<font size=30 color="%s"><b>%s</b></font><font size=11 color="%s"> / 100</font>'
        % (_band_color(band), score_txt, _MUT), S["body"])
    right = ('<font size=8 color="%s"><b>EU COMPLIANCE EXPOSURE</b></font><br/>'
             '<font size=15 color="%s"><b>%s</b></font>'
             % (_MUT, _band_color(band), _xml(_band_label(band))))
    juris = (data.get("jurisdiction") or "").strip()
    if juris:
        right += '<br/><font size=8.5 color="%s">%s</font>' % (_MUT, _xml(juris))
    tbl = Table([[score_cell, Paragraph(right, S["body"])]],
                colWidths=[34 * mm, None])
    tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (0, 0), 2, bc),
        ("LEFTPADDING", (0, 0), (0, 0), 10), ("RIGHTPADDING", (0, 0), (0, 0), 10),
        ("TOPPADDING", (0, 0), (0, 0), 8), ("BOTTOMPADDING", (0, 0), (0, 0), 8),
        ("LEFTPADDING", (1, 0), (1, 0), 14),
    ]))
    story.append(Spacer(1, 8))
    story.append(tbl)

    # --- Compliance map (traffic-light chips as a table row) ---
    chips, cstyle = [], [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                         ("TOPPADDING", (0, 0), (-1, -1), 4),
                         ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    ci = 0
    for it in items:
        sev = (it.get("severity") or "medium")
        col = "#dc2626" if sev == "high" else "#d97706"
        chips.append(Paragraph('<font color="white"><b>%s · %s</b></font>'
                               % (_xml(it.get("regulation", "")), "HIGH" if sev == "high" else "MED"),
                               S["small"]))
        cstyle += [("BACKGROUND", (ci, 0), (ci, 0), colors.HexColor(col)),
                   ("LEFTPADDING", (ci, 0), (ci, 0), 8), ("RIGHTPADDING", (ci, 0), (ci, 0), 8)]
        ci += 1
    for n in na:
        chips.append(Paragraph('<font color="#166534"><b>%s · N/A</b></font>' % _xml(n), S["small"]))
        cstyle += [("BACKGROUND", (ci, 0), (ci, 0), colors.HexColor("#dcfce7")),
                   ("LEFTPADDING", (ci, 0), (ci, 0), 8), ("RIGHTPADDING", (ci, 0), (ci, 0), 8)]
        ci += 1
    if chips:
        cmap = Table([chips], hAlign="LEFT")
        cmap.setStyle(TableStyle(cstyle))
        story.append(Spacer(1, 10))
        story.append(cmap)

    # --- Detail + remediation, per regulation ---
    story.append(Paragraph("What applies — and what to do", S["h2"]))
    story.append(Paragraph("%d applicable regulation(s), highest priority first."
                           % len(items), S["sub"]))

    for it in items:
        block = []
        sev = (it.get("severity") or "medium").upper()
        meta = it.get("meta") or {}
        head = '%s &nbsp;<font size=8 color="%s">%s</font>' % (
            _xml(it.get("regulation", "")), _band_color(band if band else "red"), sev)
        if it.get("needs_review"):
            head += ' <font size=8 color="%s">· NEEDS LEGAL REVIEW</font>' % _MUT
        if meta.get("in_force"):
            head += ' <font size=8 color="%s">· in force %s</font>' % (_MUT, _xml(meta["in_force"]))
        block.append(Paragraph(head, S["reg"]))
        if meta.get("full"):
            block.append(Paragraph(_xml(meta["full"]) +
                                   (" — " + _xml(meta["note"]) if meta.get("note") else ""),
                                   S["small"]))
        if it.get("applies_reason"):
            block.append(Paragraph(_xml(it["applies_reason"]), S["reason"]))
        for para in re.split(r"\n{1,}", (it.get("obligations") or "").strip()):
            if para.strip():
                block.append(Paragraph(_xml(para.strip()), S["body"]))

        actions = it.get("actions") or []
        if actions:
            block.append(Paragraph("REMEDIATION PLAN", S["small"]))
            li = []
            for a in actions:
                txt = _xml(a.get("action", ""))
                ref = a.get("ref", "")
                if ref:
                    txt += ' <font color="%s">— %s</font>' % (_ACCENT, _xml(ref))
                li.append(ListItem(Paragraph(txt, S["li"]), leftIndent=12))
            block.append(ListFlowable(li, bulletType="1", leftIndent=12,
                                      bulletFontSize=9, bulletColor=colors.HexColor(_MUT)))

        provs = it.get("provisions") or []
        if provs:
            block.append(Spacer(1, 3))
            block.append(Paragraph("Cited: " + " · ".join(_xml(p) for p in provs), S["prov"]))

        block.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor(_LINE), spaceBefore=8, spaceAfter=4))
        story.append(KeepTogether(block))

    if na:
        story.append(Paragraph("Not flagged from this profile: " + ", ".join(_xml(n) for n in na)
                               + " — revisit if your product or sector changes.", S["small"]))

    # --- Footer / disclaimer ---
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor(_LINE),
                            spaceBefore=2, spaceAfter=6))
    story.append(Paragraph(
        "Generated by RegAgent (Grounded Labs) — requirements grounded in cited articles, "
        "with honest abstention where the engine cannot ground a requirement. This is a "
        "preliminary EU compliance-exposure assessment and <b>decision-support, not legal "
        "advice</b>; confirm with qualified counsel before relying on it.", S["disc"]))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=15 * mm, bottomMargin=14 * mm,
                            title="RegAgent — EU Compliance Report",
                            author="Grounded Labs")
    doc.build(story)
    return buf.getvalue()
