"""Ingestion: turn regulation text into searchable, citeable chunks.

Phase 1 ships a small built-in sample of the EU AI Act so the pipeline runs
out of the box. To use the full Act, drop its text in data/ai_act.txt with
"## Article N" headers and call load_corpus("data/ai_act.txt").
"""
from __future__ import annotations

import os
import re

from .store import Chunk

# Real EU AI Act (Regulation 2024/1689) provisions, abridged for the demo corpus.
# Covers the most queried topics: prohibited practices, high-risk classification
# and obligations, transparency, general-purpose AI, governance and penalties.
SAMPLE = [
    ("AI Act — Article 3 (Definitions)",
     "'AI system' means a machine-based system designed to operate with varying "
     "levels of autonomy, that may exhibit adaptiveness after deployment and that, "
     "for explicit or implicit objectives, infers from the input it receives how "
     "to generate outputs such as predictions, content, recommendations, or "
     "decisions that can influence physical or virtual environments."),
    ("AI Act — Article 5(1)(a)",
     "The following AI practices are prohibited: the placing on the market or use "
     "of an AI system that deploys subliminal techniques beyond a person's "
     "consciousness, or purposefully manipulative or deceptive techniques, with "
     "the objective or effect of materially distorting behaviour and causing harm."),
    ("AI Act — Article 5(1)(b)",
     "Prohibited: AI systems that exploit vulnerabilities of a person or group due "
     "to their age, disability or a specific social or economic situation, with the "
     "objective or effect of materially distorting their behaviour in a harmful way."),
    ("AI Act — Article 5(1)(c)",
     "Prohibited: AI systems for social scoring — evaluating or classifying natural "
     "persons over time based on their social behaviour or personal characteristics, "
     "leading to detrimental or unfavourable treatment in unrelated contexts."),
    ("AI Act — Article 5(1)(h)",
     "Prohibited: the use of real-time remote biometric identification systems in "
     "publicly accessible spaces for law enforcement, save for narrowly defined and "
     "authorised exceptions such as searching for victims or preventing imminent threats."),
    ("AI Act — Article 6 (High-risk classification)",
     "An AI system is high-risk if it is intended as a safety component of a product "
     "covered by Union harmonisation legislation requiring third-party conformity "
     "assessment, or if it falls within the use cases listed in Annex III, such as "
     "biometrics, critical infrastructure, education, employment, and access to "
     "essential services."),
    ("AI Act — Article 9 (Risk management)",
     "Providers of high-risk AI systems shall establish, implement, document and "
     "maintain a risk management system as a continuous iterative process run "
     "throughout the entire lifecycle, requiring regular systematic review and the "
     "adoption of measures to address identified risks."),
    ("AI Act — Article 10 (Data governance)",
     "High-risk AI systems using data to train models shall be developed on the "
     "basis of training, validation and testing data sets meeting quality criteria, "
     "subject to data governance practices including examination for biases likely "
     "to affect health, safety or fundamental rights."),
    ("AI Act — Article 11 (Technical documentation)",
     "The technical documentation of a high-risk AI system shall be drawn up before "
     "it is placed on the market and kept up to date, demonstrating compliance and "
     "providing authorities the information needed to assess conformity."),
    ("AI Act — Article 12 (Record-keeping)",
     "High-risk AI systems shall technically allow for the automatic recording of "
     "events (logs) over the lifetime of the system, to ensure a level of "
     "traceability appropriate to the intended purpose."),
    ("AI Act — Article 13 (Transparency to deployers)",
     "High-risk AI systems shall be designed to ensure their operation is "
     "sufficiently transparent to enable deployers to interpret the output and use "
     "it appropriately, accompanied by clear and complete instructions for use."),
    ("AI Act — Article 14 (Human oversight)",
     "High-risk AI systems shall be designed so they can be effectively overseen by "
     "natural persons during use, including the ability to intervene, override, or "
     "stop the system, to prevent or minimise risks to health, safety or rights."),
    ("AI Act — Article 15 (Accuracy and robustness)",
     "High-risk AI systems shall be designed to achieve an appropriate level of "
     "accuracy, robustness and cybersecurity, and to perform consistently in those "
     "respects throughout their lifecycle, resilient to errors and adversarial attacks."),
    ("AI Act — Article 16 (Provider obligations)",
     "Providers of high-risk AI systems shall ensure compliance, draw up technical "
     "documentation, implement a quality management system, keep logs, undergo "
     "conformity assessment, draw up an EU declaration of conformity and affix the "
     "CE marking before placing the system on the market."),
    ("AI Act — Article 26 (Deployer obligations)",
     "Deployers of high-risk AI systems shall use them in accordance with the "
     "instructions, assign human oversight to competent persons, monitor operation, "
     "and inform the provider or authorities of risks or serious incidents."),
    ("AI Act — Article 50 (Transparency obligations)",
     "Providers shall ensure AI systems intended to interact with natural persons "
     "inform them they are interacting with an AI system, unless obvious. Providers "
     "of generative AI shall mark outputs as artificially generated, and deployers "
     "of deep fakes shall disclose that the content has been artificially generated."),
    ("AI Act — Article 53 (General-purpose AI obligations)",
     "Providers of general-purpose AI models shall draw up and keep up to date "
     "technical documentation, provide information to downstream providers, put in "
     "place a policy to comply with Union copyright law, and publish a sufficiently "
     "detailed summary of the content used for training."),
    ("AI Act — Article 55 (GPAI with systemic risk)",
     "Providers of general-purpose AI models with systemic risk shall perform model "
     "evaluation including adversarial testing, assess and mitigate systemic risks, "
     "track and report serious incidents, and ensure an adequate level of "
     "cybersecurity protection."),
    ("AI Act — Article 72 (Post-market monitoring)",
     "Providers shall establish and document a post-market monitoring system that "
     "actively and systematically collects, documents and analyses data on the "
     "performance of high-risk AI systems throughout their lifetime."),
    ("AI Act — Article 99 (Penalties)",
     "Non-compliance with the prohibited practices in Article 5 is subject to "
     "administrative fines of up to 35 000 000 EUR or up to 7 % of total worldwide "
     "annual turnover, whichever is higher. Other infringements may incur fines up "
     "to 15 000 000 EUR or 3 % of turnover."),
    ("AI Act — Article 2 (Scope)",
     "This Regulation applies to providers placing on the market or putting into "
     "service AI systems in the Union irrespective of where they are established, "
     "and to deployers located in the Union. It does not apply to systems used "
     "exclusively for military, defence or national security purposes, to scientific "
     "research and development, or to purely personal non-professional activity."),
    ("AI Act — Article 4 (AI literacy)",
     "Providers and deployers of AI systems shall take measures to ensure, to their "
     "best extent, a sufficient level of AI literacy of their staff and other persons "
     "dealing with the operation and use of AI systems on their behalf, taking into "
     "account their technical knowledge, experience, education and the context of use."),
    ("AI Act — Article 17 (Quality management system)",
     "Providers of high-risk AI systems shall put in place a quality management "
     "system that ensures compliance with this Regulation, documented in written "
     "policies, procedures and instructions, covering a strategy for regulatory "
     "compliance, design and testing procedures, data management, and the "
     "post-market monitoring system."),
    ("AI Act — Article 18 (Documentation keeping)",
     "Providers shall, for a period ending 10 years after the high-risk AI system "
     "has been placed on the market or put into service, keep at the disposal of the "
     "national competent authorities the technical documentation, the quality "
     "management system documentation, and the EU declaration of conformity."),
    ("AI Act — Article 19 (Automatically generated logs)",
     "Providers of high-risk AI systems shall keep the logs automatically generated "
     "by their systems, to the extent such logs are under their control, for a period "
     "appropriate to the intended purpose, of at least six months unless provided "
     "otherwise by applicable Union or national law."),
    ("AI Act — Article 20 (Corrective actions and duty of information)",
     "Providers who consider or have reason to consider that a high-risk AI system "
     "they have placed on the market is not in conformity shall immediately take the "
     "necessary corrective actions to bring it into conformity, withdraw it, disable "
     "it or recall it, and inform distributors, deployers and the relevant authorities."),
    ("AI Act — Article 23 (Obligations of importers)",
     "Before placing a high-risk AI system on the market, importers shall verify that "
     "the conformity assessment procedure has been carried out, that the technical "
     "documentation has been drawn up, that the system bears the CE marking and is "
     "accompanied by the EU declaration of conformity and instructions for use."),
    ("AI Act — Article 24 (Obligations of distributors)",
     "Before making a high-risk AI system available, distributors shall verify that "
     "it bears the CE marking, is accompanied by the EU declaration of conformity and "
     "instructions for use, and that the provider and importer have complied with "
     "their obligations; they shall not make it available if it is not in conformity."),
    ("AI Act — Article 25 (Responsibilities along the AI value chain)",
     "A distributor, importer, deployer or other third party shall be considered a "
     "provider of a high-risk AI system, and assume the provider's obligations, where "
     "they put their name or trademark on it, make a substantial modification, or "
     "modify the intended purpose of a system in a way that makes it high-risk."),
    ("AI Act — Article 27 (Fundamental rights impact assessment)",
     "Prior to deploying certain high-risk AI systems, deployers that are bodies "
     "governed by public law or private operators providing public services shall "
     "perform an assessment of the impact on fundamental rights that the use may "
     "produce, describing the processes, the period and frequency of use, the "
     "categories of persons affected, and the human oversight measures."),
    ("AI Act — Article 43 (Conformity assessment)",
     "High-risk AI systems shall undergo the relevant conformity assessment "
     "procedure before being placed on the market — either based on internal control "
     "(Annex VI) or involving a notified body (Annex VII) — and a new assessment is "
     "required whenever the system is substantially modified."),
    ("AI Act — Article 47 (EU declaration of conformity)",
     "The provider shall draw up a written, machine-readable EU declaration of "
     "conformity for each high-risk AI system, keep it up to date, state that the "
     "system meets the requirements of this Regulation, and keep it at the disposal "
     "of the national competent authorities for 10 years."),
    ("AI Act — Article 48 (CE marking)",
     "High-risk AI systems shall bear the CE marking to indicate their conformity "
     "with this Regulation. For digital systems, a digital CE marking shall be used "
     "where it can be easily accessed. The CE marking is affixed visibly, legibly "
     "and indelibly before the system is placed on the market."),
    ("AI Act — Article 49 (Registration)",
     "Before placing on the market or putting into service a high-risk AI system "
     "listed in Annex III, the provider and, where applicable, the deployer shall "
     "register themselves and their system in the EU database referred to in "
     "Article 71."),
    ("AI Act — Article 73 (Reporting of serious incidents)",
     "Providers of high-risk AI systems placed on the Union market shall report any "
     "serious incident to the market surveillance authorities of the Member States "
     "where it occurred, immediately after establishing a causal link, and in any "
     "event no later than 15 days after becoming aware of it."),
    ("AI Act — Article 113 (Application dates)",
     "This Regulation entered into force on 1 August 2024 and applies from 2 August "
     "2026, with staggered dates: the prohibitions in Article 5 and AI literacy "
     "obligations apply from 2 February 2025; obligations for general-purpose AI "
     "models from 2 August 2025; and certain high-risk obligations from 2 August 2027."),
]


def _chunk_text(text: str, source: str, max_chars: int = 800) -> list[Chunk]:
    parts, buf = [], ""
    for sent in re.split(r"(?<=[.])\s+", text):
        if len(buf) + len(sent) > max_chars and buf:
            parts.append(buf.strip())
            buf = ""
        buf += " " + sent
    if buf.strip():
        parts.append(buf.strip())
    return [Chunk(id=f"{source}#{i}", text=p, source=source) for i, p in enumerate(parts)]


def load_sample() -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, text in SAMPLE:
        chunks.extend(_chunk_text(text, source))
    return chunks


def load_corpus(path: str) -> list[Chunk]:
    """Load a full regulation file split on '## Article N' headers."""
    if not os.path.exists(path):
        return load_sample()
    raw = open(path, encoding="utf-8").read()
    chunks: list[Chunk] = []
    for block in re.split(r"\n(?=##\s)", raw):
        m = re.match(r"##\s*(.+)", block)
        source = ("AI Act — " + m.group(1).strip()) if m else "AI Act"
        body = block[block.find("\n"):].strip() if "\n" in block else block
        chunks.extend(_chunk_text(body, source))
    return chunks
