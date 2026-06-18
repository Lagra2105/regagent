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
