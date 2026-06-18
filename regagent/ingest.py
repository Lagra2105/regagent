"""Ingestion: turn regulation text into searchable, citeable chunks.

Phase 1 ships a small built-in sample of the EU AI Act so the pipeline runs
out of the box. To use the full Act, drop its text in data/ai_act.txt with
"## Article N" headers and call load_corpus("data/ai_act.txt").
"""
from __future__ import annotations

import os
import re

from .store import Chunk

# A few real EU AI Act provisions, paraphrased/abridged for the demo corpus.
SAMPLE = [
    ("AI Act — Article 5(1)(a)",
     "The following AI practices are prohibited: the placing on the market or use "
     "of an AI system that deploys subliminal techniques beyond a person's "
     "consciousness in order to materially distort their behaviour in a manner "
     "that causes or is likely to cause physical or psychological harm."),
    ("AI Act — Article 5(1)(c)",
     "Prohibited: AI systems used for social scoring — evaluating or classifying "
     "natural persons over time based on their social behaviour or personal "
     "characteristics, leading to detrimental or unfavourable treatment."),
    ("AI Act — Article 6",
     "An AI system is classified as high-risk if it is intended to be used as a "
     "safety component of a product, or is itself a product covered by Union "
     "harmonisation legislation, and is required to undergo a third-party "
     "conformity assessment."),
    ("AI Act — Article 9",
     "Providers of high-risk AI systems shall establish, implement, document and "
     "maintain a risk management system running throughout the entire lifecycle "
     "of the high-risk AI system, requiring regular systematic review."),
    ("AI Act — Article 10",
     "High-risk AI systems which make use of techniques involving the training of "
     "models with data shall be developed on the basis of training, validation "
     "and testing data sets that meet quality criteria, examined for biases."),
    ("AI Act — Article 13",
     "High-risk AI systems shall be designed and developed to ensure their "
     "operation is sufficiently transparent to enable deployers to interpret a "
     "system's output and use it appropriately; accompanied by instructions for use."),
    ("AI Act — Article 52",
     "Providers shall ensure that AI systems intended to interact with natural "
     "persons are designed so that persons are informed they are interacting with "
     "an AI system, unless this is obvious from the circumstances."),
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
