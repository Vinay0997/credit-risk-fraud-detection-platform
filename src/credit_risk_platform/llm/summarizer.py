"""Financial document summarization helper with local fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class SummaryResult:
    summary: str
    method: str


def summarize_text(text: str, max_sentences: int = 4) -> SummaryResult:
    """Summarize text using an extractive local fallback.

    Production deployments can replace this with Azure OpenAI summarization while
    keeping the same return shape for audit logging.
    """
    clean_text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", clean_text)
    scored = []
    keywords = {
        "risk",
        "capital",
        "default",
        "fraud",
        "loss",
        "liquidity",
        "compliance",
        "material",
        "exposure",
    }
    for sentence in sentences:
        tokens = {token.lower().strip(".,:;()") for token in sentence.split()}
        score = len(tokens.intersection(keywords)) + min(len(sentence) / 200, 1)
        scored.append((score, sentence))
    selected = [sentence for _, sentence in sorted(scored, reverse=True)[:max_sentences]]
    selected_in_original_order = [sentence for sentence in sentences if sentence in selected]
    return SummaryResult(summary=" ".join(selected_in_original_order), method="local_extractive")
