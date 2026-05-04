"""
Agent #2: Plain-English Summarizer

Reads a bill chunk and produces a 5-bullet plain-English summary.
No tool calls. Spine model.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .base import AgentBase, SPINE_ENDPOINT


class SummaryOutput(BaseModel):
    chunk_id: str = Field(description="The chunk identifier")
    title_marker: str = Field(description="The TITLE / Subtitle / DIVISION marker for this chunk")
    one_sentence_summary: str = Field(description="What this chunk does, in one plain-English sentence")
    bullets: list[str] = Field(description="3-30 plain-English bullets covering the chunk", min_length=3, max_length=40)
    affected_groups: list[str] = Field(default_factory=list, description="Which groups of people / entities this affects")


class PlainEnglishSummarizer(AgentBase):
    name = "plain_english_summarizer"
    target_endpoint = SPINE_ENDPOINT
    target_model = "spine"
    temperature = 0.0
    max_tokens = 1500
    output_schema = SummaryOutput

    def system_prompt(self) -> str:
        return (
            "You are a legislative analyst who explains complex bills in plain English. "
            "You receive a chunk of a bill (typically a Title or Subtitle) and produce a "
            "structured summary. You write for an informed-but-non-expert reader. "
            "Avoid legalese. Avoid hedging. State what the chunk *does*, not what it *says*. "
            "Aim for 5-8 high-signal bullets. Be concise — one bullet per major theme. "
            "Always return a single JSON object that matches the requested schema."
        )

    def user_prompt(self, chunk_text: str, chunk_id: str, title_marker: str = "(unknown)", **context) -> str:
        return f"""Summarize this chunk of a US legislative bill.

Chunk ID: {chunk_id}
Structural marker: {title_marker}

Return a JSON object with this exact shape:
{{
  "chunk_id": "{chunk_id}",
  "title_marker": "{title_marker}",
  "one_sentence_summary": "What this chunk does in one plain-English sentence.",
  "bullets": [
    "Bullet 1 (cover one major theme)",
    "Bullet 2",
    "Bullet 3",
    "Bullet 4",
    "Bullet 5"
  ],
  "affected_groups": ["e.g. small farms", "e.g. SNAP recipients"]
}}

Aim for 5-8 bullets total. Each bullet should cover a distinct major provision.
Do NOT enumerate every individual section — group related ones into themes.

Return ONLY the JSON object, no commentary, no markdown fences.

==== BILL TEXT ====
{chunk_text}
==== END ====
"""
