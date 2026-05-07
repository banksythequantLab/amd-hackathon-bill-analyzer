"""
Agent: Headline Ranker

Receives the 10 headline candidates from PodcastHeadlinesGenerator and ranks
them on three axes:
  - Newsworthiness (how surprising/important is the angle)
  - Specificity (does it cite real provisions, or is it vague?)
  - Listener appeal (would someone click play?)

The top-ranked headline becomes the seed for the downstream podcast script
generator (existing podcast_generator.py + Day 7 video pipeline).

No tool calls. Reasoner model preferred for nuanced judgment, falls back to spine.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .base import AgentBase, SPINE_ENDPOINT


class HeadlineScore(BaseModel):
    rank: int = Field(description="Final rank (1 = best, 10 = worst)")
    headline: str = Field(description="The original headline text")
    angle: str = Field(description="The original angle")
    newsworthiness_score: int = Field(ge=1, le=10, description="1-10: how surprising / important / scoop-worthy")
    specificity_score: int = Field(ge=1, le=10, description="1-10: cites concrete provisions vs vague")
    appeal_score: int = Field(ge=1, le=10, description="1-10: would a typical listener click play?")
    composite_score: int = Field(ge=3, le=30, description="Sum of the three scores, used for final ranking")
    rationale: str = Field(description="2-3 sentence justification for the rank")


class RankerOutput(BaseModel):
    chunk_id: str
    title_marker: str
    rankings: list[HeadlineScore] = Field(min_length=10, max_length=10, description="All 10 ranked, sorted by rank ascending")
    winner: HeadlineScore = Field(description="The #1 ranked headline (also rankings[0])")
    winner_explanation: str = Field(description="Why this is the strongest podcast hook of the 10")


class HeadlineRanker(AgentBase):
    name = "headline_ranker"
    target_endpoint = SPINE_ENDPOINT  # spine is fine for ranking; reasoner is offline
    target_model = "spine"
    temperature = 0.0  # deterministic ranking
    max_tokens = 4000
    output_schema = RankerOutput

    def system_prompt(self) -> str:
        return (
            "You are a senior editor at a current-events political podcast network. "
            "You receive 10 podcast headline candidates and rank them for production. "
            "Your judgment criteria: "
            "(1) NEWSWORTHINESS — does this angle surface something the average listener wouldn't already know? "
            "Hidden provisions, surprising cross-references, real conflicts > generic 'bill spends a lot of money'. "
            "(2) SPECIFICITY — does the headline name a concrete provision, dollar amount, USC section, or affected group? "
            "Generic adjectives ('massive', 'controversial') are weaker than specific facts. "
            "(3) LISTENER APPEAL — would a typical politically-engaged listener click play? "
            "Drama, human stakes, contrarian angles win. Wonky regulatory minutiae lose unless tied to real impact. "
            "Score each axis 1-10. Sum = composite. Sort by composite. Ties broken by newsworthiness. "
            "Return ALL 10 ranked, plus an explicit winner field. "
            "Always return a single JSON object that matches the requested schema."
        )

    def user_prompt(self, chunk_text: str, chunk_id: str, title_marker: str = "(unknown)", **context) -> str:
        # chunk_text here is the JSON-stringified output from PodcastHeadlinesGenerator
        return f"""Rank these 10 podcast headline candidates from a bill analysis.

Chunk ID: {chunk_id}
Section: {title_marker}

Return a JSON object with this exact shape:
{{
  "chunk_id": "{chunk_id}",
  "title_marker": "{title_marker}",
  "rankings": [
    {{
      "rank": 1,
      "headline": "...",
      "angle": "...",
      "newsworthiness_score": 9,
      "specificity_score": 8,
      "appeal_score": 9,
      "composite_score": 26,
      "rationale": "Two-three sentence justification for this rank."
    }},
    ... continue through rank 10 ...
  ],
  "winner": {{ ...same shape as rankings[0]... }},
  "winner_explanation": "Why this is the strongest podcast hook of the 10."
}}

Rules:
- Score every headline on all three axes (1-10 each)
- composite_score = newsworthiness + specificity + appeal
- Sort rankings by composite_score DESCENDING (rank 1 = highest composite)
- Break ties by higher newsworthiness_score
- All 10 candidates must appear in rankings
- winner field MUST be a copy of rankings[0]
- Return ONLY the JSON object, no commentary, no markdown fences

==== HEADLINE CANDIDATES (JSON) ====
{chunk_text}
==== END ====
"""
