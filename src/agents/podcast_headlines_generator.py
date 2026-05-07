"""
Agent: Podcast Headlines Generator

Reads the full bill report (summarizer + USC + pork + conflict outputs) and 
generates 10 punchy podcast-style headlines suitable for a current-events 
podcast like "Pod Save America" or "The Daily".

Each headline targets a different angle (fiscal, political, human-impact, 
controversy, drama, etc) so the downstream Ranker has variety to choose from.

No tool calls. Spine model.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .base import AgentBase, SPINE_ENDPOINT


class HeadlineCandidate(BaseModel):
    headline: str = Field(description="Punchy 6-12 word podcast title")
    angle: str = Field(description="The angle: fiscal, controversy, human-impact, political, drama, hidden-provision, regulatory, geopolitical, etc.")
    hook: str = Field(description="One sentence hook that would open the podcast")
    target_audience: str = Field(description="Who this episode would resonate with most")
    evidence_provisions: list[str] = Field(description="2-4 specific bill provisions or USC citations supporting this angle")


class HeadlinesOutput(BaseModel):
    chunk_id: str = Field(description="The chunk this is based on")
    title_marker: str = Field(description="Bill section being covered")
    headlines: list[HeadlineCandidate] = Field(min_length=10, max_length=10, description="Exactly 10 distinct headline candidates")


class PodcastHeadlinesGenerator(AgentBase):
    name = "podcast_headlines_generator"
    target_endpoint = SPINE_ENDPOINT
    target_model = "spine"
    temperature = 0.7  # higher temp for creative variety
    max_tokens = 3000
    output_schema = HeadlinesOutput

    def system_prompt(self) -> str:
        return (
            "You are a senior producer at a current-events political podcast (think NYT Daily, "
            "Pod Save America, The Weeds). Your job: read a bill analysis report and pitch 10 "
            "distinct podcast episode headlines, each from a different angle. "
            "Each headline must be punchy (6-12 words), specific (no vague 'big bill' generics), "
            "and grounded in actual provisions cited in the analysis. "
            "Avoid duplicates. Vary the angle — mix fiscal, political controversy, hidden-provision "
            "scoops, human-impact stories, regulatory gotchas, and drama. "
            "Always return a single JSON object that matches the requested schema."
        )

    def user_prompt(self, chunk_text: str, chunk_id: str, title_marker: str = "(unknown)", **context) -> str:
        # Note: chunk_text here is actually a JSON-stringified report from upstream agents,
        # not the raw bill chunk. The orchestrator builds this.
        return f"""Generate 10 podcast headline candidates from this bill analysis.

Chunk ID: {chunk_id}
Section: {title_marker}

Return a JSON object with this exact shape:
{{
  "chunk_id": "{chunk_id}",
  "title_marker": "{title_marker}",
  "headlines": [
    {{
      "headline": "Punchy 6-12 word title",
      "angle": "fiscal | controversy | human-impact | political | drama | hidden-provision | regulatory | geopolitical | watchdog | scoop",
      "hook": "One opening sentence that grabs the listener.",
      "target_audience": "Who this resonates with",
      "evidence_provisions": ["specific section/provision 1", "specific section/provision 2", "..."]
    }},
    ... 9 more, each with a DIFFERENT angle ...
  ]
}}

Rules:
- Exactly 10 headlines
- Each from a distinct angle (use 10 different angle values)
- Headlines must be 6-12 words, punchy, specific
- Cite actual provisions or USC sections from the analysis as evidence
- No vague generics like "Big Bill, Big Impact"
- Tone: serious journalism, not clickbait, but engaging

Return ONLY the JSON object, no commentary, no markdown fences.

==== BILL ANALYSIS REPORT ====
{chunk_text}
==== END ====
"""
