"""
Agent #8: Podcast Generator

Reads a bill report (the structured output of agents 1-7 for one chunk)
and produces a two-host podcast dialogue script. Each line of dialogue
includes a shot_prompt field that downstream video generation (ComfyUI
on AMD ROCm in Phase C) will use to render an accompanying visual.

Hosts:
  - Alex: the explainer. Plain-spoken. Asks the questions a smart but
    non-expert listener would ask. Translates legalese into normal English.
  - Jordan: the analyst. Brings structure and stakes. Reads from the
    actual numbers and findings the agents produced. Connects the bill's
    text to "what does this mean for real people."

Output is a list of dialogue lines, each with:
  - speaker: "Alex" | "Jordan"
  - text: what they say (1-3 sentences, plain English, no legalese)
  - shot_prompt: visual prompt for the video generator. Concrete imagery,
    not abstract concepts.
  - est_seconds: rough TTS duration estimate (~2.5 words/sec rule of thumb)

The script targets a 5-8 minute podcast (roughly 30-50 dialogue lines).
The agent itself writes the script; TTS rendering and video stitching
are separate downstream concerns.

Runs on spine endpoint - the chunk text + agent outputs are usually
already APC-warm from the report run, and 250K+ context fits trivially.
"""
from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

from .base import AgentBase, SPINE_ENDPOINT


VALID_SPEAKERS = {"Alex", "Jordan"}


class DialogueLine(BaseModel):
    speaker: str = Field(description="Alex or Jordan")
    text: str = Field(description="What they say. 1-3 plain-English sentences.")
    shot_prompt: str = Field(description="Concrete visual prompt for the accompanying shot, <=30 words")
    est_seconds: float = Field(default=0.0, description="Estimated TTS duration; 0 means caller computes")

    @field_validator("speaker")
    @classmethod
    def _valid_speaker(cls, v: str) -> str:
        if v not in VALID_SPEAKERS:
            raise ValueError(f"speaker must be Alex or Jordan, got {v!r}")
        return v


class PodcastScript(BaseModel):
    bill: str = Field(description="Bill identifier, e.g. 'bbb', 'hr1'")
    chunk_id: str
    title: str = Field(description="Episode title; <=12 words")
    hook: str = Field(description="Opening 1-sentence hook that doesn't presume the listener knows the bill")
    lines: list[DialogueLine] = Field(default_factory=list)
    closer: str = Field(description="One-sentence sign-off pointing at where listeners can find the report")
    note: Optional[str] = Field(default=None)
    model_config = {"extra": "allow"}


class PodcastGenerator(AgentBase):
    """Two-host podcast script generator.

    Input: a structured report dict (the kind compute_totals produces for
    fiscal, or the canonical multi-agent report). The agent reads it,
    writes a script in plain English, and tags each line with a shot
    prompt for the future video generator.
    """
    name = "podcast_generator"
    target_endpoint = SPINE_ENDPOINT
    target_model = "spine"
    temperature = 0.4   # higher than the analytical agents - this is creative writing
    max_tokens = 6000   # ~30-50 dialogue lines * ~80 tokens each
    output_schema = PodcastScript

    def system_prompt(self) -> str:
        return (
            "You are a podcast scriptwriter. You take a structured report about a US "
            "legislative bill and produce a two-host conversation that explains what "
            "the bill does, what the numbers say, and what it means for real people. "
            "You write in plain English. No legalese. No 'this section appropriates' "
            "language - you say 'this would put $10 billion into wildfire prevention'.\n\n"
            "TWO HOSTS:\n"
            "  Alex: the explainer. Curious, asks the questions a smart non-expert "
            "  would ask. Plays the audience surrogate. Slightly skeptical but charitable.\n"
            "  Jordan: the analyst. Brings the structure. References specific numbers "
            "  from the report. Connects the text to outcomes. Doesn't lecture; "
            "  responds to Alex's questions.\n\n"
            "FORMAT:\n"
            "- Open with a hook from Alex (1 sentence) that doesn't presume any prior knowledge\n"
            "- 30-50 alternating dialogue lines, each 1-3 sentences\n"
            "- Each line has a shot_prompt for visual generation. Make these CONCRETE: "
            "  'aerial shot of a forest on a sunny day' not 'forest scene'. "
            "  Avoid abstract metaphors that won't render well. No text overlays in the prompt.\n"
            "- Close with Jordan pointing at where listeners can find the full report\n\n"
            "TONE:\n"
            "- Warm but not breathless. The bill is interesting on its own; you don't need to oversell.\n"
            "- Honest about uncertainty. If the report flags something as wrong-section, say so plainly.\n"
            "- Specific over general. 'The Forest Service gets $10B' beats 'significant funding for forestry.'\n"
            "- No partisan framing. The bill is what it is; you're explaining it.\n\n"
            "WHAT TO INCLUDE FROM THE REPORT:\n"
            "- The plain-English summary's key bullets\n"
            "- Top 3-5 dollar amounts from the fiscal data (if present)\n"
            "- The most interesting USC cross-references (especially any wrong-section flags)\n"
            "- The pork findings (if any) - but acknowledge when the answer is 'no flags'\n"
            "- The conflict findings (if any)\n"
            "- The stakeholder map's biggest entities (if present)\n\n"
            "WHAT NOT TO DO:\n"
            "- Do not invent facts not in the report\n"
            "- Do not put text overlays in shot prompts\n"
            "- Do not break the 5-8 minute target (30-50 lines is the right range)\n\n"
            "Return a single JSON object matching the schema."
        )

    def user_prompt(self, chunk_text: str, chunk_id: str, title_marker: str = "(unknown)", **context) -> str:
        """chunk_text here is the JSON-serialized REPORT, not bill text."""
        bill = context.get("bill", "unknown")
        return f"""Write a 5-8 minute podcast script from this bill report.

Bill: {bill}
Chunk: {chunk_id}
Bill section: {title_marker}

Return a JSON object with this exact shape:
{{
  "bill": "{bill}",
  "chunk_id": "{chunk_id}",
  "title": "Episode title here (<=12 words)",
  "hook": "Single-sentence opener that doesn't presume prior knowledge.",
  "lines": [
    {{
      "speaker": "Alex",
      "text": "Did you know there's a bill that would put $10 billion into preventing wildfires?",
      "shot_prompt": "aerial shot of a forest on a sunny day, dense evergreens, no people",
      "est_seconds": 0
    }},
    {{
      "speaker": "Jordan",
      "text": "Yeah, and that's just one piece of TITLE I of the Build Back Better Act. There's a lot more in there.",
      "shot_prompt": "extreme close-up of a hardback US legislative bill with the title page visible",
      "est_seconds": 0
    }}
  ],
  "closer": "Read the full agent-by-agent breakdown at bills.nota.lawyer.",
  "note": null
}}

Reminders:
- speaker MUST be "Alex" or "Jordan"
- 30-50 dialogue lines total (alternating, doesn't have to be strict)
- Each line text: 1-3 sentences plain English, no legalese
- shot_prompt: <=30 words, concrete imagery, no text overlays
- est_seconds: leave as 0; the caller fills this in from word count
- Do not invent numbers or findings not in the report
- Return ONLY the JSON object. No commentary, no markdown fences.

==== REPORT JSON ====
{chunk_text}
==== END ====
"""