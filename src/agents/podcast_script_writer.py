"""
Agent: Podcast Script Writer

Reads a bill report + winning headline and writes a 19-line two-host dialog
(Alex / Jordan) suitable for direct synthesis by Qwen3-TTS.

Hosts:
  - Alex (Ryan voice)  - confident, slightly skeptical, anchors the segment
  - Jordan (Ono_anna voice) - sharp, asks the smart-listener questions

19 lines = 19 scenes (alternating speakers), each ~5-12 seconds spoken.
This count maps 1:1 to the existing slide pipeline (19 slides).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .base import AgentBase, SPINE_ENDPOINT


class DialogLine(BaseModel):
    scene: int = Field(description="1-indexed scene number, 1 to 19")
    speaker: str = Field(description="Either 'Alex' or 'Jordan'")
    line: str = Field(description="Spoken dialog, 12-30 words, conversational")
    beat: str = Field(description="One-line description of what this scene establishes (cold-open, hook, fiscal-detail, controversy, takeaway, etc.)")


class PodcastScriptOutput(BaseModel):
    bill_short: str = Field(description="Short code for the bill (e.g. border25)")
    headline: str = Field(description="Winning headline this episode is built around")
    cold_open: str = Field(description="Optional 1-sentence pre-roll teaser, can be empty")
    dialog: list[DialogLine] = Field(min_length=19, max_length=19)


class PodcastScriptWriter(AgentBase):
    name = "podcast_script_writer"
    target_endpoint = SPINE_ENDPOINT
    target_model = "spine"
    temperature = 0.5
    max_tokens = 4000
    output_schema = PodcastScriptOutput

    def system_prompt(self) -> str:
        return (
            "You are head writer for a 4-minute current-events podcast modeled on NYT Daily and "
            "Pod Save America. Two hosts: Alex (anchor, slightly skeptical) and Jordan (sharp, "
            "asks smart-listener questions). They are reading from a bill analysis. "
            "Write a 19-line dialog that opens with a hook, walks through the bill\'s most newsworthy "
            "provisions, and lands a memorable closing beat. Voice: serious journalism, "
            "conversational, NEVER sensational or hyperbolic. Avoid filler like \'so basically\'. "
            "Each line 12-30 words; alternates speakers; line 1 = Alex, line 2 = Jordan, etc. "
            "Always output JSON matching the requested schema."
        )

    def user_prompt(self, chunk_text: str, chunk_id: str,
                    bill_short: str = "(unknown)",
                    headline: str = "(unknown)",
                    **context) -> str:
        return f"""Write a 19-line podcast dialog about this bill.

Bill: {bill_short}
Winning headline: {headline}

Return JSON:
{{
  "bill_short": "{bill_short}",
  "headline": "{headline}",
  "cold_open": "Optional one-sentence teaser before the dialog.",
  "dialog": [
    {{"scene": 1, "speaker": "Alex",   "line": "...", "beat": "cold-open hook"}},
    {{"scene": 2, "speaker": "Jordan", "line": "...", "beat": "..."}},
    ... 17 more ...
  ]
}}

Rules:
- EXACTLY 19 dialog entries.
- Speakers alternate strictly: odd scenes = Alex, even scenes = Jordan.
- Each line 12-30 words.
- Cite specific provisions, dollar amounts, USC sections from the report.
- Arc: hook (1-3) -> setup (4-7) -> meat / controversy (8-14) -> stakes (15-17) -> close (18-19).
- No cold-open inside dialog; use the cold_open field if you want one.
- Tone: serious, intelligent, dry humor allowed.

Return ONLY the JSON object.

==== BILL ANALYSIS ====
{chunk_text}
==== END ====
"""
