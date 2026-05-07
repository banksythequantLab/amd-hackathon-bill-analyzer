"""
Agent: Slide Prompt Generator

Reads the 19-line podcast script and generates 19 Qwen-Image generation prompts,
one per scene. Output prompts are tuned for the editorial-podcast-slide visual
style: bold headline on dark background, Capitol/policy iconography, clean
typography, 16:9.

Style locked to match BBB slide that user approved.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .base import AgentBase, SPINE_ENDPOINT


SLIDE_STYLE_DNA = (
    "editorial podcast slide, bold white sans-serif headline (1-6 words), "
    "deep navy or near-black background, single relevant policy icon or silhouette "
    "(US Capitol dome, scales of justice, dollar sign, border fence, gavel, etc.) "
    "subtly placed, thin red or gold accent line, clean modern typography, "
    "high contrast, professional news graphic, 16:9 aspect"
)

NEG_STYLE_DNA = (
    "blurry, low quality, watermark, signature, distorted text, illegible, "
    "photorealistic faces, cluttered layout, multiple competing elements, "
    "comic style, anime, low-res text"
)


class SlidePrompt(BaseModel):
    scene: int = Field(description="1-19")
    headline_text: str = Field(description="The 1-6 word headline that should appear in the slide image")
    visual_concept: str = Field(description="One sentence describing what the slide visualizes")
    positive_prompt: str = Field(description="Full Qwen-Image positive prompt, ready to send")
    negative_prompt: str = Field(description="Negative prompt, may reuse the default")


class SlidePromptOutput(BaseModel):
    bill_short: str
    style_notes: str = Field(description="Any global style notes used across all 19 slides")
    slides: list[SlidePrompt] = Field(min_length=19, max_length=19)


class SlidePromptGenerator(AgentBase):
    name = "slide_prompt_generator"
    target_endpoint = SPINE_ENDPOINT
    target_model = "spine"
    temperature = 0.4
    max_tokens = 4500
    output_schema = SlidePromptOutput

    def system_prompt(self) -> str:
        return (
            "You write image-generation prompts for editorial podcast slides. "
            "Each slide pairs ONE 1-6 word headline with ONE clean policy graphic on a dark "
            "background. The visual style is locked: think NYT Opinion, Axios, Pod Save America "
            "promo cards. NEVER write prompts that include faces, photorealistic people, "
            "logos of real companies, or clutter. Each prompt must explicitly include the "
            "exact headline text in quotes so the image generator renders it. "
            "Always output JSON matching the schema."
        )

    def user_prompt(self, chunk_text: str, chunk_id: str,
                    bill_short: str = "(unknown)",
                    **context) -> str:
        return f"""Generate 19 Qwen-Image slide prompts, one per dialog line.

Bill: {bill_short}
Style DNA: {SLIDE_STYLE_DNA}
Default negative: {NEG_STYLE_DNA}

Return JSON:
{{
  "bill_short": "{bill_short}",
  "style_notes": "Any global style direction you applied",
  "slides": [
    {{
      "scene": 1,
      "headline_text": "Bold short headline",
      "visual_concept": "One sentence what this slide shows",
      "positive_prompt": "Full prompt with the exact headline text in quotes...",
      "negative_prompt": "{NEG_STYLE_DNA}"
    }},
    ... 18 more ...
  ]
}}

Rules:
- EXACTLY 19 slides matching the 19 dialog lines.
- headline_text must be 1-6 words, all-caps friendly, distilled from that scene\'s dialog beat.
- Each positive_prompt MUST start with the locked Style DNA, then add the scene-specific concept,
  then explicitly include `bold white headline "EXACT HEADLINE HERE"` so the model renders the text.
- Vary the icon/silhouette across the 19 slides, but stay in the Capitol/legal/finance/policy family.
- Scene 1 = the cold-open / title card. Scene 19 = the closing beat.
- Do NOT include people\'s faces, real company logos, or photographic detail.

Return ONLY the JSON.

==== PODCAST DIALOG ====
{chunk_text}
==== END ====
"""
