"""
Agent: Wan Motion Prompt Generator

Reads the 19 slide prompts and generates the matching motion descriptions for
Wan 2.2 i2v animation. Each scene gets TWO motion prompts (pre-anim + post-anim,
~5 seconds each, total 10 seconds) so the slide breathes through the dialog.

Motion is constrained: editorial slide animation, NOT cinematic camera moves.
Think subtle parallax, slow text shimmer, gentle icon drift, light ray sweep.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .base import AgentBase, SPINE_ENDPOINT


MOTION_PALETTE = (
    "very subtle camera push-in, gentle parallax, slow zoom (5%), faint light ray sweep, "
    "drifting particle dust, soft text shimmer, slight icon rotation under 5 degrees, "
    "gentle vignette pulse - NEVER fast cuts, character motion, scene change, or camera flips"
)


class WanMotion(BaseModel):
    scene: int
    pre_motion_prompt: str = Field(description="Wan i2v prompt for pre-anim 5s clip (intro half)")
    post_motion_prompt: str = Field(description="Wan i2v prompt for post-anim 5s clip (closing half)")
    motion_intensity: str = Field(description="One of: low, medium, high (rare)")


class WanMotionOutput(BaseModel):
    bill_short: str
    motions: list[WanMotion] = Field(min_length=19, max_length=19)


class WanMotionPromptGenerator(AgentBase):
    name = "wan_motion_prompt_generator"
    target_endpoint = SPINE_ENDPOINT
    target_model = "spine"
    temperature = 0.3
    max_tokens = 4000
    output_schema = WanMotionOutput

    def system_prompt(self) -> str:
        return (
            "You write motion prompts for Wan 2.2 image-to-video animation of static editorial "
            "podcast slides. The slides are NOT photographs - they are headline cards. "
            "Motion must be SUBTLE: parallax, drift, zoom, light sweep, particle dust, text shimmer. "
            "NEVER prompt for character motion, scene changes, fast cuts, or camera flips - "
            "those break the slide illusion. Each scene gets two prompts (pre + post) that should "
            "feel continuous when concatenated. Always output JSON matching the schema."
        )

    def user_prompt(self, chunk_text: str, chunk_id: str,
                    bill_short: str = "(unknown)",
                    **context) -> str:
        return f"""Write Wan 2.2 motion prompts for 19 editorial podcast slides.

Bill: {bill_short}
Motion palette: {MOTION_PALETTE}

Return JSON:
{{
  "bill_short": "{bill_short}",
  "motions": [
    {{
      "scene": 1,
      "pre_motion_prompt": "subtle camera push-in on the headline, gentle parallax of the Capitol silhouette, slow particle dust drift, cinematic atmosphere",
      "post_motion_prompt": "soft light ray sweep across the headline from left to right, faint vignette pulse, lingering on the headline",
      "motion_intensity": "low"
    }},
    ... 18 more ...
  ]
}}

Rules:
- EXACTLY 19 motion entries.
- Each prompt 12-30 words.
- Use the locked motion palette ingredients only.
- Pre and post should feel continuous - same direction of drift, complementary light.
- intensity = "low" for 80%, "medium" for hooks (scenes 1, 8, 15), never "high".
- Tie the motion subject to the slide subject (e.g. push-in on dollar sign for fiscal scenes).

Return ONLY the JSON.

==== SLIDE PROMPTS ====
{chunk_text}
==== END ====
"""
