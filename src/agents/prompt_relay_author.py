"""
Agent #9: PromptRelay Authoring Agent

Reads a podcast script (the output of Agent #8) and authors a PromptRelayEncode
``smart_prompt`` for each scene. Output drives the Kijai/ComfyUI-PromptRelay node
which routes different text instructions to different time spans of an LTX 2.3
image-to-video generation, producing one coherent multi-beat clip per scene.

PromptRelay smart_prompt format (numbered-header style):
  Establish 30: <wide setting + character pose at scene open>
  | Action 50: <main action and dialogue beat>
  | Reaction 20: <reaction / camera move / closing pose>

Numbers are RELATIVE WEIGHTS, not frame counts. Header lines (e.g. "Establish 30:")
are stripped before encoding; only the description after the colon hits the
tokenizer. The first segment auto-becomes the global anchor for character
identity, lighting, and lens language.

DESIGN CHOICES:
  - One smart_prompt per "scene" - we group dialogue lines into scenes of 2-4
    lines each (so each scene becomes one ~5-8 sec LTX clip). 38 lines / 3 lines
    avg = ~13 scenes => ~13 video clips to stitch.
  - The agent reads all lines in a scene at once, plus a character description,
    plus the desired clip length, plus the previous scene's ending state (for
    continuity). It returns the smart_prompt text + the reference image prompt
    needed for Z-Image-Turbo to generate the opening frame.
  - Reference image prompt is the scene's establishing shot, written in
    Z-Image-Turbo style (concrete, photographic, no text overlays).

Two outputs per scene:
  1. reference_image_prompt - fed to Z-Image-Turbo to make the opening frame
  2. smart_prompt - fed to PromptRelayEncode in the LTX 2.3 I2V workflow

The agent runs on the spine endpoint (Qwen3-30B-A3B, 256K context). One scene's
worth of context is small (~1KB), so calls are fast.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field

from .base import AgentBase, SPINE_ENDPOINT


class RelayScenePrompt(BaseModel):
    """The two text prompts needed to produce one video clip."""
    scene_id: str = Field(description="Identifier like 'scene-01'")
    line_indices: list[int] = Field(description="Indices into the parent podcast script's lines[]")
    reference_image_prompt: str = Field(
        description=(
            "Concrete photographic prompt for the establishing frame. <=40 words. "
            "Includes character description, setting, lighting, framing. "
            "No text overlays. No abstract metaphors."
        )
    )
    smart_prompt: str = Field(
        description=(
            "PromptRelayEncode smart_prompt format. 3 segments separated by '|', each "
            "preceded by a header line like 'Establish 30:'. Numbers are relative weights."
        )
    )
    notes: Optional[str] = Field(default=None)


class RelayAuthoringOutput(BaseModel):
    bill: str = Field(description="Bill identifier (passed through)")
    chunk_id: str
    podcast_title: str
    character_alex: str = Field(
        description="Persistent description of Alex used in every reference image. "
                    "Includes face, hair, clothing, age. <=30 words."
    )
    character_jordan: str = Field(
        description="Persistent description of Jordan used in every reference image. "
                    "Includes face, hair, clothing, age. <=30 words."
    )
    studio: str = Field(
        description="Persistent description of the podcast studio set. <=25 words."
    )
    scenes: list[RelayScenePrompt] = Field(default_factory=list)
    note: Optional[str] = Field(default=None)
    model_config = {"extra": "allow"}


class PromptRelayAuthor(AgentBase):
    """Authors PromptRelayEncode smart_prompts from a podcast script."""
    name = "prompt_relay_author"
    target_endpoint = SPINE_ENDPOINT
    target_model = "spine"
    temperature = 0.4   # creative writing, like the podcast generator
    max_tokens = 6000   # ~13 scenes * ~150 tokens each = ~2K tokens; 6K is generous
    output_schema = RelayAuthoringOutput

    def system_prompt(self) -> str:
        return (
            "You are a video director and prompt engineer. You read a 2-host podcast "
            "script and author the prompts needed to render it as a multi-shot video using "
            "an image-to-video pipeline (LTX 2.3 with Kijai's PromptRelay node).\n\n"
            "Your output has three parts:\n"
            "  1. Persistent character descriptions for Alex and Jordan, used to keep "
            "     them consistent across every reference image. These describe face, "
            "     hair, clothing, age - 30 words max each. Choose looks that read clearly: "
            "     Alex is the curious explainer (think podcast-cohost casual), Jordan is "
            "     the analyst (think podcast-cohost professional). They are NOT the same "
            "     race or hairstyle.\n"
            "  2. Persistent studio description, used in every reference image so the set "
            "     stays the same. <=25 words. Modern podcast studio: warm lighting, two "
            "     mics, simple backdrop. Avoid logos.\n"
            "  3. A scene list. Group the podcast's dialogue lines into scenes of 2-4 "
            "     consecutive lines each. For each scene write:\n"
            "       - reference_image_prompt: the OPENING FRAME of the clip. Concrete, "
            "         photographic, names the character + studio + framing. No text overlays. "
            "         <=40 words. Format example: 'Wide shot of Alex and Jordan at the studio "
            "         desk. <character descriptions baked in>. Warm lighting from key spot. "
            "         Two condenser microphones in foreground.'\n"
            "       - smart_prompt: PromptRelayEncode format. EXACTLY 3 segments separated "
            "         by '|'. Each segment is one line: a header like 'Establish 30:' or "
            "         'Action 50:' or 'Reaction 20:' followed by the description. Numbers "
            "         are RELATIVE WEIGHTS for time allocation. Headers are stripped before "
            "         encoding so the description after the colon is what reaches the model. "
            "         First segment establishes setting (auto-becomes the global anchor). "
            "         Second segment is the action. Third segment is a reaction or camera "
            "         move. Keep each segment 1-2 sentences.\n\n"
            "EXAMPLE smart_prompt (single value, 3 segments):\n"
            "  Establish 30: Alex and Jordan seated at the wooden desk. Warm key light. "
            "  Two mics in foreground. Both hosts focused on the conversation. | Action 50: "
            "  Alex leans forward, gestures with both hands while speaking. Jordan listens "
            "  attentively, takes a note. | Reaction 20: Slow push-in to Jordan's face as she "
            "  begins to respond. Soft smile.\n\n"
            "RULES:\n"
            "  - Do NOT invent dialogue. The scenes describe the framing only; the "
            "    audio comes from the existing podcast TTS render.\n"
            "  - Do NOT use text overlays in any prompt. No '$30.8B' captions, no signs.\n"
            "  - Keep characters CONSISTENT. Bake the FULL character descriptions INLINE "
            "    into every reference_image_prompt. Do NOT use placeholders like "
            "    \"<Alex desc>\" — write out the description text in full each time.\n"
            "  - VARY the framing per scene. Wide two-shot, medium two-shot, single "
            "    of Alex, single of Jordan, over-shoulder, close-up - mix it up so the "
            "    video doesn't look static.\n"
            "  - Match scene boundaries to natural conversational beats (e.g. when the "
            "    speaker changes topic, group those lines together).\n"
            "  - HARD CAP: 16 scenes total. For a 38-line podcast that means about 2-3 "
            "    lines per scene. The current podcast is the canonical 38-line one; "
            "    target 13-16 scenes.\n"
            "  - Each scene should cover ~3-8 seconds of audio. The Alex/Jordan TTS lines "
            "    are roughly 4-7 seconds each, so 2-4 lines per scene is ~10-25 seconds of "
            "    a single LTX clip - too long. Cap scenes at 1-2 lines per scene if the "
            "    podcast is long. For a 38-line script, prefer 2-3 lines per scene = "
            "    13-19 scenes total.\n"
            "  - Return ONLY the JSON object. No commentary, no markdown fences."
        )

    def user_prompt(self, chunk_text: str, chunk_id: str, title_marker: str = "(unknown)", **context) -> str:
        bill = context.get("bill", "unknown")
        return f"""Author PromptRelay scene prompts from this podcast script.

Bill: {bill}
Chunk: {chunk_id}

Return a JSON object with this exact shape:
{{
  "bill": "{bill}",
  "chunk_id": "{chunk_id}",
  "podcast_title": "<copy from input>",
  "character_alex": "30s male, light brown hair, navy button-down, friendly demeanor",
  "character_jordan": "30s woman, dark hair pulled back, charcoal blazer over white shirt, focused gaze",
  "studio": "Modern podcast studio: wooden desk, two condenser mics, warm key lighting, soft blue backdrop",
  "scenes": [
    {{
      "scene_id": "scene-01",
      "line_indices": [0, 1],
      "reference_image_prompt": "Medium two-shot of Alex (30s male, light brown hair, navy button-down) and Jordan (30s woman, dark hair pulled back, charcoal blazer over white shirt) seated at a wooden podcast desk. Warm key light from above. Two condenser microphones in the foreground.",
      "smart_prompt": "Establish 30: Alex and Jordan seated at the wooden desk. Warm key light from above. Two mics in foreground. Both hosts smiling, ready to start. | Action 50: Alex leans into his mic and begins to speak. Jordan turns toward him attentively. | Reaction 20: Slow push-in to Jordan's face as she nods.",
      "notes": null
    }}
  ],
  "note": null
}}

Reminders:
- Return ONLY the JSON. No prose around it.
- Pick character descriptions ONCE and reuse them verbatim in every scene's reference_image_prompt.
- 13-19 scenes for a 38-line podcast.
- smart_prompt MUST have exactly 3 segments separated by ' | '.

==== PODCAST SCRIPT JSON ====
{chunk_text}
==== END ====
"""