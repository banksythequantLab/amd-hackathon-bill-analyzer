"""
Agent #10: SceneCritic

Vision-based quality gate for the 19 reference scenes. For each generated scene
image, compares it side-by-side with the master image and judges whether to
keep the frame or re-render.

Input  : master image path + scene image path + scene metadata (the relay scene record)
Output : { scene_id, verdict, confidence, framing_observed, issues[], suggested_fix }
  verdict ∈ {keep, reroll, caveat}
    - keep    : both characters present, plausibly the same identities, framing matches intent
    - reroll  : missing a host, characters swapped, identity drift, broken composition
    - caveat  : usable but flag specific issue (e.g. wrong-side seating but otherwise fine)

Runs against the vllm-vision endpoint (Qwen3-VL-8B-Thinking-FP8 at port 8002).
The Thinking variant produces a chain of reasoning plus a final answer; we
extract the JSON from the answer.

This is NOT an AgentBase subclass because the I/O shape (multi-image input,
per-image output) is fundamentally different from chunk-text agents. It calls
the OpenAI-compatible vision API directly.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, Field, field_validator


VISION_ENDPOINT_DEFAULT = "http://165.245.134.1:8002/v1"
VISION_MODEL = "vision"
TIMEOUT_S = 120.0


class SceneCritique(BaseModel):
    scene_id: str
    verdict: str = Field(description="keep | reroll | caveat")
    confidence: str = Field(description="high | medium | low")
    framing_observed: str = Field(description="One-sentence description of the actual framing in the image, e.g. 'medium two-shot, both hosts visible'")
    both_hosts_visible: bool = Field(description="True if both Alex and Jordan appear in frame")
    seating_matches_master: bool = Field(description="True if hosts are on the same sides of the desk as in the master")
    identity_consistent: bool = Field(description="True if the two faces look like the same Alex+Jordan as the master")
    back_to_camera: bool = Field(default=False, description="True if BOTH visible hosts have their backs to the camera (faces hidden)")
    director_intent_followed: bool = Field(default=True, description="True if the framing matches what the director's intent text asked for")
    issues: list[str] = Field(default_factory=list, description="Concrete problems, empty if none")
    suggested_fix: Optional[str] = Field(default=None, description="If reroll, what to tell Qwen-Image-Edit on the retry")

    @field_validator("verdict")
    @classmethod
    def _verdict_valid(cls, v: str) -> str:
        if v not in ("keep", "reroll", "caveat"):
            raise ValueError(f"verdict must be keep/reroll/caveat, got {v!r}")
        return v

    @field_validator("confidence")
    @classmethod
    def _conf_valid(cls, v: str) -> str:
        if v not in ("high", "medium", "low"):
            raise ValueError(f"confidence must be high/medium/low, got {v!r}")
        return v


def _b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _build_messages(master: Path, scene_img: Path, scene_meta: dict) -> list[dict]:
    """Two-image multi-modal prompt. Image 1 = master (the canonical look),
    Image 2 = scene candidate (to be judged)."""
    intent = scene_meta.get("reference_image_prompt", "(no intent provided)")
    sid = scene_meta.get("scene_id", "?")

    system = (
        "You are a film QA reviewer. You compare two images and decide whether the second "
        "image is acceptable as a frame in a podcast video where character identity and "
        "studio continuity must be preserved across all shots.\n\n"
        "Image 1 is the MASTER reference: the canonical look of Alex (left) and Jordan "
        "(right) in their podcast studio. Treat the master as ground truth for who Alex "
        "and Jordan are (faces, hair, clothes), the desk, and the studio.\n\n"
        "Image 2 is a CANDIDATE for a different scene. The director's intent text "
        "describes what THIS scene's framing was supposed to be. Judge the candidate "
        "against the master AND the director's intent.\n\n"
        "FRAMING CALCULUS — what each verdict means:\n\n"
        "verdict=keep when:\n"
        "  - The framing matches the director's intent (or close enough)\n"
        "  - Identity is consistent with the master\n"
        "  - Composition is clean (no melted faces, broken hands, surreal artifacts)\n"
        "  - If intent calls for a single-host close-up, having only one host IN FRAME "
        "    is correct — that is NOT a 'missing host' failure\n"
        "  - An over-the-shoulder shot legitimately shows the back of one host's head; "
        "    that is intentional framing, return KEEP\n\n"
        "verdict=caveat when:\n"
        "  - Hosts are on the WRONG sides of the desk (Alex on right, Jordan on left) "
        "    in a two-shot — usable but flagged\n"
        "  - Identity drift is mild but recognizable\n"
        "  - Framing is plausibly close to intent but not quite right\n\n"
        "verdict=reroll when:\n"
        "  - The intent calls for a TWO-SHOT (medium two-shot, wide shot) but the image "
        "    only shows ONE host\n"
        "  - BOTH hosts have their backs to the camera in the same shot — neither face "
        "    is visible (a single back-of-head over-shoulder shot is fine; two backs is not)\n"
        "  - Identity drift is severe (Alex or Jordan looks like a totally different person)\n"
        "  - Composition is broken: cropped face on the wrong axis, melted hands, "
        "    duplicated body parts, surreal artifacts\n\n"
        "Set 'both_hosts_visible' to whether both Alex and Jordan are in frame (regardless "
        "of intent). Set 'director_intent_followed' to whether the framing matches what was "
        "requested. Set 'back_to_camera' to True only if BOTH visible hosts have their "
        "backs/sides turned so neither face is readable.\n\n"
        "Return ONLY a single JSON object matching the schema. No prose, no markdown fences."
    )

    user_text = (
        f"Scene ID: {sid}\n\n"
        f"DIRECTOR'S INTENT for this frame:\n  {intent}\n\n"
        f"Compare image 2 (CANDIDATE) against image 1 (MASTER), keeping the director's intent in mind. "
        f"For example: if intent says 'Close-up of Alex' and the image shows Alex alone, that is CORRECT. "
        f"If intent says 'Wide shot of both hosts' but the image shows only one host, that is a FAILURE.\n\n"
        f"Return JSON with this exact shape (no markdown fences):\n"
        f"{{\n"
        f'  "scene_id": "{sid}",\n'
        f'  "verdict": "keep" | "reroll" | "caveat",\n'
        f'  "confidence": "high" | "medium" | "low",\n'
        f'  "framing_observed": "short description of what is actually in image 2",\n'
        f'  "both_hosts_visible": true | false,\n'
        f'  "seating_matches_master": true | false,\n'
        f'  "identity_consistent": true | false,\n'
        f'  "back_to_camera": true | false,\n'
        f'  "director_intent_followed": true | false,\n'
        f'  "issues": ["one issue per string"],\n'
        f'  "suggested_fix": "edit instruction for Qwen-Image-Edit retry, or null if keep"\n'
        f"}}\n\n"
        f"IMPORTANT: Output the JSON object directly. Do NOT include thinking, reasoning, "
        f"or any text before the opening curly brace. Start your reply with the character "
        f"and end with the closing brace."
    )

    master_b64 = _b64_image(master)
    scene_b64 = _b64_image(scene_img)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "text", "text": "MASTER reference (image 1):"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{master_b64}"}},
            {"type": "text", "text": "CANDIDATE to judge (image 2):"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{scene_b64}"}},
            {"type": "text", "text": user_text},
        ]},
    ]


def _extract_json_from_response(content: str) -> dict:
    """Qwen3-VL-Thinking emits <think>...</think> reasoning, then the JSON answer."""
    # Strip leading/trailing whitespace + markdown fences if any
    text = content.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    # Strip code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    # Find the first { and the last } and parse
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in response: {text[:300]!r}")
    return json.loads(text[start:end + 1])


def critique_scene(
    master_path: Path,
    scene_path: Path,
    scene_meta: dict,
    *,
    endpoint: str = VISION_ENDPOINT_DEFAULT,
    max_tokens: int = 4000,
    temperature: float = 0.0,
) -> tuple[SceneCritique, dict]:
    """Returns (validated_critique, metric_dict)."""
    messages = _build_messages(master_path, scene_path, scene_meta)
    payload = {
        "model": VISION_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.perf_counter()
    with httpx.Client(timeout=TIMEOUT_S) as client:
        r = client.post(f"{endpoint}/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    elapsed = time.perf_counter() - t0
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    parsed = _extract_json_from_response(content)
    critique = SceneCritique(**parsed)
    metric = {
        "scene_id": scene_meta.get("scene_id"),
        "elapsed_s": round(elapsed, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "raw_response_chars": len(content),
    }
    return critique, metric