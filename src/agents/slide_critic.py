"""
Agent: Slide Critic (vision) â€” DUAL-CALL version

Belt-and-suspenders quality gate. Each slide is evaluated by TWO independent
vision calls; both must vote pass for the slide to ship.

CALL 1 (OCR): transcribe the headline character-for-character.
  Pass criterion: normalized transcription == normalized expected headline.
  This is DETERMINISTIC â€” no LLM judgment about "close enough".

CALL 2 (JUDGMENT): qualitative review by the same model with a different
  prompt that does NOT include the expected headline. The model has to
  decide on its own whether the slide is legible, on-brand, and free of
  artifacts. By withholding the expected text we prevent it from rubber-
  stamping based on what we told it to expect.

FINAL VERDICT: pass only if (ocr_match) AND (judgment_pass).
Either disagreeing â†’ fail with detailed reasons from both halves.
"""
from __future__ import annotations

import base64, re
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from .base import AgentBase, VISION_ENDPOINT


def normalize(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def char_diff(expected: str, got: str, max_len: int = 80) -> str:
    e = expected[:max_len]; g = got[:max_len]
    if len(g) < len(e):
        return f"transcribed shorter ({len(g)}<{len(e)}): expected '{e}' got '{g}'"
    if len(g) > len(e):
        return f"transcribed longer ({len(g)}>{len(e)}): expected '{e}' got '{g}'"
    for i, (a, b) in enumerate(zip(e, g)):
        if a != b:
            return f"first divergence at char {i}: expected '...{e[max(0,i-3):i+5]}...' got '...{g[max(0,i-3):i+5]}...'"
    return f"normalized mismatch: expected '{e}' got '{g}'"


class SlideCritique(BaseModel):
    pass_fail: str
    headline_present: bool
    headline_legible: bool
    headline_matches_expected: bool
    style_on_brand: bool
    no_artifacts: bool
    description: str
    failure_reasons: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    transcribed_headline: str = ""
    expected_headline: str = ""
    normalized_match: bool = False
    judgment_pass: bool = False
    ocr_pass: bool = False
    agreement: str = ""  # "both_pass" | "both_fail" | "ocr_only" | "judgment_only"


class SlideCritic(AgentBase):
    name = "slide_critic"
    target_endpoint = VISION_ENDPOINT
    target_model = "vision"
    temperature = 0.0
    max_tokens = 600

    def _call_vision(self, sys_prompt: str, user_text: str, image_b64: str) -> dict:
        payload = {
            "model": self.target_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": user_text},
                ]}
            ]
        }
        timeout = httpx.Timeout(connect=10.0, read=180.0, write=120.0, pool=30.0)
        with httpx.Client(timeout=timeout, http2=False, headers={"Connection": "close"}) as client:
            r = client.post(f"{self.target_endpoint}/chat/completions", json=payload)
            r.raise_for_status()
            r.read()
            resp = r.json()
        content = resp["choices"][0]["message"]["content"]
        parsed = self.extract_json(content) or {}
        parsed.pop("_truncated_recovered", None)
        return parsed

    def _call_ocr(self, image_b64: str) -> str:
        sys_p = (
            "You are an OCR engine. Read text from images. Do NOT correct spelling. "
            "Do NOT guess at what was intended. Return EXACTLY the characters you see, "
            "even if they look misspelled. Output only JSON."
        )
        user_p = (
            "Read the LARGEST text on this slide character-by-character. "
            "Preserve any typos exactly. Do not normalize case or punctuation. "
            "Output ONLY this JSON: "
            '{"transcribed_headline": "EXACTLY THE CHARACTERS YOU SEE"}'
        )
        result = self._call_vision(sys_p, user_p, image_b64)
        return result.get("transcribed_headline", "")

    def _call_judgment(self, image_b64: str) -> dict:
        # NOTE: expected headline is intentionally NOT passed in. The model
        # must decide on its own whether the slide is legible / on-brand /
        # artifact-free, without being primed by what we want to see.
        sys_p = (
            "You are a strict editorial slide reviewer for an AI-generated podcast. "
            "Your job: judge slide quality on its own merits. You do NOT know what "
            "the headline was supposed to say â€” only judge what is visually present. "
            "Be honest about typos, garbled text, and artifacts. Output only JSON."
        )
        user_p = (
            "Evaluate this slide. Answer each question independently:\n"
            "1. headline_clearly_visible: Is there a single dominant headline that\'s "
            "clearly readable (not blurred, occluded, or split)?\n"
            "2. text_well_formed: Are all the words spelled like real English words "
            "(no garbled characters, no fused or missing letters)?\n"
            "3. style_on_brand: Editorial podcast aesthetic â€” dark background, clean "
            "single-color typography, ONE relevant icon/silhouette? Reject if it looks "
            "like a stock photo, comic, or cluttered web page.\n"
            "4. no_artifacts: No human faces, no NSFW, no duplicate text, no obvious "
            "AI rendering glitches?\n"
            "5. description: ONE sentence summarizing what is actually in the slide.\n\n"
            "Output ONLY: "
            '{"headline_clearly_visible": true|false, "text_well_formed": true|false, '
            '"style_on_brand": true|false, "no_artifacts": true|false, '
            '"description": "...", "concerns": ["..."]}'
        )
        return self._call_vision(sys_p, user_p, image_b64)

    def critique(self, image_path, expected_headline: str) -> SlideCritique:
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()

        # === CALL 1: OCR ===
        try:
            transcribed = self._call_ocr(b64)
        except Exception as e:
            transcribed = ""

        norm_e = normalize(expected_headline)
        norm_g = normalize(transcribed)
        ocr_pass = bool(norm_e) and (norm_e == norm_g)

        # === CALL 2: JUDGMENT (independent) ===
        try:
            j = self._call_judgment(b64)
        except Exception as e:
            j = {}

        visible = bool(j.get("headline_clearly_visible", False))
        well_formed = bool(j.get("text_well_formed", False))
        on_brand = bool(j.get("style_on_brand", False))
        clean = bool(j.get("no_artifacts", False))
        desc = j.get("description", "")
        concerns = j.get("concerns", []) or []

        judgment_pass = visible and well_formed and on_brand and clean

        # === BOTH MUST AGREE ===
        overall_pass = ocr_pass and judgment_pass

        failure_reasons = []
        if not ocr_pass:
            if not transcribed:
                failure_reasons.append("OCR returned no transcription")
            else:
                failure_reasons.append(f"OCR mismatch: {char_diff(expected_headline, transcribed)}")
        if not judgment_pass:
            if not visible:    failure_reasons.append("JUDGE: headline not clearly visible")
            if not well_formed: failure_reasons.append("JUDGE: text not well-formed (garbled/typos)")
            if not on_brand:   failure_reasons.append("JUDGE: style off-brand")
            if not clean:      failure_reasons.append("JUDGE: artifacts present")
            for c in concerns[:3]:
                failure_reasons.append(f"JUDGE concern: {c}")

        # Disagreement diagnostic
        if ocr_pass and not judgment_pass:
            agreement = "ocr_only"
        elif judgment_pass and not ocr_pass:
            agreement = "judgment_only"
        elif ocr_pass and judgment_pass:
            agreement = "both_pass"
        else:
            agreement = "both_fail"

        return SlideCritique(
            pass_fail="pass" if overall_pass else "fail",
            headline_present=bool(transcribed.strip()),
            headline_legible=visible and well_formed,
            headline_matches_expected=ocr_pass,
            style_on_brand=on_brand,
            no_artifacts=clean,
            description=desc or f"transcribed: {transcribed[:120]}",
            failure_reasons=failure_reasons,
            confidence=1.0 if overall_pass else (0.5 if agreement in ("ocr_only", "judgment_only") else 0.0),
            transcribed_headline=transcribed,
            expected_headline=expected_headline,
            normalized_match=ocr_pass,
            judgment_pass=judgment_pass,
            ocr_pass=ocr_pass,
            agreement=agreement,
        )
