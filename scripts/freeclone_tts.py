"""FreeClone TTS wrapper for the 3090-fork bill-analyzer pipeline.

This module wraps FreeClone's POST /api/podcast endpoint into a clean
Python API the agent suite can call. FreeClone runs locally on port 8300
(see B:/freeclone-backend/START_BFORK.bat) and uses VoxCPM2 + Whisper
large-v3 on CUDA.

Architecture context (why this exists):
  - The AMD-baseline pipeline submitted Qwen-TTS ComfyUI workflows to
    the AMD droplet's ComfyUI server. On the 3090 fork that's replaced
    with FreeClone's VoxCPM2-based podcast endpoint, which is faster
    and avoids the ComfyUI kill/restart dance between Qwen-Image/Wan/
    InfiniteTalk stages.

  - FreeClone returns a single rendered WAV blob synchronously per
    request. The pipeline already chunks bills into chapters and runs
    agents per-chapter, so calling FreeClone once per chapter is the
    natural granularity.

Public API:
  - render_podcast(script, output_path, voices=None, ...) -> RenderResult
  - VOICE_PRESETS: typed mapping of default speaker_id -> default voice
  - FreeCloneError: raised on non-2xx or unexpected response shape

Default voice mapping for Alex/Jordan (the AMD-canonical hosts):
  speaker_1 -> echo (deep male, American)  matches Alex profile
  speaker_2 -> nova (bright female, American)  matches Jordan profile

Endpoint contract (verified against server.py 2026-05-18):
  POST http://127.0.0.1:8300/api/podcast
  multipart/form-data:
    script:           JSON string [{speaker, text, lang}, ...]
    speed:            float string (0.5-2.0). VoxCPM2 ignores this in
                      2.0.x, accepted for forward compat.
    enhance:          "true"|"false". Free tier ignored; studio honors.
    voice_{N}:        uploaded file (custom voice clone, pro/studio only)
    default_voice_{N}: voice id from manifest (echo, nova, etc.) -- free OK
  Returns: 200 audio/wav file, OR 4xx JSON error, OR 5xx text.
  Filename in content-disposition: podcast_<job_id>.wav

Tier handling:
  Localhost requests get tier=studio via FreeClone's 127.0.0.1 ->
  FREE_RATE_BYPASS_IPS bypass (see server.py get_user_tier patch).
  Studio = unlimited speakers + script lines + audio enhance.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import httpx


logger = logging.getLogger(__name__)


# Default FreeClone endpoint on Johnson. Override via constructor or env.
FREECLONE_URL = "http://127.0.0.1:8300"

# Default voice mapping for the AMD-canonical two-host podcast format.
# Override per-call by passing voices={"1": "voice_id", "2": "..."}.
VOICE_PRESETS = {
    "1": "echo",   # Alex, deep male American
    "2": "nova",   # Jordan, bright female American
}


class FreeCloneError(Exception):
    """Raised when FreeClone returns a non-success response or the
    response shape doesn't match what we expect. The original status
    code and response body (or excerpt) are attached for inspection."""

    def __init__(self, message: str, status_code: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass
class ScriptLine:
    """One line of dialogue. speaker is a string id ('1', '2', ...)
    matching the keys used in the voices mapping. lang is an ISO-639
    code (FreeClone treats unknown languages as English)."""
    speaker: str
    text: str
    lang: str = "en"

    def to_dict(self) -> dict:
        return {"speaker": str(self.speaker), "text": self.text, "lang": self.lang}


@dataclass
class RenderResult:
    """What render_podcast returns. The WAV file has already been
    written to output_path; the other fields are reported for
    observability (logging, metrics, pipeline gating)."""
    output_path: Path
    elapsed_s: float
    audio_bytes: int
    server_filename: str | None  # the pod_<timestamp>_<uuid>.wav FreeClone chose
    n_lines: int
    n_speakers: int


def render_podcast(
    script: list[ScriptLine] | list[dict],
    output_path: Path | str,
    voices: dict[str, str] | None = None,
    speed: float = 1.0,
    enhance: bool = False,
    freeclone_url: str = FREECLONE_URL,
    timeout_s: float = 600.0,
) -> RenderResult:
    """Render a multi-speaker podcast WAV via FreeClone.

    Args:
        script: List of ScriptLine objects OR raw dicts matching
            {speaker, text, lang}. Empty lines are silently skipped
            by FreeClone, but for clarity callers should pre-filter.
        output_path: Where to write the resulting WAV. Parent dirs are
            created automatically. Existing file is overwritten.
        voices: Mapping of speaker_id -> default voice id. Defaults to
            VOICE_PRESETS (alex=echo, jordan=nova). Set to {} to send
            no default_voice_N fields (then FreeClone falls back to
            speaker-specific defaults or 400s if no voice info present).
        speed: Speech rate multiplier, 0.5-2.0. VoxCPM2.0.x ignores
            this; reserved for forward compatibility with future
            VoxCPM versions or alternate TTS backends.
        enhance: Run the audio enhancer (denoise + clean) on the
            output. Studio tier only. Ignored on free tier.
        freeclone_url: Base URL of the FreeClone server. Defaults to
            http://127.0.0.1:8300 -- which triggers the localhost
            studio-tier bypass in server.py.
        timeout_s: httpx request timeout. First call cold-loads
            Whisper + VoxCPM (~30-60s on a 3090), so the default is
            generous. Subsequent warm calls are ~4s per script line.

    Returns:
        RenderResult with the written file path, elapsed wall time,
        and server-reported metadata.

    Raises:
        FreeCloneError: non-2xx response, non-audio content type, or
            empty response body.
        httpx.RequestError: network-level failure (connection refused,
            DNS, etc.) -- caller should treat as "FreeClone not up".
        FileNotFoundError: parent directory of output_path doesn't exist
            and couldn't be created.
    """
    # Normalize script to list[dict]
    norm_script: list[dict] = [
        line.to_dict() if isinstance(line, ScriptLine) else dict(line)
        for line in script
    ]
    if not norm_script:
        raise ValueError("script is empty; FreeClone will 400 on this")

    # Default voice mapping. Caller can pass {} to skip.
    voices = VOICE_PRESETS if voices is None else voices

    # Build form data. httpx accepts dict for application/x-www-form
    # but for multipart/form we need files= or data= with each value
    # as str. Here we use data= (no file uploads -- all voices come
    # from default_voice_N keys).
    speakers_in_script = {str(line["speaker"]) for line in norm_script}
    data = {
        "script": json.dumps(norm_script),
        "speed": f"{max(0.5, min(2.0, speed)):.2f}",
        "enhance": "true" if enhance else "false",
    }
    for speaker_id, voice_id in voices.items():
        if str(speaker_id) in speakers_in_script:
            data[f"default_voice_{speaker_id}"] = voice_id

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "FreeClone render: %d lines, %d speakers, voices=%s, url=%s",
        len(norm_script), len(speakers_in_script),
        {k: v for k, v in voices.items() if k in speakers_in_script},
        freeclone_url,
    )

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.post(f"{freeclone_url}/api/podcast", data=data)
    except httpx.RequestError:
        # Let httpx errors propagate -- caller distinguishes "FreeClone
        # not running" (RequestError) from "FreeClone returned bad data"
        # (FreeCloneError).
        raise
    elapsed = time.perf_counter() - t0

    if r.status_code != 200:
        # Try to surface the structured error FastAPI emits, falling
        # back to the raw body excerpt.
        body_excerpt = r.text[:600] if r.text else ""
        try:
            detail = r.json().get("detail", body_excerpt)
        except Exception:
            detail = body_excerpt
        raise FreeCloneError(
            f"FreeClone returned {r.status_code}: {detail}",
            status_code=r.status_code,
            body=body_excerpt,
        )

    ctype = r.headers.get("content-type", "").lower()
    if "audio" not in ctype:
        raise FreeCloneError(
            f"FreeClone returned 200 but content-type is {ctype!r}, not audio/*. "
            f"First 200 bytes: {r.content[:200]!r}",
            status_code=200,
            body=r.text[:600] if hasattr(r, "text") else None,
        )

    if not r.content:
        raise FreeCloneError("FreeClone returned 200 with empty body", status_code=200)

    output_path.write_bytes(r.content)

    # Extract server-side filename from content-disposition if present.
    server_filename = None
    cd = r.headers.get("content-disposition", "")
    if "filename=" in cd:
        # filename="podcast_pod_<ts>_<uuid>.wav"
        try:
            server_filename = cd.split("filename=", 1)[1].strip().strip('"')
        except Exception:
            server_filename = None

    result = RenderResult(
        output_path=output_path,
        elapsed_s=elapsed,
        audio_bytes=len(r.content),
        server_filename=server_filename,
        n_lines=len(norm_script),
        n_speakers=len(speakers_in_script),
    )
    logger.info(
        "FreeClone render done: %d bytes in %.1fs (server file: %s)",
        result.audio_bytes, result.elapsed_s, result.server_filename,
    )
    return result


def healthcheck(freeclone_url: str = FREECLONE_URL, timeout_s: float = 3.0) -> dict:
    """Probe FreeClone /health. Returns the parsed JSON.

    Useful for pipeline pre-flight: if FreeClone isn't up, fail fast
    before generating the podcast script (which takes >2 min on its own).

    Returns:
        {"status": "ok", "gpu": "cuda", "whisperLoaded": bool,
         "voxcpmLoaded": bool, "activeJobs": int, "totalJobs": int}

    Raises:
        httpx.RequestError if FreeClone isn't reachable.
        FreeCloneError if /health returns non-200.
    """
    with httpx.Client(timeout=timeout_s) as client:
        r = client.get(f"{freeclone_url}/health")
    if r.status_code != 200:
        raise FreeCloneError(
            f"/health returned {r.status_code}",
            status_code=r.status_code,
            body=r.text[:300],
        )
    return r.json()


def list_default_voices(freeclone_url: str = FREECLONE_URL, timeout_s: float = 5.0) -> list[dict]:
    """List FreeClone's default voice manifest.

    Returns a list of {id, name, gender, description, file?} dicts.
    Useful for letting callers pick voices programmatically or for
    surfacing options in a UI.
    """
    with httpx.Client(timeout=timeout_s) as client:
        r = client.get(f"{freeclone_url}/api/voices")
    if r.status_code != 200:
        raise FreeCloneError(
            f"/api/voices returned {r.status_code}",
            status_code=r.status_code,
            body=r.text[:300],
        )
    return r.json()
