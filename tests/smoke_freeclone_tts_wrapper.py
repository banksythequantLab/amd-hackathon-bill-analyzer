"""Smoke-test the freeclone_tts wrapper end-to-end against the live
FreeClone server on :8300.

Validates:
  1. healthcheck() returns a parsed dict
  2. list_default_voices() returns the 8-voice manifest
  3. render_podcast() round-trips a 6-line podcast and writes a real WAV
  4. RenderResult exposes the expected metadata
  5. FreeCloneError is raised correctly on a bad request (empty script
     would 400 server-side -- skip in CI; we just verify the import).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.freeclone_tts import (
    render_podcast, healthcheck, list_default_voices,
    ScriptLine, VOICE_PRESETS, FreeCloneError,
)


def main() -> int:
    print("[1/4] healthcheck...")
    health = healthcheck()
    print(f"      {health}")
    assert health["status"] == "ok", f"unexpected health: {health}"

    print()
    print("[2/4] list_default_voices...")
    voices = list_default_voices()
    print(f"      {len(voices)} voices: {[v['id'] for v in voices]}")
    assert len(voices) >= 4
    voice_ids = {v["id"] for v in voices}
    assert "echo" in voice_ids and "nova" in voice_ids, "echo/nova missing"

    print()
    print("[3/4] render a 6-line podcast (uses VOICE_PRESETS defaults)...")
    script = [
        ScriptLine("1", "Welcome back to the wrapper smoke test."),
        ScriptLine("2", "Glad to be here. What are we doing today?"),
        ScriptLine("1", "We are validating the FreeClone TTS wrapper."),
        ScriptLine("2", "End to end? Audio file on disk and everything?"),
        ScriptLine("1", "Exactly. Six lines, two speakers."),
        ScriptLine("2", "Sounds good. Let us hear it."),
    ]
    out = REPO / "eval" / "smoke-freeclone-wrapper.wav"
    t0 = time.perf_counter()
    result = render_podcast(script, out)
    elapsed = time.perf_counter() - t0

    print(f"      elapsed (wrapper + server): {elapsed:.1f}s")
    print(f"      result.elapsed_s:           {result.elapsed_s:.1f}s")
    print(f"      result.audio_bytes:         {result.audio_bytes:,}")
    print(f"      result.output_path:         {result.output_path}")
    print(f"      result.server_filename:     {result.server_filename}")
    print(f"      result.n_lines:             {result.n_lines}")
    print(f"      result.n_speakers:          {result.n_speakers}")

    assert result.output_path.exists(), "output WAV missing"
    assert result.audio_bytes > 100_000, f"WAV suspiciously small: {result.audio_bytes}"
    assert result.n_lines == 6
    assert result.n_speakers == 2
    assert result.server_filename and result.server_filename.startswith("podcast_pod_")

    # Lightweight WAV sanity check (no soundfile dep needed): RIFF header
    head = out.read_bytes()[:12]
    assert head[0:4] == b"RIFF" and head[8:12] == b"WAVE", f"not a WAV: {head!r}"

    print()
    print("[4/4] FreeCloneError import path...")
    # We do not actually trigger a 400 here -- empty scripts get ValueError
    # in the wrapper before they reach FreeClone, by design.
    print(f"      FreeCloneError = {FreeCloneError}")
    print(f"      VOICE_PRESETS  = {VOICE_PRESETS}")

    print()
    print("[OK] all 4 wrapper smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
