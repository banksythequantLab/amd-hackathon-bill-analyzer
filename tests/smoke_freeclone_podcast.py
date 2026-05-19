"""Smoke-test the /api/podcast endpoint against the patched FreeClone.

Validates:
  1. Localhost gets studio tier (bypass patch works) -- 2 lines is below
     even the free 4-line limit but we'll push to 6 lines (>4) to prove
     the studio tier is actually being assigned.
  2. multipart/form-data request format with script JSON + default_voice_N
     mappings is accepted.
  3. Response is a WAV audio file we can save and play.

If this works, the wrapper has a known-good contract to build against.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import httpx


FREECLONE = "http://127.0.0.1:8300"
OUT = Path(__file__).resolve().parent.parent / "eval" / "smoke-freeclone-podcast.wav"


def main() -> int:
    # 6 lines, 2 speakers -- proves studio tier (free is 4 lines max).
    script = [
        {"speaker": "1", "text": "Welcome to today's policy briefing. I am Alex.",     "lang": "en"},
        {"speaker": "2", "text": "And I am Jordan. Today we're discussing HR1.",       "lang": "en"},
        {"speaker": "1", "text": "HR1 is the One Big Beautiful Bill from 2025.",       "lang": "en"},
        {"speaker": "2", "text": "It covers agriculture, healthcare, and tax policy.", "lang": "en"},
        {"speaker": "1", "text": "Let's start with Title One: agriculture.",           "lang": "en"},
        {"speaker": "2", "text": "Sounds good. Where do you want to begin?",           "lang": "en"},
    ]

    # Use httpx for multipart; explicitly NOT using files= here because we
    # want default voices, not uploaded ones.
    data = {
        "script": json.dumps(script),
        "speed": "1.0",
        "enhance": "false",
        "default_voice_1": "echo",   # Alex: deep male, American
        "default_voice_2": "nova",   # Jordan: bright female, American
    }

    print(f"[smoke] POST {FREECLONE}/api/podcast")
    print(f"        script: {len(script)} lines, 2 speakers")
    print(f"        voices: speaker_1=echo, speaker_2=nova")
    print()

    t0 = time.perf_counter()
    try:
        # 600s timeout: first call cold-loads Whisper + VoxCPM (~30-60s)
        # then runs 6 TTS lines. Subsequent calls will be much faster.
        with httpx.Client(timeout=600.0) as client:
            r = client.post(f"{FREECLONE}/api/podcast", data=data)
    except Exception as e:
        print(f"[FAIL] request exception: {type(e).__name__}: {e}")
        return 1
    elapsed = time.perf_counter() - t0

    print(f"[smoke] status: {r.status_code}, elapsed: {elapsed:.1f}s")
    print(f"[smoke] content-type: {r.headers.get('content-type', '?')}")
    print(f"[smoke] content-length: {len(r.content)} bytes")
    print(f"[smoke] filename header: {r.headers.get('content-disposition', '?')}")

    if r.status_code != 200:
        print(f"[FAIL] non-200 response:")
        try:
            print(r.json())
        except Exception:
            print(r.text[:500])
        return 1

    ctype = r.headers.get("content-type", "").lower()
    if "audio" not in ctype:
        print(f"[FAIL] expected audio/wav, got {ctype}")
        print(r.text[:500])
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(r.content)
    print(f"\n[OK ] wrote {OUT} ({len(r.content)/1024:.0f} KB)")
    print(f"      ~{len(r.content)/2/16000:.1f}s of audio at 16kHz mono 16-bit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
