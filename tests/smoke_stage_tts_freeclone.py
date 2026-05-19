"""Smoke test for the patched stage_tts in make_podcast_cloud.

Builds a tiny 4-line dialog script that mirrors the shape
PodcastScriptWriter produces, calls stage_tts directly, and verifies
that scene-NN.flac files appear with sensible sizes.

This is the TODO #6 acceptance test."""
from __future__ import annotations
import sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.make_podcast_cloud import stage_tts

def main() -> int:
    script = {
        "bill_short": "smoke_test",
        "headline": "make_podcast_cloud stage_tts FreeClone integration",
        "cold_open": "smoke test cold open",
        "dialog": [
            {"scene": 1, "speaker": "Alex",   "line": "Welcome to the bill analyzer smoke test.", "beat": "hook"},
            {"scene": 2, "speaker": "Jordan", "line": "We are testing the FreeClone TTS hook end to end.", "beat": "setup"},
            {"scene": 3, "speaker": "Alex",   "line": "Each line should produce a FLAC file on disk.", "beat": "body"},
            {"scene": 4, "speaker": "Jordan", "line": "And the file sizes should look plausible.", "beat": "close"},
        ],
    }
    eval_dir = REPO / "eval" / "smoke-stage-tts"
    eval_dir.mkdir(parents=True, exist_ok=True)
    # Clear any prior cache so we exercise the full path.
    for p in (eval_dir / "tts").glob("*.flac"):
        p.unlink()

    print(f"[smoke] stage_tts -> {eval_dir / 'tts'}")
    t0 = time.perf_counter()
    stage_tts(script, eval_dir)
    elapsed = time.perf_counter() - t0
    print(f"[smoke] total elapsed: {elapsed:.1f}s")

    files = sorted((eval_dir / "tts").glob("scene-*.flac"))
    print(f"[smoke] {len(files)} FLAC files written:")
    for p in files:
        kb = p.stat().st_size // 1024
        print(f"        {p.name}: {kb} KB")

    # Acceptance: 4 files, each > 10 KB (the cache threshold)
    assert len(files) == 4, f"expected 4 FLAC files, got {len(files)}"
    for p in files:
        assert p.stat().st_size > 10_000, f"file too small: {p}"
    # Verify FLAC magic bytes "fLaC"
    for p in files:
        assert p.read_bytes()[:4] == b"fLaC", f"not FLAC: {p.name}"
    print("[OK] all assertions passed")

    # Also test the cached path: second call should skip all 4
    print()
    print("[smoke 2] second call -- should hit cache for all 4")
    t1 = time.perf_counter()
    stage_tts(script, eval_dir)
    elapsed2 = time.perf_counter() - t1
    print(f"[smoke 2] cached pass elapsed: {elapsed2:.1f}s  (should be << first pass)")
    return 0


if __name__ == "__main__":
    sys.exit(main())