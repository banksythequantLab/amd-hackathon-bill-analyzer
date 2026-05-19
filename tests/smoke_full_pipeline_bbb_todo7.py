"""TODO #7 acceptance smoke: run_full_pipeline against bbb with TTS+compose
exercised on the new 3090-fork stack, all other stages cached from the
AMD-canonical 5/7 run.

skip_text=True  -> reuse existing script/slides/motions.json
skip_slides=True -> reuse 38 cached PNGs in slides/
skip_wan=True   -> reuse 38 cached MP4 clips in wan/
skip_tts=False  -> FRESH: generate 19 FLACs via FreeClone+VoxCPM2
                   then stage_compose re-encodes 19 scene MP4s with new audio,
                   stage_avatar_compose tries to assemble lipsync pairs (may fail),
                   stage_hybrid_compose stitches a final hybrid master.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Force unbuffered logging so the log file streams progress in real time.
import os; os.environ.setdefault("PYTHONUNBUFFERED", "1")

from scripts.make_podcast_cloud import run_full_pipeline

def main() -> int:
    print(f"=== TODO #7 acceptance smoke ===")
    print(f"target bill: bbb (Build Back Better Act, ch01)")
    print(f"flags: skip_text=True skip_slides=True skip_wan=True skip_tts=False")
    print()
    t0 = time.perf_counter()
    final = run_full_pipeline(
        'bbb',
        skip_text=True,
        skip_slides=True,
        skip_wan=True,
        skip_tts=False,
    )
    elapsed = time.perf_counter() - t0
    print()
    print(f"=== total wall: {elapsed:.1f}s ===")
    if final is None:
        print("FAIL: run_full_pipeline returned None")
        return 1
    print(f"final master: {final}")
    if Path(final).exists():
        sz = Path(final).stat().st_size
        print(f"  size: {sz/1024/1024:.1f} MB")
    else:
        print(f"  WARN: final path returned but file missing")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())