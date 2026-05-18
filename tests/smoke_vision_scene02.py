"""Vision-model smoke test: critique scene-02 vs master-seed256.

Goal: prove that
  (a) the 'vision' Ollama alias loads and responds via OpenAI-compat chat API
  (b) scene_critic.critique_scene() returns a validated SceneCritique
  (c) the Instruct variant does NOT emit <think> blocks that break parsing

If this works, scene_critic is unblocked on the 3090 fork and we can run
the full 19-scene pass via comfy/run_scene_critic.py.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.agents.scene_critic import critique_scene, VISION_ENDPOINT_DEFAULT  # noqa: E402


MASTER = REPO / "eval" / "day6-master-candidates" / "master-seed256.png"
SCENE = REPO / "eval" / "day6-references" / "scene-02.png"
RELAY = REPO / "eval" / "prompt-relay-bbb-ch01-day6.json"


def main() -> int:
    for p in (MASTER, SCENE, RELAY):
        if not p.exists():
            print(f"[FAIL] missing: {p}")
            return 2

    relay = json.loads(RELAY.read_text(encoding="utf-8"))
    scene_meta = next((s for s in relay["scenes"] if s["scene_id"] == "scene-02"), None)
    if scene_meta is None:
        print("[FAIL] scene-02 not in prompt relay")
        return 2
    print(f"[INFO] scene-02 intent: {scene_meta.get('reference_image_prompt', '')[:120]!r}")

    print(f"[INFO] endpoint: {VISION_ENDPOINT_DEFAULT}  model: vision")
    print(f"[INFO] master: {MASTER.name}  ({MASTER.stat().st_size/1024:.0f} KB)")
    print(f"[INFO] scene:  {SCENE.name}   ({SCENE.stat().st_size/1024:.0f} KB)")
    print()

    print("[RUN]  critiquing ...")
    t0 = time.perf_counter()
    try:
        critique, metric = critique_scene(MASTER, SCENE, scene_meta)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"[FAIL] {type(e).__name__}: {e}  (after {elapsed:.1f}s)")
        return 1
    elapsed = time.perf_counter() - t0
    print(f"[OK ]  elapsed={elapsed:.1f}s")
    print(f"       prompt_tokens={metric['prompt_tokens']}  completion_tokens={metric['completion_tokens']}")
    print()
    print(f"  scene_id:               {critique.scene_id}")
    print(f"  verdict:                {critique.verdict}")
    print(f"  confidence:             {critique.confidence}")
    print(f"  framing_observed:       {critique.framing_observed}")
    print(f"  both_hosts_visible:     {critique.both_hosts_visible}")
    print(f"  seating_matches_master: {critique.seating_matches_master}")
    print(f"  identity_consistent:    {critique.identity_consistent}")
    print(f"  back_to_camera:         {critique.back_to_camera}")
    print(f"  director_intent_followed: {critique.director_intent_followed}")
    print(f"  issues ({len(critique.issues)}):           {critique.issues}")
    print(f"  suggested_fix:          {critique.suggested_fix}")

    # Persist for downstream comparison.
    out_file = REPO / "eval" / "smoke-vision-scene02-vs-seed256.json"
    out_file.write_text(json.dumps({
        "test": "vision_smoke_scene02_vs_seed256",
        "model_alias": "vision",
        "endpoint": VISION_ENDPOINT_DEFAULT,
        "metric": metric,
        "critique": critique.model_dump(),
    }, indent=2), encoding="utf-8")
    print(f"\n[ARTIFACT] {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
