"""Quick pre-run verification: do the 19 scene PNGs all exist for relay scene_ids?"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
relay = json.loads((REPO / "eval" / "prompt-relay-bbb-ch01-day6.json").read_text(encoding="utf-8"))
scenes = relay.get("scenes", [])
print(f"scenes in relay: {len(scenes)}")

scenes_dir = REPO / "eval" / "day6-references"
missing = []
for s in scenes:
    sid = s["scene_id"]
    p = scenes_dir / f"{sid}.png"
    if not p.exists():
        missing.append(sid)

print(f"missing PNGs: {len(missing)} {missing}")

master = REPO / "eval" / "day6-master-candidates" / "master-seed256.png"
print(f"master exists: {master.exists()}  ({master})")

# show a couple sample intents so we know what kinds of shots are coming
for s in scenes[:3]:
    intent = s.get("reference_image_prompt", "")[:90].replace("\n", " ")
    print(f"  {s['scene_id']}: {intent}")
