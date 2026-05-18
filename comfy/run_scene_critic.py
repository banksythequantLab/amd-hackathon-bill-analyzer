"""Run agent #10 (SceneCritic) across the 19 day6-references scenes.

For each scene, calls Qwen3-VL-8B-Thinking on the cloud vision endpoint with
(master image, scene image, scene metadata) and saves the structured critique.

Usage:
    python comfy/run_scene_critic.py
        [--endpoint http://127.0.0.1:11434/v1]   # was http://165.245.134.1:8002/v1 on AMD
        [--master eval/day6-master-candidates/master-seed256.png]
        [--scenes-dir eval/day6-references]
        [--relay-file eval/prompt-relay-bbb-ch01-day6.json]
        [--out eval/scene-critic-bbb-ch01-day6.json]
        [--smoke]   # only do scene-02 as a smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.scene_critic import (
    SceneCritique,
    critique_scene,
    VISION_ENDPOINT_DEFAULT,
)


def main():
    # 3090 FORK: argparse defaults derived from script location, were
    # hardcoded to B:\amd-hackathon-bill-analyzer\... (old fork).
    repo_root = Path(__file__).resolve().parent.parent
    eval_dir = repo_root / "eval"

    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=VISION_ENDPOINT_DEFAULT)
    ap.add_argument(
        "--master",
        type=Path,
        default=eval_dir / "day6-master-candidates" / "master-seed256.png",
    )
    ap.add_argument(
        "--scenes-dir",
        type=Path,
        default=eval_dir / "day6-references",
    )
    ap.add_argument(
        "--relay-file",
        type=Path,
        default=eval_dir / "prompt-relay-bbb-ch01-day6.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=eval_dir / "scene-critic-bbb-ch01-day6.json",
    )
    ap.add_argument("--smoke", action="store_true", help="Only critique scene-02 (smoke test)")
    args = ap.parse_args()

    if not args.master.exists():
        raise SystemExit(f"master not found: {args.master}")
    relay = json.loads(args.relay_file.read_text(encoding="utf-8"))
    scenes = relay["scenes"]
    if args.smoke:
        scenes = [s for s in scenes if s["scene_id"] == "scene-02"]

    print(f"=== SceneCritic: {len(scenes)} scenes vs master {args.master.name}")
    print(f"  endpoint: {args.endpoint}")
    print(f"  scenes dir: {args.scenes_dir}")
    print()

    results = []
    metrics = []
    failures = []
    summary = {"keep": 0, "caveat": 0, "reroll": 0}
    t_start = time.perf_counter()

    for s in scenes:
        sid = s["scene_id"]
        scene_path = args.scenes_dir / f"{sid}.png"
        if not scene_path.exists():
            print(f"  [{sid}] MISSING scene file: {scene_path}")
            failures.append(sid)
            continue
        try:
            critique, metric = critique_scene(args.master, scene_path, s, endpoint=args.endpoint)
            results.append(critique.model_dump())
            metrics.append(metric)
            summary[critique.verdict] += 1
            issues_str = "; ".join(critique.issues) if critique.issues else "—"
            print(
                f"  [{sid}] {critique.verdict.upper()} ({critique.confidence}) "
                f"in {metric['elapsed_s']}s  | both={critique.both_hosts_visible} "
                f"seat={critique.seating_matches_master} id={critique.identity_consistent} "
                f"back2cam={critique.back_to_camera} intent={critique.director_intent_followed}"
            )
            if critique.issues:
                for issue in critique.issues:
                    print(f"      • {issue}")
            if critique.suggested_fix:
                print(f"      fix: {critique.suggested_fix[:140]}")
        except Exception as e:
            print(f"  [{sid}] FAILED: {type(e).__name__}: {e}")
            failures.append(sid)

    total_elapsed = time.perf_counter() - t_start
    out_payload = {
        "agent": "scene_critic",
        "master_image": args.master.name,
        "elapsed_s": round(total_elapsed, 2),
        "summary": summary,
        "results": results,
        "metrics": metrics,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print()
    print(f"=== summary: {summary}, failures={len(failures)}, total {total_elapsed:.1f}s")
    print(f"   wrote {args.out}")


if __name__ == "__main__":
    main()