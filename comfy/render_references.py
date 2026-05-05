"""Render scene-N reference images via cloud ComfyUI Z-Image-Turbo workflow.

For Phase C smoke + batch:
  - smoke (default): scene 1 only - confirms the API path works
  - batch: all 19 scenes from the canonical relay output

Pulls reference_image_prompt from the relay JSON, sets character_alex/jordan/studio
descriptions as a prefix to keep cross-scene consistency, calls cloud ComfyUI,
downloads the PNG to local disk for review.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_HOST = "165.245.134.1:8188"
DEFAULT_RELAY_FILE = Path(r"B:\amd-hackathon-bill-analyzer\eval\prompt-relay-bbb-ch01-day6.json")
DEFAULT_WORKFLOW = Path(r"B:\amd-hackathon-bill-analyzer\comfy\workflows\z-image-turbo-api.json")
DEFAULT_OUT_DIR = Path(r"B:\amd-hackathon-bill-analyzer\eval\day6-references")


def http_post_json(url: str, payload: dict, timeout: float = 30.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get_json(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get_bytes(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def submit_workflow(host: str, workflow: dict, client_id: str = "phase-c-render") -> str:
    res = http_post_json(f"http://{host}/prompt", {"prompt": workflow, "client_id": client_id})
    if "prompt_id" not in res:
        raise RuntimeError(f"submit failed: {res}")
    return res["prompt_id"]


def wait_history(host: str, prompt_id: str, timeout_s: float = 180.0, poll_s: float = 1.5) -> dict:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            h = http_get_json(f"http://{host}/history/{prompt_id}")
            if prompt_id in h:
                return h[prompt_id]
        except Exception:
            pass
        time.sleep(poll_s)
    raise TimeoutError(f"prompt {prompt_id} not in history after {timeout_s}s")


def download_outputs(host: str, history: dict, out_dir: Path, scene_id: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    outputs = history.get("outputs", {})
    for node_id, node_out in outputs.items():
        for img in (node_out.get("images") or []):
            qs = urllib.parse.urlencode({
                "filename": img["filename"],
                "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output"),
            })
            data = http_get_bytes(f"http://{host}/view?{qs}")
            local = out_dir / f"{scene_id}.png"
            local.write_bytes(data)
            written.append(local)
    return written


def build_workflow(template: dict, prompt_text: str, seed: int, width: int, height: int, filename_prefix: str) -> dict:
    """Mutate a deep copy of the workflow template with our scene parameters."""
    wf = json.loads(json.dumps(template))  # deep copy

    # Cloud has fp8 variant of qwen_3_4b - patch the CLIPLoader
    for nid, node in wf.items():
        cls = node.get("class_type", "")
        ins = node.get("inputs", {})
        if cls == "CLIPLoader" and ins.get("clip_name", "").startswith("qwen_3_4b"):
            ins["clip_name"] = "qwen_3_4b_fp8_mixed.safetensors"
        if cls == "CLIPTextEncode" and "clip" in ins:  # set the prompt
            ins["text"] = prompt_text
        if cls == "EmptySD3LatentImage":
            ins["width"] = width
            ins["height"] = height
            ins["batch_size"] = 1
        if cls == "KSampler":
            ins["seed"] = seed
        if cls == "SaveImage":
            ins["filename_prefix"] = filename_prefix
    return wf


def render_scene(host: str, workflow_template: dict, scene: dict, persistent: dict, out_dir: Path, seed: int, width: int, height: int) -> Path:
    """Build a full prompt string from scene + persistent character/studio descriptions, submit, download."""
    sid = scene["scene_id"]
    base_prompt = scene["reference_image_prompt"]
    # Bake in studio if not already, and add a quality preamble
    full_prompt = (
        f"Cinematic photograph, professional podcast studio. {base_prompt} "
        f"{persistent['studio']} Sharp focus, natural skin tones, detailed faces."
    )
    print(f"  [{sid}] submitting...")
    print(f"      prompt: {full_prompt[:140]}{'...' if len(full_prompt) > 140 else ''}")
    wf = build_workflow(workflow_template, full_prompt, seed, width, height, f"phase-c-{sid}")
    t0 = time.perf_counter()
    prompt_id = submit_workflow(host, wf)
    history = wait_history(host, prompt_id, timeout_s=180.0)
    written = download_outputs(host, history, out_dir, sid)
    elapsed = time.perf_counter() - t0
    if not written:
        print(f"  [{sid}] WARNING: no images written")
        return None
    print(f"  [{sid}] done in {elapsed:.1f}s -> {written[0].name} ({written[0].stat().st_size:,} bytes)")
    return written[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--relay-file", type=Path, default=DEFAULT_RELAY_FILE)
    ap.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--mode", default="smoke", choices=["smoke", "batch"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    args = ap.parse_args()

    relay = json.loads(args.relay_file.read_text(encoding="utf-8"))
    template = json.loads(args.workflow.read_text(encoding="utf-8"))

    persistent = {
        "alex": relay["character_alex"],
        "jordan": relay["character_jordan"],
        "studio": relay["studio"],
    }
    scenes = relay["scenes"]
    if args.mode == "smoke":
        scenes = scenes[:1]
    print(f"=== rendering {len(scenes)} scene(s), mode={args.mode}, host={args.host}")
    print(f"  alex:   {persistent['alex']}")
    print(f"  jordan: {persistent['jordan']}")
    print(f"  studio: {persistent['studio']}")
    print()

    out = []
    failures = []
    for s in scenes:
        try:
            p = render_scene(args.host, template, s, persistent, args.out_dir, args.seed, args.width, args.height)
            if p:
                out.append(p)
            else:
                failures.append(s["scene_id"])
        except Exception as e:
            print(f"  [{s['scene_id']}] FAILED: {type(e).__name__}: {e}")
            failures.append(s["scene_id"])
    print()
    print(f"=== done: {len(out)} ok, {len(failures)} failed")
    if failures:
        print(f"   failed: {failures}")


if __name__ == "__main__":
    main()