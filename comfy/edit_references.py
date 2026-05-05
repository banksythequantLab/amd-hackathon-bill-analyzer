"""Render scene N reference images by EDITING the master image with Qwen-Image-Edit.

For each scene, we:
1. Upload the master reference (only needs to happen once)
2. Synthesize an edit instruction from the scene's reference_image_prompt
3. Run qwen-image-edit-api workflow on cloud ComfyUI
4. Download the edited image to local disk

The edit-instruction synthesis: Qwen-Image-Edit needs a SHORT directive about what
to change while keeping characters identical. We extract the framing/action delta
from each scene's prompt and prepend "keep the same two people in the same studio".

Mode flags:
  --master <path>: which file to use as the source (one of the master-seed*.png)
  --mode smoke    : edit scene 2 only as a test
  --mode batch    : edit scenes 2 through 19
  (scene 1 stays as the master itself - no edit needed)
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
DEFAULT_WORKFLOW = Path(r"B:\amd-hackathon-bill-analyzer\comfy\workflows\qwen-image-edit-api.json")
DEFAULT_OUT_DIR = Path(r"B:\amd-hackathon-bill-analyzer\eval\day6-references")


def http_post_json(url, payload, timeout=30.0):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get_json(url, timeout=10.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get_bytes(url, timeout=60.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def upload_image(host: str, image_path: Path, server_filename: str) -> str:
    """POST a multipart/form-data upload to ComfyUI's /upload/image endpoint.
    Returns the filename ComfyUI assigns (we pass overwrite=true so it's deterministic)."""
    boundary = "----PhaseCBoundary"
    body_parts = []
    image_bytes = image_path.read_bytes()
    body_parts.append(f"--{boundary}".encode("utf-8"))
    body_parts.append(f'Content-Disposition: form-data; name="image"; filename="{server_filename}"'.encode("utf-8"))
    body_parts.append(b"Content-Type: image/png")
    body_parts.append(b"")
    body_parts.append(image_bytes)
    body_parts.append(f"--{boundary}".encode("utf-8"))
    body_parts.append(b'Content-Disposition: form-data; name="overwrite"')
    body_parts.append(b"")
    body_parts.append(b"true")
    body_parts.append(f"--{boundary}--".encode("utf-8"))
    body = b"\r\n".join(body_parts)
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    req = urllib.request.Request(f"http://{host}/upload/image", data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60.0) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    print(f"  uploaded -> ComfyUI input/{result.get('name', server_filename)}")
    return result.get("name", server_filename)


def submit_workflow(host, workflow, client_id="phase-c-edit"):
    res = http_post_json(f"http://{host}/prompt", {"prompt": workflow, "client_id": client_id})
    if "prompt_id" not in res:
        raise RuntimeError(f"submit failed: {res}")
    return res["prompt_id"]


def wait_history(host, prompt_id, timeout_s=240.0, poll_s=1.5):
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


def download_outputs(host, history, out_dir, scene_id):
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for nid, node_out in (history.get("outputs") or {}).items():
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


def build_edit_workflow(template, source_filename, edit_instruction, seed, filename_prefix):
    wf = json.loads(json.dumps(template))
    for nid, node in wf.items():
        cls = node.get("class_type", "")
        ins = node.get("inputs", {})
        if cls == "LoadImage":
            ins["image"] = source_filename
        elif cls == "PrimitiveStringMultiline":
            ins["value"] = edit_instruction
        elif cls == "KSampler":
            ins["seed"] = seed
        elif cls == "SaveImage":
            ins["filename_prefix"] = filename_prefix
    return wf


def synthesize_edit_instruction(scene: dict, persistent: dict) -> str:
    """Turn a scene's reference_image_prompt into an edit instruction.

    Strategy: keep characters/studio identical, only describe the FRAMING/POSE delta
    relative to the master shot (which is a medium two-shot of both at the desk).
    The model is good at interpreting these as relative instructions.
    """
    rip = scene["reference_image_prompt"]
    # The reference_image_prompt already says things like "Wide shot of Alex...",
    # "Over-the-shoulder shot...", "Close-up on Jordan..." — Qwen-Image-Edit can
    # interpret those as edit directives directly. Just bookend with "keep same".
    return (
        f"Keep Alex and Jordan looking exactly the same as in the source image. "
        f"Keep the wooden podcast desk, two condenser microphones, and warm lighting identical. "
        f"Change only the framing and pose to match: {rip}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--master", required=True, type=Path,
                    help="Path to the master reference image (e.g. master-seed42.png)")
    ap.add_argument("--relay-file", type=Path, default=DEFAULT_RELAY_FILE)
    ap.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--mode", default="smoke", choices=["smoke", "batch", "single"])
    ap.add_argument("--scene", type=int, default=2, help="For --mode single, which scene number (1-19)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.master.exists():
        raise SystemExit(f"master image not found: {args.master}")

    relay = json.loads(args.relay_file.read_text(encoding="utf-8"))
    template = json.loads(args.workflow.read_text(encoding="utf-8"))

    persistent = {
        "alex": relay["character_alex"],
        "jordan": relay["character_jordan"],
        "studio": relay["studio"],
    }
    all_scenes = relay["scenes"]

    print(f"=== uploading master reference: {args.master.name} ===")
    server_name = upload_image(args.host, args.master, "phase-c-master.png")
    print(f"  master available as: {server_name}")
    print()

    # First, just save scene-01 as a copy of the master (no edit needed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    scene1_target = args.out_dir / "scene-01.png"
    shutil.copy2(args.master, scene1_target)
    print(f"=== scene-01 = master copy: {scene1_target.name} ({scene1_target.stat().st_size:,} bytes) ===")
    print()

    if args.mode == "smoke":
        # edit scene 2 only
        scenes_to_edit = [s for s in all_scenes if s["scene_id"] == "scene-02"]
    elif args.mode == "single":
        scenes_to_edit = [s for s in all_scenes if s["scene_id"] == f"scene-{args.scene:02d}"]
    else:
        # batch: scenes 2 through 19
        scenes_to_edit = [s for s in all_scenes if s["scene_id"] != "scene-01"]
    print(f"=== editing {len(scenes_to_edit)} scene(s) using master ===")
    print()

    failures = []
    written = []
    for s in scenes_to_edit:
        sid = s["scene_id"]
        instruction = synthesize_edit_instruction(s, persistent)
        print(f"  [{sid}] instruction: {instruction[:200]}{'...' if len(instruction) > 200 else ''}")
        try:
            wf = build_edit_workflow(template, server_name, instruction, args.seed, f"phase-c-edit-{sid}")
            t0 = time.perf_counter()
            pid = submit_workflow(args.host, wf)
            history = wait_history(args.host, pid, timeout_s=240.0)
            local = download_outputs(args.host, history, args.out_dir, sid)
            elapsed = time.perf_counter() - t0
            if local:
                print(f"  [{sid}] done in {elapsed:.1f}s -> {local[0].name} ({local[0].stat().st_size:,} bytes)")
                written.append(local[0])
            else:
                print(f"  [{sid}] FAILED no output")
                failures.append(sid)
        except Exception as e:
            print(f"  [{sid}] FAILED: {type(e).__name__}: {e}")
            failures.append(sid)
        print()

    print(f"=== summary: {len(written)} ok, {len(failures)} failed")
    if failures:
        print(f"   failed: {failures}")


if __name__ == "__main__":
    main()