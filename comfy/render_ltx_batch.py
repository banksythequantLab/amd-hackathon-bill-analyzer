"""Batch-render all 19 scenes using the proven LTX 2.3 I2V workflow.
Skip scene-01 (already done). Per-scene substitution: frames, prompt, filename.
"""
import json, time, urllib.parse, urllib.request, urllib.error
from pathlib import Path

HOST = "127.0.0.1:8188"  # 3090 FORK: was 165.245.134.1:8188 on AMD cluster
# 3090 FORK: paths derived from script location (were hardcoded to old fork).
_REPO_ROOT = Path(__file__).resolve().parent.parent
WF_FILE = _REPO_ROOT / "comfy" / "workflows" / "ltx23-relay-i2v-cloud-api.json"
RELAY_FILE = _REPO_ROOT / "eval" / "prompt-relay-bbb-ch01-day6.json"
DUR_FILE = _REPO_ROOT / "eval" / "scene-durations-bbb-ch01-day6.json"
OUT_DIR = _REPO_ROOT / "eval" / "day6-ltx-clips"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def http_post_json(url, payload, timeout=60.0):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_get_json(url, timeout=15.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_get_bytes(url, timeout=300.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def round_to_8n_plus_1(n):
    base = ((n - 1) // 8) * 8 + 1
    if base < n:
        base += 8
    return base


def render_scene(scene_meta, dur, fps=25, seed=42):
    template = json.loads(WF_FILE.read_text(encoding="utf-8"))
    raw = int(dur * fps)
    frames = round_to_8n_plus_1(raw)
    sid = scene_meta["scene_id"]

    template["577"]["inputs"]["length"] = frames
    template["547"]["inputs"]["frames_number"] = frames
    template["547"]["inputs"]["frame_rate"] = fps
    template["164"]["inputs"]["frame_rate"] = fps
    template["605"]["inputs"]["local_prompts"] = scene_meta["smart_prompt"]
    template["584"]["inputs"]["image"] = "phase-c-master.png"
    template["561"]["inputs"]["noise_seed"] = seed
    template["604"]["inputs"]["filename_prefix"] = f"ltx23-{sid}"

    print(f"\n[{sid}] frames={frames} duration={dur}s")

    t0 = time.perf_counter()
    try:
        res = http_post_json(f"http://{HOST}/prompt", {"prompt": template, "client_id": f"ltx-batch-{sid}"})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"  submit error HTTP {e.code}: {body[:500]}")
        return None
    pid = res.get("prompt_id")
    if not pid:
        print(f"  no prompt_id: {res}")
        return None
    print(f"  prompt_id: {pid}")

    last_log = 0
    while True:
        elapsed = time.perf_counter() - t0
        if elapsed > 1500:
            print(f"  TIMEOUT after {elapsed:.0f}s")
            return None
        if elapsed - last_log > 30:
            print(f"    {elapsed:.0f}s...")
            last_log = elapsed
        try:
            h = http_get_json(f"http://{HOST}/history/{pid}", timeout=10)
            if pid in h:
                history = h[pid]
                status = history.get("status", {})
                if status.get("status_str") == "error":
                    print(f"  EXECUTION ERROR after {elapsed:.0f}s")
                    for m in status.get("messages", []):
                        if m[0] == "execution_error":
                            err = m[1]
                            print(f"    node={err.get('node_id')} type={err.get('exception_type')}")
                            print(f"    msg={err.get('exception_message')[:200]}")
                    return None
                if status.get("completed"):
                    print(f"  done in {elapsed:.1f}s")
                    return history
        except Exception:
            pass
        time.sleep(5)


def download(history, sid, out_dir):
    written = []
    for nid, node_out in (history.get("outputs") or {}).items():
        for key in ("videos", "gifs"):
            for vid in (node_out.get(key) or []):
                qs = urllib.parse.urlencode({
                    "filename": vid["filename"],
                    "subfolder": vid.get("subfolder", ""),
                    "type": vid.get("type", "output"),
                })
                data = http_get_bytes(f"http://{HOST}/view?{qs}")
                local = out_dir / f"{sid}.mp4"
                local.write_bytes(data)
                written.append(local)
    return written


def main():
    relay = json.loads(RELAY_FILE.read_text(encoding="utf-8"))
    durs_by_id = {x["scene_id"]: x["duration_s"] for x in json.loads(DUR_FILE.read_text(encoding="utf-8"))["scenes"]}

    total = len(relay["scenes"])
    print(f"=== batch-rendering {total} scenes ===")

    overall_t0 = time.perf_counter()
    succeeded = []
    failed = []

    for i, s in enumerate(relay["scenes"]):
        sid = s["scene_id"]
        existing = OUT_DIR / f"{sid}.mp4"
        if existing.exists() and existing.stat().st_size > 100_000:
            print(f"\n[{sid}] SKIP (already rendered, {existing.stat().st_size:,} bytes)")
            succeeded.append(sid)
            continue

        dur = durs_by_id.get(sid)
        if dur is None:
            print(f"\n[{sid}] no duration found, skipping")
            failed.append(sid)
            continue

        history = render_scene(s, dur)
        if history is None:
            failed.append(sid)
            continue

        written = download(history, sid, OUT_DIR)
        if written:
            for w in written:
                print(f"  saved {w.name} ({w.stat().st_size:,} bytes)")
            succeeded.append(sid)
        else:
            print(f"  no MP4 in history outputs!")
            failed.append(sid)

        elapsed_total = time.perf_counter() - overall_t0
        remaining = total - len(succeeded) - len(failed)
        print(f"  progress: {len(succeeded)}/{total} ok, {len(failed)} fail, ~{remaining} left, {elapsed_total/60:.1f}min total")

    print(f"\n=== BATCH COMPLETE in {(time.perf_counter()-overall_t0)/60:.1f} minutes ===")
    print(f"  succeeded: {len(succeeded)}/{total}")
    if failed:
        print(f"  failed: {failed}")


if __name__ == "__main__":
    main()