"""Patched LTX 2.3 smoke test — set length directly on 547+577 (578 PrimitiveNode is gone)."""
import json, time, urllib.parse, urllib.request, copy
from pathlib import Path

HOST = "165.245.134.1:8188"
WF_FILE = Path(r"B:\amd-hackathon-bill-analyzer\comfy\workflows\ltx23-relay-i2v-cloud-api.json")
RELAY_FILE = Path(r"B:\amd-hackathon-bill-analyzer\eval\prompt-relay-bbb-ch01-day6.json")
DUR_FILE = Path(r"B:\amd-hackathon-bill-analyzer\eval\scene-durations-bbb-ch01-day6.json")
OUT_DIR = Path(r"B:\amd-hackathon-bill-analyzer\eval\day6-ltx-clips")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def http_post_json(url, payload, timeout=60.0):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_get_json(url, timeout=15.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_get_bytes(url, timeout=180.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def round_to_8n_plus_1(n):
    base = ((n - 1) // 8) * 8 + 1
    if base < n:
        base += 8
    return base


def smoke_one_scene(scene_meta, dur, fps=25, seed=42):
    template = json.loads(WF_FILE.read_text(encoding="utf-8"))
    raw = int(dur * fps)
    frames = round_to_8n_plus_1(raw)

    # Per-scene substitutions
    template["577"]["inputs"]["length"] = frames     # EmptyLTXVLatentVideo
    template["547"]["inputs"]["frames_number"] = frames  # LTXVEmptyLatentAudio
    template["164"]["inputs"]["frame_rate"] = fps    # LTXVConditioning
    template["547"]["inputs"]["frame_rate"] = fps    # LTXVEmptyLatentAudio
    template["605"]["inputs"]["local_prompts"] = scene_meta["smart_prompt"]
    template["584"]["inputs"]["image"] = "phase-c-master.png"
    template["561"]["inputs"]["noise_seed"] = seed
    template["604"]["inputs"]["filename_prefix"] = f"ltx23-{scene_meta['scene_id']}"

    print(f"  frames={frames} (target {raw}, rounded to 8N+1)")
    print(f"  prompt[:120]={scene_meta['smart_prompt'][:120]}...")
    print(f"  submitting...")

    t0 = time.perf_counter()
    try:
        res = http_post_json(f"http://{HOST}/prompt", {"prompt": template, "client_id": "ltx-smoke-3"})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"  submit error: HTTP {e.code}\n  {body[:1000]}")
        return None

    pid = res.get("prompt_id")
    if not pid:
        print(f"  no prompt_id: {res}")
        return None
    print(f"  prompt_id: {pid}")

    last_status_t = 0
    while True:
        elapsed = time.perf_counter() - t0
        if elapsed > 1500:
            print(f"  TIMEOUT after {elapsed:.0f}s")
            return None
        if elapsed - last_status_t > 30:
            print(f"    waiting {elapsed:.0f}s...")
            last_status_t = elapsed
        try:
            h = http_get_json(f"http://{HOST}/history/{pid}", timeout=10)
            if pid in h:
                history = h[pid]
                status = history.get("status", {})
                if status.get("status_str") == "error":
                    print(f"\n  EXECUTION ERROR after {elapsed:.0f}s")
                    msgs = status.get("messages", [])
                    for m in msgs:
                        if m[0] == "execution_error":
                            err = m[1]
                            print(f"    node: {err.get('node_id')} {err.get('node_type')}")
                            print(f"    {err.get('exception_type')}: {err.get('exception_message')}")
                            tb = err.get("traceback", [])
                            for line in tb[-3:]:
                                print(f"    {line.strip()}")
                    return None
                if status.get("completed"):
                    print(f"  done in {elapsed:.1f}s")
                    return history
        except Exception:
            pass
        time.sleep(5)


def download_and_save(history, scene_id, out_dir):
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
                local = out_dir / f"{scene_id}.mp4"
                local.write_bytes(data)
                written.append(local)
    return written


def main():
    relay = json.loads(RELAY_FILE.read_text(encoding="utf-8"))
    durs = json.loads(DUR_FILE.read_text(encoding="utf-8"))
    s = relay["scenes"][0]
    sid = s["scene_id"]
    dur = next(x["duration_s"] for x in durs["scenes"] if x["scene_id"] == sid)

    print(f"=== smoke test {sid} (duration {dur}s) ===")
    history = smoke_one_scene(s, dur)
    if history is None:
        return
    written = download_and_save(history, sid, OUT_DIR)
    if written:
        for f in written:
            print(f"  saved {f.name} ({f.stat().st_size:,} bytes)")
    else:
        print(f"  no video output. history outputs: {list((history.get('outputs') or {}).keys())}")
        for nid, no in (history.get('outputs') or {}).items():
            print(f"    [{nid}] {list(no.keys())}: {no}")


if __name__ == "__main__":
    import urllib.error
    main()