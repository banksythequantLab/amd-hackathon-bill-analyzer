"""InfiniteTalk two-speaker integration for the Dead Air podcast pipeline.

Wraps Derek's verified `video_wan2_1_infinitetalk mask api.json` workflow
(Wan 2.1 I2V 14B + InfiniteTalk multi patch + lightx2v 4-step LoRA) into
a Python builder that generates one pair-render per call.

DURATION MATH:
  total_frames     = ceil((dur_alex + dur_jordan) * fps)
  motion_frame_count = 9  (the model's inter-stage priming overlap)
  Two-stage chunking covers up to 633 frames (25.32s) per pair.
  Single-stage covers up to 321 frames (12.84s).

ASSETS REQUIRED IN ComfyUI's INPUT DIR (uploaded once, reused per render):
  podcast_pair_alex_mask.png    RGBA painted PNG: ref image with alex region
                                in alpha=0 (so LoadImage extracts it as mask)
  podcast_pair_jordan_mask.png  Same idea, jordan region in alpha=0
  pair_NN_alex.flac             TTS for Alex's line (uploaded per render)
  pair_NN_jordan.flac           TTS for Jordan's line (uploaded per render)
"""
from __future__ import annotations
import math, time, uuid
from pathlib import Path
from typing import Optional

INFINITETALK_DEFAULTS = {
    "stage_max": 321,
    "motion_frame_count": 9,
    "fps": 25,
    "width": 832,
    "height": 480,
    "audio_scale": 1,
    "sampler": "euler",
    "scheduler": "normal",
    "steps": 4,
    "denoise": 1,
    "cfg": 1,
    "unet": "Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors",
    "lora": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
    "vae": "Wan2_1_VAE_bf16.safetensors",
    "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "audio_encoder": "wav2vec2-chinese-base_fp16.safetensors",
    "model_patch": "wan2.1_infiniteTalk_multi_fp16.safetensors",
}

DEFAULT_PROMPT = (
    "Two podcast hosts at a Dead Air Broadcasting desk having a focused conversation. "
    "Alex on the left, Jordan on the right. Subtle lifelike facial expression and "
    "mouth movement, natural blinking, eye contact between speakers. The camera holds "
    "steady on a medium two-shot. Cyberpunk studio backdrop, cyan accent lighting."
)


def _quantize_length(L: int) -> int:
    """Round L up to the nearest value where L % 4 == 1.

    The WanInfiniteTalkToVideo node's audio embedding pipeline reshapes
    `latter_frame_audio_emb` with a chunk-of-4 rearrange. The relevant
    tensor has size (L - 1), so (L - 1) must be divisible by 4 — i.e.
    L % 4 == 1. If L violates this, ComfyUI raises EinopsError mid-render.

    Adds 0..3 frames of padding (max 0.12s @ 25fps), imperceptible.
    """
    rem = L % 4
    if rem == 1:
        return L
    # (1 - rem) % 4 gives correct add: rem=0 -> +1, rem=2 -> +3, rem=3 -> +2
    return L + ((1 - rem) % 4)


def compute_stage_lengths(total_frames: int, stage_max: int = 321,
                          motion_frame_count: int = 9):
    """Return (stage_1_len, stage_2_len) for target total visible frame count.

    Returns (total_frames, 0) for short pairs that fit in one stage.
    Raises ValueError if pair exceeds the 2-stage envelope.

    Both returned lengths are quantized to L % 4 == 1 (see _quantize_length).
    """
    max_two_stage = stage_max + (stage_max - motion_frame_count)
    if total_frames > max_two_stage:
        raise ValueError(
            f"pair too long for 2-stage rendering: {total_frames} frames "
            f"> max {max_two_stage}. split pairs or implement 3-stage chunking."
        )
    if total_frames <= stage_max:
        return _quantize_length(total_frames), 0
    stage_1_len = stage_max  # 321 % 4 == 1 already, no quantize needed
    stage_2_raw = total_frames - stage_1_len + motion_frame_count
    stage_2_len = _quantize_length(stage_2_raw)
    return stage_1_len, stage_2_len


def make_infinitetalk_pair_workflow(
    audio_1_filename: str,
    audio_2_filename: str,
    image_1_filename: str,
    image_2_filename: str,
    total_frames: int,
    prompt: str = DEFAULT_PROMPT,
    seed: int = 42,
    prefix: str = "infinitetalk_pair",
    cfg: dict = None,
):
    """Build a ComfyUI API-format workflow for one InfiniteTalk pair render.

    Mirrors the verified workflow JSON structure but parameterizes audio
    + image filenames, frame lengths, seeds, prefix. Always two_speakers mode.
    """
    cfg = {**INFINITETALK_DEFAULTS, **(cfg or {})}
    stage_1_len, stage_2_len = compute_stage_lengths(
        total_frames, cfg["stage_max"], cfg["motion_frame_count"]
    )
    single_stage = (stage_2_len == 0)
    seed_2 = seed ^ 0xA5A5A5A5

    wf = {
        "13": {"class_type": "UNETLoader", "inputs": {
            "unet_name": cfg["unet"], "weight_dtype": "default"}},
        "16": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": cfg["clip"], "type": "wan", "device": "default"}},
        "29": {"class_type": "VAELoader", "inputs": {"vae_name": cfg["vae"]}},
        "26": {"class_type": "AudioEncoderLoader", "inputs": {
            "audio_encoder_name": cfg["audio_encoder"]}},
        "112": {"class_type": "ModelPatchLoader", "inputs": {
            "name": cfg["model_patch"]}},
        "33": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "lora_name": cfg["lora"], "strength_model": 1, "model": ["13", 0]}},
        "149": {"class_type": "PrimitiveInt", "inputs": {"value": cfg["width"]}},
        "150": {"class_type": "PrimitiveInt", "inputs": {"value": cfg["height"]}},
        "14": {"class_type": "CLIPTextEncode", "inputs": {
            "text": prompt, "clip": ["16", 0]}},
        "17": {"class_type": "ConditioningZeroOut", "inputs": {
            "conditioning": ["14", 0]}},
        "24": {"class_type": "LoadAudio", "inputs": {"audio": audio_1_filename}},
        "90": {"class_type": "LoadAudio", "inputs": {"audio": audio_2_filename}},
        "25": {"class_type": "AudioEncoderEncode", "inputs": {
            "audio_encoder": ["26", 0], "audio": ["24", 0]}},
        "93": {"class_type": "AudioEncoderEncode", "inputs": {
            "audio_encoder": ["26", 0], "audio": ["90", 0]}},
        "113": {"class_type": "AudioConcat", "inputs": {
            "direction": "after", "audio1": ["24", 0], "audio2": ["90", 0]}},
        "32": {"class_type": "LoadImage", "inputs": {"image": image_1_filename}},
        "137": {"class_type": "LoadImage", "inputs": {"image": image_2_filename}},
        "129": {"class_type": "WanInfiniteTalkToVideo", "inputs": {
            "mode": "two_speakers",
            "width": ["149", 0], "height": ["150", 0],
            "length": stage_1_len,
            "motion_frame_count": cfg["motion_frame_count"],
            "audio_scale": cfg["audio_scale"],
            "model": ["33", 0],
            "model_patch": ["112", 0],
            "positive": ["14", 0],
            "negative": ["17", 0],
            "vae": ["29", 0],
            "audio_encoder_output_1": ["25", 0],
            "start_image": ["32", 0],
            "mode.audio_encoder_output_2": ["93", 0],
            "mode.mask_1": ["32", 1],
            "mode.mask_2": ["137", 1],
        }},
        "145:108": {"class_type": "KSamplerSelect", "inputs": {
            "sampler_name": cfg["sampler"]}},
        "145:107": {"class_type": "CFGGuider", "inputs": {
            "cfg": cfg["cfg"], "model": ["129", 0],
            "positive": ["129", 1], "negative": ["129", 2]}},
        "145:19": {"class_type": "BasicScheduler", "inputs": {
            "scheduler": cfg["scheduler"], "steps": cfg["steps"],
            "denoise": cfg["denoise"], "model": ["129", 0]}},
        "145:106": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "145:120": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["145:106", 0], "guider": ["145:107", 0],
            "sampler": ["145:108", 0], "sigmas": ["145:19", 0],
            "latent_image": ["129", 3]}},
        "119": {"class_type": "VAEDecode", "inputs": {
            "samples": ["145:120", 0], "vae": ["29", 0]}},
        "138": {"class_type": "CreateVideo", "inputs": {
            "fps": cfg["fps"], "images": ["119", 0], "audio": ["113", 0]}},
        "162": {"class_type": "SaveVideo", "inputs": {
            "filename_prefix": f"video/{prefix}",
            "format": "auto", "codec": "auto", "video-preview": "",
            "video": ["138", 0]}},
    }

    if not single_stage:
        wf["130"] = {"class_type": "WanInfiniteTalkToVideo", "inputs": {
            "mode": "two_speakers",
            "width": ["149", 0], "height": ["150", 0],
            "length": stage_2_len,
            "motion_frame_count": cfg["motion_frame_count"],
            "audio_scale": cfg["audio_scale"],
            "model": ["33", 0],
            "model_patch": ["112", 0],
            "positive": ["14", 0],
            "negative": ["17", 0],
            "vae": ["29", 0],
            "audio_encoder_output_1": ["25", 0],
            "start_image": ["32", 0],
            "previous_frames": ["119", 0],
            "mode.audio_encoder_output_2": ["93", 0],
            "mode.mask_1": ["32", 1],
            "mode.mask_2": ["137", 1],
        }}
        wf["146:167"] = {"class_type": "KSamplerSelect", "inputs": {
            "sampler_name": cfg["sampler"]}}
        wf["146:168"] = {"class_type": "CFGGuider", "inputs": {
            "cfg": cfg["cfg"], "model": ["130", 0],
            "positive": ["130", 1], "negative": ["130", 2]}}
        wf["146:169"] = {"class_type": "BasicScheduler", "inputs": {
            "scheduler": cfg["scheduler"], "steps": cfg["steps"],
            "denoise": cfg["denoise"], "model": ["130", 0]}}
        wf["146:170"] = {"class_type": "RandomNoise", "inputs": {
            "noise_seed": seed_2}}
        wf["146:171"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["146:170", 0], "guider": ["146:168", 0],
            "sampler": ["146:167", 0], "sigmas": ["146:169", 0],
            "latent_image": ["130", 3]}}
        wf["124"] = {"class_type": "VAEDecode", "inputs": {
            "samples": ["146:171", 0], "vae": ["29", 0]}}
        wf["126"] = {"class_type": "ImageFromBatch", "inputs": {
            "batch_index": ["130", 4],
            "length": 4096,
            "image": ["124", 0]}}
        wf["127"] = {"class_type": "ImageBatch", "inputs": {
            "image1": ["119", 0], "image2": ["126", 0]}}
        wf["140"] = {"class_type": "CreateVideo", "inputs": {
            "fps": cfg["fps"], "images": ["127", 0], "audio": ["113", 0]}}
        wf["162"]["inputs"]["video"] = ["140", 0]

    return wf, stage_1_len, stage_2_len


def _upload_audio(audio_path: Path, name: str):
    """Upload an audio file to ComfyUI's input dir via /upload/image."""
    import sys, httpx
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_podcast_cloud import COMFY
    ext = audio_path.suffix.lower().lstrip(".")
    mime = {"wav": "audio/wav", "flac": "audio/flac",
            "mp3": "audio/mpeg"}.get(ext, "audio/wav")
    with open(audio_path, "rb") as f:
        files = {
            "image": (name, f, mime),
            "type": (None, "input"),
            "overwrite": (None, "1"),
        }
        r = httpx.post(f"{COMFY}/upload/image", files=files, timeout=120)
    r.raise_for_status()
    return r.json()


def render_pair(
    audio_1_path,
    audio_2_path,
    image_1_filename: str,
    image_2_filename: str,
    out_path,
    prompt: str = DEFAULT_PROMPT,
    seed: Optional[int] = None,
    prefix: Optional[str] = None,
    log=print,
    timeout: int = 900,
):
    """End-to-end: upload audios, build workflow, submit, wait, download."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_podcast_cloud import submit, wait_for, download, dur

    audio_1_path = Path(audio_1_path)
    audio_2_path = Path(audio_2_path)
    out_path = Path(out_path)

    if seed is None:
        seed = int.from_bytes(uuid.uuid4().bytes[:4], "big")
    if prefix is None:
        prefix = f"infinitetalk_{out_path.stem}"

    a1_name = f"{prefix}_alex{audio_1_path.suffix}"
    a2_name = f"{prefix}_jordan{audio_2_path.suffix}"
    log(f"  uploading audio 1: {audio_1_path.name} -> {a1_name}")
    _upload_audio(audio_1_path, a1_name)
    log(f"  uploading audio 2: {audio_2_path.name} -> {a2_name}")
    _upload_audio(audio_2_path, a2_name)

    fps = INFINITETALK_DEFAULTS["fps"]
    d1 = dur(audio_1_path)
    d2 = dur(audio_2_path)
    total_dur = d1 + d2
    total_frames = math.ceil(total_dur * fps)
    log(f"  audio: alex={d1:.2f}s + jordan={d2:.2f}s = {total_dur:.2f}s "
        f"-> {total_frames} frames @ {fps}fps")

    wf, s1, s2 = make_infinitetalk_pair_workflow(
        audio_1_filename=a1_name,
        audio_2_filename=a2_name,
        image_1_filename=image_1_filename,
        image_2_filename=image_2_filename,
        total_frames=total_frames,
        prompt=prompt,
        seed=seed,
        prefix=prefix,
    )
    if s2 == 0:
        log(f"  single-stage: {s1} frames")
    else:
        log(f"  two-stage: {s1} + {s2} frames (visible {s1 + s2 - 9})")

    client_id = f"infinitetalk-{uuid.uuid4().hex[:8]}"
    pid = submit(wf, client_id)
    log(f"  submitted: pid={pid}")
    outs, elapsed = wait_for(pid, label=f"infinitetalk_{prefix}",
                             timeout=timeout, poll=4)
    log(f"  rendered in {elapsed:.1f}s")

    save_node = outs.get("162", {})
    videos = (save_node.get("videos") or save_node.get("images") or
              save_node.get("gifs") or [])
    if not videos:
        raise RuntimeError(
            f"no video output from SaveVideo node 162: keys={list(outs.keys())}, "
            f"node162_subkeys={list(save_node.keys())}"
        )
    v = videos[0]
    out_path.parent.mkdir(exist_ok=True, parents=True)
    nb = download(v["filename"], v.get("subfolder", ""),
                  v.get("type", "output"), str(out_path))
    log(f"  saved: {out_path.name} ({nb//1024} KB)")
    return out_path


if __name__ == "__main__":
    capr_tts = Path(r"B:\amd-hackathon-bill-analyzer\eval\capr26-cloud\tts")
    out = Path(r"B:\hackathon-build\_infinitetalk_smoke\pair_01.mp4")
    render_pair(
        audio_1_path=capr_tts / "scene-01.flac",
        audio_2_path=capr_tts / "scene-02.flac",
        image_1_filename="podcast_pair_alex_mask.png",
        image_2_filename="podcast_pair_jordan_mask.png",
        out_path=out,
        seed=42,
        prefix="smoke_pair_01",
    )
    print(f"\nSMOKE OK: {out}")
