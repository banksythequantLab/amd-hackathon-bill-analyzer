"""Cloud podcast pipeline. All compute on MI300X. Idempotent + resumable."""
from __future__ import annotations
import argparse, base64, copy, json, sys, time
from pathlib import Path
import httpx

# Repo root, derived from this file's location so it works on every OS:
# scripts/make_podcast_cloud.py -> parent is scripts/ -> parent is repo root.
# (Was a hardcoded Windows path B:\amd-hackathon-bill-analyzer for a long
# time; that quietly broke canonical-report lookups on the HF Space because
# Path("B:\\...") resolves to a junk relative path on Linux.)
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.agents.podcast_script_writer import PodcastScriptWriter
from src.agents.slide_prompt_generator import SlidePromptGenerator
from src.agents.wan_motion_prompt_generator import WanMotionPromptGenerator
from src.agents.slide_critic import SlideCritic
from src.agents.youtube_metadata_generator import YouTubeMetadataGenerator

# 3090 fork: TTS now runs through the local FreeClone server (VoxCPM2 +
# Whisper) on :8300, replacing the AMD cluster Qwen-TTS ComfyUI nodes.
# scripts/freeclone_tts.py is the thin wrapper. See TODO #5 commit.
from scripts.freeclone_tts import render_podcast, ScriptLine, FreeCloneError, healthcheck as freeclone_health

# 3090 FORK: ComfyUI runs locally on Johnson. The orchestrator
# kill+restarts ComfyUI between Qwen-Image / Wan / InfiniteTalk stages
# (subprocess-per-stage pipeline) because the 3090 only has 24 GB and
# ComfyUI tends to retain ~21 GB VRAM after a render. AMD canonical
# baseline pointed this at the cluster droplet on port 8188 (see
# docs/day3-runbook.md for the historical address).
COMFY = "http://127.0.0.1:8188"
VOICE_MAP = {'Alex': 'Ryan', 'Jordan': 'Ono_anna'}
# 3090 fork: speaker -> FreeClone default-voice id. Matches AMD-canonical
# host genders: Alex/Ryan -> echo (deep male American); Jordan/Ono_anna
# -> nova (bright female American). Override via stage_tts(..., freeclone_voices=...).
FREECLONE_VOICE_MAP = {'Alex': 'echo', 'Jordan': 'nova'}
BRAND_DIR = REPO / 'brand'  # Dead Air intro/outro/closeout cards live here

# ---- COMFY HELPERS ----
def submit(wf, client_id):
    r = httpx.post(f'{COMFY}/prompt', json={'prompt': wf, 'client_id': client_id}, timeout=15)
    r.raise_for_status()
    return r.json()['prompt_id']

def wait_for(pid, label='', timeout=600, poll=4):
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = httpx.get(f'{COMFY}/history/{pid}', timeout=10).json()
        if h.get(pid):
            st = h[pid].get('status', {})
            if st.get('completed'):
                elapsed = time.time() - t0
                return h[pid].get('outputs', {}), elapsed
            else:
                raise RuntimeError(f'{label} did not complete: {st}')
        time.sleep(poll)
    raise TimeoutError(f'{label} timed out after {timeout}s')

def download(filename, subfolder, kind, out_path):
    url = f'{COMFY}/api/view?filename={filename}&subfolder={subfolder}&type={kind}'
    r = httpx.get(url, timeout=120)
    r.raise_for_status()
    Path(out_path).write_bytes(r.content)
    return len(r.content)

def upload_image(path, name=None):
    name = name or Path(path).name
    with open(path, 'rb') as f:
        files = {'image': (name, f, 'image/png'), 'type': (None, 'input'), 'overwrite': (None, '1')}
        r = httpx.post(f'{COMFY}/upload/image', files=files, timeout=60)
    r.raise_for_status()
    return r.json()

# ---- WORKFLOW BUILDERS ----
def qwen_image_workflow(positive, negative, seed, prefix, w=1280, h=720):
    return {
        '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'qwen_image_2512_fp8_e4m3fn.safetensors', 'weight_dtype': 'default'}},
        '2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen_2.5_vl_7b_fp8_scaled.safetensors', 'type': 'qwen_image', 'device': 'default'}},
        '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'qwen_image_vae.safetensors'}},
        '4': {'class_type': 'LoraLoaderModelOnly', 'inputs': {'model': ['1', 0], 'lora_name': 'Qwen-Image-Lightning-4steps-V2.0.safetensors', 'strength_model': 1.0}},
        '5': {'class_type': 'ModelSamplingAuraFlow', 'inputs': {'model': ['4', 0], 'shift': 3.0}},
        '10': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 0], 'text': positive}},
        '11': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 0], 'text': negative}},
        '20': {'class_type': 'EmptySD3LatentImage', 'inputs': {'width': w, 'height': h, 'batch_size': 1}},
        '30': {'class_type': 'KSampler', 'inputs': {'model': ['5', 0], 'positive': ['10', 0], 'negative': ['11', 0], 'latent_image': ['20', 0], 'seed': seed, 'steps': 4, 'cfg': 1.0, 'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}},
        '40': {'class_type': 'VAEDecode', 'inputs': {'samples': ['30', 0], 'vae': ['3', 0]}},
        '50': {'class_type': 'SaveImage', 'inputs': {'images': ['40', 0], 'filename_prefix': prefix}}
    }

def wan_i2v_workflow(image_filename, motion_prompt, seed, prefix):
    base = json.load(open(REPO / 'comfy/workflows/wan22-i2v-simple-api.json'))
    base['10']['inputs']['text'] = motion_prompt
    base['20']['inputs']['image'] = image_filename
    base['40']['inputs']['noise_seed'] = seed
    base['70']['inputs']['filename_prefix'] = prefix
    return base

def tts_workflow(text, speaker, prefix):
    return {
        '1': {'class_type': 'FB_Qwen3TTSCustomVoice', 'inputs': {
            'text': text, 'speaker': speaker, 'model_choice': '1.7B', 'device': 'cuda',
            'precision': 'bf16', 'language': 'Auto', 'attention': 'auto',
            'unload_model_after_generate': False}},
        '2': {'class_type': 'SaveAudio', 'inputs': {'audio': ['1', 0], 'filename_prefix': prefix}}
    }
# ---- STAGE FUNCTIONS ----
def build_report_text(rep, log=print):
    a = rep['agents']
    parts = [f'BILL: {rep.get("bill_label")} (short: {rep.get("bill_short")})',
             f'NOTE: {rep.get("bill_note", "")}',
             f'Section: {rep.get("title_marker", "")}', '']
    sm = a.get('summarizer', {}).get('output', {})
    if sm:
        parts.append('=== Plain-English Summary ===')
        parts.append(json.dumps(sm, indent=2)[:6000])
    pf = a.get('pork_finder', {}).get('output', {})
    if pf:
        parts.append('\n=== Pork & Earmarks ===')
        parts.append(json.dumps(pf, indent=2)[:3000])
    cs = a.get('conflict_spotter', {}).get('output', {})
    if cs:
        parts.append('\n=== Conflicts ===')
        parts.append(json.dumps(cs, indent=2)[:3000])
    ux = a.get('usc_cross_ref', {}).get('output', {})
    if ux:
        parts.append('\n=== USC Citations ===')
        parts.append(json.dumps(ux, indent=2)[:3000])
    return '\n'.join(parts)

def stage_text_agents(report, eval_dir, headline, log=print, creative_direction=None):
    bill_short = report['bill_short']
    rep_text = build_report_text(report)
    # If the user provided extra creative direction, prepend it to the report
    # context so the script writer sees it before the bill analysis. This is
    # a low-risk way to inject custom prompt instructions without modifying
    # the agent's prompt schema.
    if creative_direction:
        rep_text = (
            "=== ADDITIONAL CREATIVE DIRECTION FROM USER ===\n"
            f"{creative_direction}\n"
            "=== END DIRECTION ===\n\n"
        ) + rep_text
    sf = eval_dir / 'script.json'
    if sf.exists():
        log(f'  [skip] script.json exists')
        script = json.load(open(sf))
    else:
        log('  generating dialog script via spine...')
        psw = PodcastScriptWriter()
        t0 = time.time()
        res = psw.run(rep_text, 'ch01', bill_short=bill_short, headline=headline)
        log(f'    done in {time.time()-t0:.1f}s | errors={res.errors}')
        if res.errors: raise RuntimeError(res.errors)
        script = res.output
        json.dump(script, open(sf, 'w'), indent=2)
    sl = eval_dir / 'slides.json'
    if sl.exists():
        log(f'  [skip] slides.json exists')
        slides_obj = json.load(open(sl))
    else:
        log('  generating 19 slide prompts via spine...')
        spg = SlidePromptGenerator()
        t0 = time.time()
        res = spg.run(json.dumps(script, indent=2), 'ch01', bill_short=bill_short)
        log(f'    done in {time.time()-t0:.1f}s | errors={res.errors}')
        if res.errors: raise RuntimeError(res.errors)
        slides_obj = res.output
        json.dump(slides_obj, open(sl, 'w'), indent=2)
    mo = eval_dir / 'motions.json'
    if mo.exists():
        log(f'  [skip] motions.json exists')
        motions_obj = json.load(open(mo))
    else:
        log('  generating 19 motion prompts via spine...')
        wmp = WanMotionPromptGenerator()
        t0 = time.time()
        res = wmp.run(json.dumps(slides_obj, indent=2), 'ch01', bill_short=bill_short)
        log(f'    done in {time.time()-t0:.1f}s | errors={res.errors}')
        if res.errors: raise RuntimeError(res.errors)
        motions_obj = res.output
        json.dump(motions_obj, open(mo, 'w'), indent=2)
    return script, slides_obj, motions_obj

def stage_slides(slides_obj, eval_dir, log=print):
    """Generate 19 slides via Qwen-Image. Critique each. Retry up to 3 times on fail."""
    slides_dir = eval_dir / 'slides'
    slides_dir.mkdir(exist_ok=True, parents=True)
    critiques = []
    critic = SlideCritic()
    for s in slides_obj['slides']:
        scene = s['scene']
        out_path = slides_dir / f'scene-{scene:02d}.png'
        crit_path = slides_dir / f'scene-{scene:02d}-crit.json'
        if out_path.exists() and crit_path.exists():
            crit = json.load(open(crit_path))
            if crit.get('pass_fail') == 'pass':
                log(f'  [{scene:02d}] cached PASS ({out_path.stat().st_size//1024}KB)')
                critiques.append(crit)
                continue
        positive = s['positive_prompt']
        negative = s.get('negative_prompt', 'blurry low quality watermark')
        expected = s['headline_text']
        last_crit = None
        # Up to 15 retry attempts: SlideCritic dual-call (OCR + judgment) is
        # strict about typos and visual quality. With 4-step Lightning the
        # tail of the seed distribution still produces ~10-20% slides that
        # need a re-roll. 15 attempts gives essentially 100% pass probability
        # and rare failure cases (a tricky long headline, a stubborn typo)
        # finally clear instead of pinning the slide on a bad seed for the
        # whole render.
        for attempt in range(15):
            seed = scene * 1000 + attempt * 17 + 42
            prefix = f'border-cloud/scene-{scene:02d}-att{attempt}'
            wf = qwen_image_workflow(positive, negative, seed, prefix)
            t0 = time.time()
            pid = submit(wf, f'qimg-{scene}-{attempt}')
            outs, elapsed = wait_for(pid, f'qimg-{scene}-att{attempt}', timeout=300)
            imgs = outs.get('50', {}).get('images', [])
            if not imgs:
                log(f'  [{scene:02d}] att{attempt} no images!')
                continue
            sz = download(imgs[0]['filename'], imgs[0]['subfolder'], 'output', out_path)
            crit = critic.critique(out_path, expected)
            crit_d = crit.model_dump()
            crit_d['_attempt'] = attempt
            crit_d['_seed'] = seed
            crit_d['_elapsed'] = round(elapsed, 1)
            last_crit = crit_d
            log(f'  [{scene:02d}] att{attempt} {elapsed:.1f}s -> {sz//1024}KB | critic={crit.pass_fail} (conf={crit.confidence:.2f}) | reasons={crit.failure_reasons}')
            if crit.pass_fail == 'pass':
                break
        json.dump(last_crit, open(crit_path, 'w'), indent=2)
        critiques.append(last_crit)
    return critiques

def stage_wan(slides_obj, motions_obj, eval_dir, log=print):
    """Animate 19 slides with Wan 2.2 i2v. Each gets pre + post (38 jobs total)."""
    slides_dir = eval_dir / 'slides'
    wan_dir = eval_dir / 'wan'
    wan_dir.mkdir(exist_ok=True, parents=True)
    motions_by_scene = {m['scene']: m for m in motions_obj['motions']}
    for s in slides_obj['slides']:
        scene = s['scene']
        slide_png = slides_dir / f'scene-{scene:02d}.png'
        if not slide_png.exists():
            log(f'  [{scene:02d}] missing slide PNG; skipping wan')
            continue
        m = motions_by_scene.get(scene, {})
        upload_image(slide_png, name=f'border-scene-{scene:02d}.png')
        for half, mp in [('pre', m.get('pre_motion_prompt', 'subtle camera push-in, gentle parallax')),
                          ('post', m.get('post_motion_prompt', 'soft light ray sweep, vignette pulse'))]:
            out_path = wan_dir / f'scene-{scene:02d}-{half}.mp4'
            if out_path.exists() and out_path.stat().st_size > 100_000:
                log(f'  [{scene:02d}-{half}] cached ({out_path.stat().st_size//1024}KB)')
                continue
            seed = scene * 100 + (1 if half == 'post' else 0)
            prefix = f'border-cloud/scene-{scene:02d}-{half}'
            wf = wan_i2v_workflow(f'border-scene-{scene:02d}.png', mp, seed, prefix)
            pid = submit(wf, f'wan-{scene}-{half}')
            outs, elapsed = wait_for(pid, f'wan-{scene}-{half}', timeout=240)
            vids = outs.get('70', {}).get('images', [])
            if not vids:
                log(f'  [{scene:02d}-{half}] no video! outs={outs}')
                continue
            sz = download(vids[0]['filename'], vids[0]['subfolder'], 'output', out_path)
            log(f'  [{scene:02d}-{half}] {elapsed:.1f}s -> {sz//1024}KB')

def stage_tts(script, eval_dir, log=print, freeclone_voices=None, freeclone_url="http://127.0.0.1:8300"):
    """Render one FLAC per dialog line via the local FreeClone server.

    3090 fork: replaces the AMD cluster's Qwen-TTS ComfyUI workflow with
    a sequence of small FreeClone /api/podcast calls (one per dialog
    line). Each call returns a WAV which we transcode to FLAC at
    scene-NN.flac for downstream compatibility (compose/avatar stages
    read these via ffmpeg, which is format-agnostic; the .flac extension
    just preserves the existing cached-output naming).

    Args:
        script: PodcastScriptWriter output. Must have a 'dialog' list
            of {'scene': int, 'speaker': str, 'line': str} entries.
        eval_dir: parent dir; files written to eval_dir / 'tts' /
            'scene-NN.flac'.
        log: logger callable (defaults to print).
        freeclone_voices: optional speaker -> FreeClone-voice-id map.
            Defaults to FREECLONE_VOICE_MAP (Alex=echo, Jordan=nova).
            Pass {} to use FreeClone's per-speaker fallback voices.
        freeclone_url: FreeClone base URL. 127.0.0.1:8300 triggers the
            studio-tier bypass in server.py (lifts the free-tier 4-line
            cap so 19-30 line scripts work).

    Cached files (>10 KB) are skipped on subsequent runs.
    """
    import subprocess  # subprocess is also imported at module level below the function defs

    tts_dir = eval_dir / 'tts'
    tts_dir.mkdir(exist_ok=True, parents=True)
    voices = freeclone_voices if freeclone_voices is not None else FREECLONE_VOICE_MAP

    # Pre-flight: fail fast if FreeClone isn't up, before we spend any
    # time on lines. healthcheck raises httpx.RequestError on connection
    # refused -- let that propagate so the caller sees "FreeClone down".
    try:
        h = freeclone_health(freeclone_url)
        log(f'  freeclone /health: status={h.get("status")} gpu={h.get("gpu")} '
            f'whisper={h.get("whisperLoaded")} voxcpm={h.get("voxcpmLoaded")}')
    except Exception as e:
        log(f'  [WARN] freeclone /health failed: {type(e).__name__}: {e}')
        log(f'         Make sure FreeClone is running: B:\\freeclone-backend\\START_BFORK.bat')
        raise

    for line in script['dialog']:
        scene = line['scene']
        speaker = line['speaker']
        text = line['line']
        out_path = tts_dir / f'scene-{scene:02d}.flac'
        if out_path.exists() and out_path.stat().st_size > 10_000:
            log(f'  [{scene:02d}] cached tts ({out_path.stat().st_size//1024}KB)')
            continue
        voice_id = voices.get(speaker, 'echo')
        # FreeClone speaker key is a string id ('1','2',...). We always
        # use '1' for the single-line call and pass the resolved voice
        # via default_voice_1. (FreeClone treats each request
        # independently, so the speaker id doesn't need to match the
        # logical Alex/Jordan distinction; the VOICE id is what carries
        # the timbre.)
        one_line_script = [ScriptLine('1', text, lang='en')]
        wav_tmp = tts_dir / f'_scene-{scene:02d}.wav'
        t0 = time.time()
        try:
            result = render_podcast(
                one_line_script,
                wav_tmp,
                voices={'1': voice_id},
                freeclone_url=freeclone_url,
            )
        except FreeCloneError as e:
            log(f'  [{scene:02d}] FreeClone error (status {e.status_code}): {e}')
            continue
        elapsed = time.time() - t0
        # Transcode WAV -> FLAC for downstream compat (.flac is what the
        # compose/avatar stages expect by name).
        r = subprocess.run(
            [FFMPEG, '-y', '-i', str(wav_tmp), '-c:a', 'flac', str(out_path)],
            capture_output=True,
        )
        if r.returncode != 0:
            log(f'  [{scene:02d}] ffmpeg flac transcode failed: {r.stderr[:200]!r}')
            continue
        wav_tmp.unlink(missing_ok=True)
        sz = out_path.stat().st_size
        log(f'  [{scene:02d}] {speaker}->{voice_id}: {elapsed:.1f}s -> {sz//1024}KB | "{text[:60]}..."')

# ---- COMPOSE ----
import subprocess
import shutil

# ffmpeg / ffprobe location is OS-dependent. The HF Space container has
# them on PATH (apt-installed); the local Windows dev box has them inside
# the WinGet package directory. Resolve in this order:
#   1) Whatever is on PATH (works on Linux container + macOS + any Windows
#      box where ffmpeg has been added to PATH).
#   2) Known WinGet path on the dev workstation.
# If neither is found we still set the strings so the import succeeds; the
# subprocess.run() call will surface the real error at compose time instead
# of crashing the import.
# 3090 fork: search a list of plausible WinGet FFmpeg dirs because winget
# installs different package versions over time (we have 7.1.1 + 8.1.1
# side by side after Day 1 installed Gyan.FFmpeg.Shared 8.1.1 for the
# FreeClone torchcodec/avcodec DLL dependency). Order: prefer system
# PATH, then newest Gyan.FFmpeg install, then Gyan.FFmpeg.Shared, then
# the historical 7.1.1 path. The WinGet alias dir is also tried; it
# contains a launcher that resolves to whichever real install is active.
def _find_ffmpeg_bin(name):
    """Locate ffmpeg.exe / ffprobe.exe by trying PATH then known WinGet
    install dirs. Returns the first existing path, or just the bare
    command name (so subprocess.run will surface the FileNotFound)."""
    via_path = shutil.which(name)
    if via_path:
        return via_path
    winget_packages = Path(r"C:\Users\solti\AppData\Local\Microsoft\WinGet\Packages")
    winget_links = Path(r"C:\Users\solti\AppData\Local\Microsoft\WinGet\Links")
    candidates = []
    # Newest first: any Gyan.FFmpeg* package directory with a build subdir.
    for pkg_glob in ("Gyan.FFmpeg.Shared_*", "Gyan.FFmpeg_*"):
        for pkg_dir in sorted(winget_packages.glob(pkg_glob), reverse=True):
            for build_dir in sorted(pkg_dir.glob("ffmpeg-*"), reverse=True):
                candidates.append(build_dir / "bin" / f"{name}.exe")
    candidates.append(winget_links / f"{name}.exe")
    for c in candidates:
        if c.exists():
            return str(c)
    return name  # fall through to bare name; subprocess will FileNotFound clearly.

FFMPEG  = _find_ffmpeg_bin('ffmpeg')
FFPROBE = _find_ffmpeg_bin('ffprobe')

def dur(path):
    r = subprocess.run([FFPROBE, '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', str(path)], capture_output=True, text=True)
    return float(r.stdout.strip()) if r.stdout.strip() else 0.0


def _make_card_clip(png_path, out_path, duration_sec, log=print, w=832, h=480, fps=25):
    """Build a still-frame mp4 clip from a brand PNG card.

    Output format is locked to match scene clips EXACTLY so the final
    concat with `-c copy` doesn't fail. Reference (probed from existing
    scene clips):
      video: libx264 High@L30 yuv420p {w}x{h} @ {fps}fps
      audio: AAC LC mono 24kHz (silent)

    The card image is scaled with `force_original_aspect_ratio=decrease`
    plus a centered black pad so wider intro/outro cards (1920x1080) get
    a thin 6px letterbox top/bottom and the square closeout card
    (1024x1024) gets pillarboxing (176px black bars left/right).

    Caches: if `out_path` already exists with reasonable size, skip rebuild.
    """
    png_path = Path(png_path)
    out_path = Path(out_path)

    # Day 7.20: Lipsync overlay short-circuit. If a pre-rendered lipsync
    # intro/outro overlay exists at brand/lipsync/, use it instead of a
    # still-frame card. Audio is downmixed stereo->mono 24kHz so `-c copy`
    # final concat works (overlay mp4s are AAC stereo; cards/scenes are mono).
    # Falls through to still-card logic if the overlay file is absent.
    _overlay_map = {'intro.mp4': 'intro_lipsync_overlay.mp4',
                    'outro.mp4': 'outro_lipsync_overlay.mp4'}
    _overlay_name = _overlay_map.get(out_path.name)
    if _overlay_name:
        _overlay_src = BRAND_DIR / 'lipsync' / _overlay_name
        if _overlay_src.exists():
            if out_path.exists() and out_path.stat().st_size > 10_000:
                log(f'  card cached (lipsync): {out_path.name} ({out_path.stat().st_size//1024} KB)')
                return out_path
            _cmd = [FFMPEG, '-y', '-i', str(_overlay_src),
                    '-c:v', 'copy',
                    '-c:a', 'aac', '-b:a', '96k', '-ar', '24000', '-ac', '1',
                    str(out_path)]
            _r = subprocess.run(_cmd, capture_output=True, text=True)
            if _r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 10_000:
                log(f'  card built (lipsync): {out_path.name} ({dur(out_path):.1f}s, {out_path.stat().st_size//1024} KB)')
                return out_path
            log(f'  card lipsync FAIL ({_overlay_src.name}); falling through to still-card: {_r.stderr[-200:]}')

    if not png_path.exists():
        log(f'  card SKIP: source PNG not found at {png_path.name}')
        return None
    if out_path.exists() and out_path.stat().st_size > 10_000:
        log(f'  card cached: {out_path.name} ({out_path.stat().st_size//1024} KB)')
        return out_path

    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,"
        f"format=yuv420p,fps={fps}"
    )
    # -loop 1 + -t makes ffmpeg generate `duration_sec * fps` frames from the still image.
    # -f lavfi anullsrc generates a silent MONO 24kHz track to match the
    #  AAC LC mono 24kHz audio in the existing dialog scene clips.
    # Pinning -profile:v high -level 3.0 so `-c copy` concat sees identical
    #  H.264 params across all clips.
    cmd = [
        FFMPEG, '-y',
        '-loop', '1', '-t', f'{duration_sec:.3f}', '-i', str(png_path),
        '-f', 'lavfi', '-t', f'{duration_sec:.3f}',
        '-i', 'anullsrc=channel_layout=mono:sample_rate=24000',
        '-vf', vf,
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
        '-profile:v', 'high', '-level', '3.0',
        '-pix_fmt', 'yuv420p', '-r', str(fps),
        '-c:a', 'aac', '-b:a', '96k', '-ar', '24000', '-ac', '1',
        '-shortest', str(out_path)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 10_000:
        log(f'  card built: {out_path.name} ({duration_sec:.1f}s, {out_path.stat().st_size//1024} KB)')
        return out_path
    else:
        log(f'  card FAIL ({png_path.name}): {r.stderr[-200:]}')
        return None


def stage_compose(eval_dir, bill_short, n_scenes=19, log=print):
    wan_dir = eval_dir / 'wan'
    tts_dir = eval_dir / 'tts'
    out_dir = eval_dir / 'compose'
    out_dir.mkdir(exist_ok=True, parents=True)
    scenes = []
    for n in range(1, n_scenes + 1):
        pre = wan_dir / f'scene-{n:02d}-pre.mp4'
        post = wan_dir / f'scene-{n:02d}-post.mp4'
        aud = tts_dir / f'scene-{n:02d}.flac'
        out = out_dir / f'scene-{n:02d}.mp4'
        if not pre.exists() or not post.exists() or not aud.exists():
            log(f'  [{n:02d}] missing assets; skipping')
            continue
        if out.exists() and out.stat().st_size > 50_000:
            log(f'  [{n:02d}] cached compose')
            scenes.append(out)
            continue
        ad = dur(aud); vd = dur(pre) + dur(post)
        stretch = ad / vd if vd > 0 else 1.0
        # Add 200ms silence padding to audio for natural pacing between lines
        fc = '[0:v][1:v]concat=n=2:v=1:a=0[vc];[vc]setpts=' + f'{stretch:.4f}' + '*PTS,fps=25[v];[2:a]apad=pad_dur=0.2[a]'
        cmd = [FFMPEG, '-y', '-i', str(pre), '-i', str(post), '-i', str(aud),
               '-filter_complex', fc, '-map', '[v]', '-map', '[a]',
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
               '-shortest', str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            log(f'  [{n:02d}] composed (stretch={stretch:.2f}x audio={ad:.1f}s) -> {out.stat().st_size//1024}KB')
            scenes.append(out)
        else:
            log(f'  [{n:02d}] FAIL: {r.stderr[-200:]}')
    if not scenes:
        log('NO SCENES; aborting final concat')
        return None

    # ---- BRAND CARDS: intro / outro / closeout ----
    # Cards are pre-rendered to mp4s with the SAME codec params as scene
    # clips so the final concat with `-c copy` works without re-encoding.
    cards_dir = out_dir / '_cards'
    cards_dir.mkdir(exist_ok=True, parents=True)
    log('')
    log('STAGE 5b: Brand cards (intro / outro / closeout)')

    intro_clip = _make_card_clip(
        BRAND_DIR / 'deadair_intro-1080.png',
        cards_dir / 'intro.mp4',
        duration_sec=4.0,
        log=log,
    )
    outro_clip = _make_card_clip(
        BRAND_DIR / 'deadair_outro-1080.png',
        cards_dir / 'outro.mp4',
        duration_sec=4.0,
        log=log,
    )
    closeout_clip = _make_card_clip(
        BRAND_DIR / 'deadair_closeout-1024.png',
        cards_dir / 'closeout.mp4',
        duration_sec=2.0,
        log=log,
    )

    # Build final clip list: intro + scenes + outro + closeout.
    # Any missing card is logged and silently dropped (graceful fallback)
    # so a missing brand asset doesn't kill the whole render.
    final_clips = []
    if intro_clip:
        final_clips.append(intro_clip)
    final_clips.extend(scenes)
    if outro_clip:
        final_clips.append(outro_clip)
    if closeout_clip:
        final_clips.append(closeout_clip)
    log(f'  final clip list: {len(final_clips)} clips ({len(scenes)} scenes + branding)')

    list_file = out_dir / '_master.txt'
    list_file.write_text('\n'.join(f"file '{s}'" for s in final_clips) + '\n')
    final = eval_dir / f'final-{bill_short}-cloud-podcast.mp4'
    r = subprocess.run([FFMPEG, '-y', '-f', 'concat', '-safe', '0', '-i', str(list_file),
                        '-c', 'copy', str(final)], capture_output=True, text=True)
    if r.returncode == 0:
        log(f'\n*** MASTER: {final}')
        log(f'    size = {final.stat().st_size/1024/1024:.1f} MB')
        log(f'    dur  = {dur(final):.1f}s ({dur(final)/60:.2f} min)')
        return final
    else:
        log(f'FINAL CONCAT FAIL: {r.stderr[-400:]}')
        return None


def stage_avatar_render(script, eval_dir, log=print, prompt=None,
                         image_1='podcast_pair_jordan_mask.png',
                         image_2='podcast_pair_alex_mask.png'):
    """Render each consecutive pair of dialog lines via InfiniteTalk on the cloud.

    IMPORTANT - mask assignment (verified on capr26 Day 7.13):
      audio_1 = Alex's line  -> uses VOICE_MAP['Alex'] = 'Ryan' (MALE voice)
      audio_2 = Jordan's line -> uses VOICE_MAP['Jordan'] = 'Ono_anna' (FEMALE voice)
      image_1 = jordan_mask  -> covers RIGHT half of ref (visible MALE character)
      image_2 = alex_mask    -> covers LEFT half of ref (visible FEMALE character)
    Result: voices match faces. The mask filenames are intentionally
    "swapped" relative to the speaker name they're attached to, because
    the original brand/podcast_pair_ref.png placed the female on the left
    while Alex (left-named) speaks with a male voice.

    Pair grouping: scene-01 + scene-02 -> pair_01.mp4, scene-03 + scene-04 ->
    pair_02.mp4, etc. If the dialog has an odd number of lines, the last line
    is dropped (single-speaker mode could be added later if needed).

    Pre-conditions:
      eval_dir/tts/scene-NN.flac     for each dialog line (1-indexed).
      Mask + ref PNGs already uploaded to ComfyUI input dir (via Day 7.10
      brand asset commit, persisted server-side until ComfyUI restart).

    Resumable: if eval_dir/infinitetalk/pair_NN.mp4 exists with reasonable
    size, that pair is skipped. Otherwise submitted to the cloud queue.

    Concurrency: all pairs are submitted up-front, then a poll loop drains
    the queue. ComfyUI processes them serially (one at a time), but submitting
    in a batch means we don't sit blocked between renders.
    """
    import sys, math, uuid
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from infinitetalk_pipeline import (
        make_infinitetalk_pair_workflow, _upload_audio, DEFAULT_PROMPT,
    )
    pair_prompt = prompt or DEFAULT_PROMPT

    tts_dir = eval_dir / 'tts'
    out_dir = eval_dir / 'infinitetalk'
    out_dir.mkdir(exist_ok=True, parents=True)

    dialog = script.get('dialog', [])
    n_pairs = len(dialog) // 2
    if not n_pairs:
        log(f'STAGE 4-AVATAR: NO PAIRS (dialog has {len(dialog)} lines)')
        return
    log(f'STAGE 4-AVATAR: {len(dialog)} lines -> {n_pairs} pairs')

    submissions = []
    for pair_idx in range(1, n_pairs + 1):
        scene_a = (pair_idx - 1) * 2 + 1
        scene_b = scene_a + 1
        out_path = out_dir / f'pair_{pair_idx:02d}.mp4'
        if out_path.exists() and out_path.stat().st_size > 50_000:
            log(f'  pair_{pair_idx:02d}: cached')
            continue
        a1 = tts_dir / f'scene-{scene_a:02d}.flac'
        a2 = tts_dir / f'scene-{scene_b:02d}.flac'
        if not a1.exists() or not a2.exists():
            log(f'  pair_{pair_idx:02d}: MISSING ({a1.name} or {a2.name})')
            continue
        prefix = f'{eval_dir.name}_p{pair_idx:02d}'
        a1n = f'{prefix}_a.flac'; a2n = f'{prefix}_b.flac'
        _upload_audio(a1, a1n); _upload_audio(a2, a2n)
        tf = math.ceil((dur(a1) + dur(a2)) * 25)
        wf, s1, s2 = make_infinitetalk_pair_workflow(
            audio_1_filename=a1n, audio_2_filename=a2n,
            image_1_filename=image_1, image_2_filename=image_2,
            total_frames=tf, prompt=pair_prompt,
            seed=42 + pair_idx, prefix=prefix,
        )
        pid = submit(wf, f'avatar-{uuid.uuid4().hex[:8]}')
        submissions.append({'idx': pair_idx, 'pid': pid, 'out': out_path,
                            'frames': tf, 's1': s1, 's2': s2})
        log(f'  pair_{pair_idx:02d}: queued ({tf} fr, {s1}+{s2}) pid={pid[:8]}')

    if not submissions:
        log('  all pairs cached; nothing to render')
        return

    log(f'  waiting for {len(submissions)} renders...')
    t_start = time.time()
    pending = {s['idx']: s for s in submissions}
    while pending:
        time.sleep(20)
        finished = []
        for idx, s in pending.items():
            try:
                h = httpx.get(f'{COMFY}/history/{s["pid"]}', timeout=10).json()
            except Exception:
                continue
            if s['pid'] not in h:
                continue
            st = h[s['pid']]['status']
            if st.get('status_str') == 'error':
                log(f'  pair_{idx:02d}: ERROR')
                finished.append(idx)
                continue
            if not st.get('completed'):
                continue
            outs = h[s['pid']].get('outputs', {}).get('162', {})
            vids = outs.get('videos') or outs.get('images') or []
            if vids:
                v = vids[0]
                download(v['filename'], v.get('subfolder', ''),
                         v.get('type', 'output'), str(s['out']))
                log(f'  pair_{idx:02d}: done [{int(time.time()-t_start)}s elapsed]')
            else:
                log(f'  pair_{idx:02d}: NO VIDEO output keys={list(h[s["pid"]].get("outputs",{}).keys())}')
            finished.append(idx)
        for idx in finished:
            pending.pop(idx)

    log(f'STAGE 4-AVATAR: done in {time.time()-t_start:.0f}s')


def _downmix_pair_to_mono(src_mp4, out_mp4, log=print):
    """Re-encode an InfiniteTalk pair clip's audio from stereo->mono 24kHz.

    Video is copied without re-encode (-c:v copy), so this is essentially free.
    The downmix is needed because InfiniteTalk's CreateVideo node emits AAC
    LC stereo 24kHz, but the brand cards (and existing slide-mode scenes)
    use AAC LC mono 24kHz. Mismatched channel layouts break `-c copy` concat.
    """
    src_mp4 = Path(src_mp4); out_mp4 = Path(out_mp4)
    if out_mp4.exists() and out_mp4.stat().st_size > 10_000:
        log(f'  pair mono cached: {out_mp4.name}')
        return out_mp4
    cmd = [FFMPEG, '-y', '-i', str(src_mp4),
           '-c:v', 'copy',
           '-c:a', 'aac', '-b:a', '96k', '-ar', '24000', '-ac', '1',
           str(out_mp4)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and out_mp4.exists() and out_mp4.stat().st_size > 10_000:
        log(f'  pair mono: {out_mp4.name} ({out_mp4.stat().st_size//1024} KB)')
        return out_mp4
    log(f'  pair mono FAIL ({src_mp4.name}): {r.stderr[-200:]}')
    return None


def stage_avatar_compose(eval_dir, bill_short, log=print):
    """Concat InfiniteTalk pair clips + Dead Air brand cards into avatar master.

    Reads pair_NN.mp4 files from <eval_dir>/infinitetalk/ (the dir created by
    the avatar-mode render loop). Downmixes each pair's audio to mono 24kHz
    so concat with the existing card format works. Bookends with the same
    intro/outro/closeout cards used by slides-mode stage_compose.

    Output: <eval_dir>/final-<bill_short>-cloud-avatar-podcast.mp4
    (distinct from slides-mode final-<bill_short>-cloud-podcast.mp4 so both
    can coexist for A/B comparison)
    """
    pairs_dir = eval_dir / 'infinitetalk'
    if not pairs_dir.exists():
        log(f'NO INFINITETALK DIR: {pairs_dir}')
        return None
    pair_clips = sorted(pairs_dir.glob('pair_*.mp4'))
    if not pair_clips:
        log(f'NO PAIR CLIPS in {pairs_dir}')
        return None
    log(f'STAGE 5-AVATAR: found {len(pair_clips)} InfiniteTalk pair clips')

    out_dir = pairs_dir / 'compose'
    out_dir.mkdir(exist_ok=True)
    mono_clips = []
    for src in pair_clips:
        out = out_dir / f'{src.stem}_mono.mp4'
        m = _downmix_pair_to_mono(src, out, log=log)
        if m:
            mono_clips.append(m)
    if not mono_clips:
        log('NO MONO CLIPS produced; aborting')
        return None

    cards_dir = out_dir / '_cards'
    cards_dir.mkdir(exist_ok=True, parents=True)
    log('STAGE 5b: Brand cards (intro / outro / closeout)')
    intro_clip = _make_card_clip(BRAND_DIR / 'deadair_intro-1080.png',
                                  cards_dir / 'intro.mp4', duration_sec=4.0, log=log)
    outro_clip = _make_card_clip(BRAND_DIR / 'deadair_outro-1080.png',
                                  cards_dir / 'outro.mp4', duration_sec=4.0, log=log)
    closeout_clip = _make_card_clip(BRAND_DIR / 'deadair_closeout-1024.png',
                                     cards_dir / 'closeout.mp4', duration_sec=2.0, log=log)

    final_clips = []
    if intro_clip: final_clips.append(intro_clip)
    final_clips.extend(mono_clips)
    if outro_clip: final_clips.append(outro_clip)
    if closeout_clip: final_clips.append(closeout_clip)
    log(f'  final clip list: {len(final_clips)} clips ({len(mono_clips)} pairs + branding)')

    list_file = out_dir / '_master.txt'
    list_file.write_text('\n'.join(f"file '{s}'" for s in final_clips) + '\n')
    final = eval_dir / f'final-{bill_short}-cloud-avatar-podcast.mp4'
    r = subprocess.run([FFMPEG, '-y', '-f', 'concat', '-safe', '0', '-i', str(list_file),
                        '-c', 'copy', str(final)], capture_output=True, text=True)
    if r.returncode == 0:
        log(f'\n*** AVATAR MASTER: {final}')
        log(f'    size = {final.stat().st_size/1024/1024:.1f} MB')
        log(f'    dur  = {dur(final):.1f}s ({dur(final)/60:.2f} min)')
        return final
    log(f'FINAL CONCAT FAIL: {r.stderr[-400:]}')
    return None


def stage_hybrid_compose(eval_dir, bill_short, log=print):
    """Hybrid master: alternating talking-head and slide pairs (Day 7.20).

    Pair-index alternation (1-indexed):
      odd  -> slides       (uses scene-(2N-1).mp4 + scene-(2N).mp4 from compose/)
      even -> talking head (uses pair_NN_mono.mp4 from infinitetalk/compose/)
    Order is intro -> slide -> talking -> slide -> talking ... -> outro,
    so the lipsync intro hands off to a slide first (which usually carries
    the section setup / headline), then a talking-head pair reacts to it.

    Bookends with the same brand cards as stage_avatar_compose, which means
    the lipsync intro/outro overlay (Day 7.20) is picked up automatically
    via _make_card_clip's overlay short-circuit. Replaces both the slides-
    only and avatar-only masters; A/B comparison is no longer needed once
    the hybrid is verified.

    Output: <eval_dir>/final-<bill_short>-cloud-hybrid-podcast.mp4

    All input clips share codec params (h264 High yuv420p 832x480 @25fps,
    AAC LC mono 24kHz) so `-c copy` concat works without re-encoding. This
    was verified on bbb-cloud Day 7.20 -- if the upstream stages ever change
    encoding settings, this function will need a re-encode pass added.

    Pre-conditions:
      - infinitetalk/compose/pair_NN_mono.mp4 for each odd pair index
      - compose/scene-NN.mp4 for each scene used by an even pair index
    """
    avatar_dir = eval_dir / 'infinitetalk' / 'compose'
    slides_dir = eval_dir / 'compose'

    pair_clips = sorted(avatar_dir.glob('pair_*_mono.mp4'))
    if not pair_clips:
        log(f'NO PAIR MONO CLIPS in {avatar_dir}; run stage_avatar_compose first')
        return None
    n_pairs = len(pair_clips)
    log(f'STAGE 5-HYBRID: {n_pairs} pairs (alternating slides/avatar, odd=slides)')

    body_clips = []
    for pair_idx in range(1, n_pairs + 1):
        if pair_idx % 2 == 1:
            scene_a = (pair_idx - 1) * 2 + 1
            scene_b = scene_a + 1
            sa = slides_dir / f'scene-{scene_a:02d}.mp4'
            sb = slides_dir / f'scene-{scene_b:02d}.mp4'
            if not sa.exists() or not sb.exists():
                log(f'  pair_{pair_idx:02d} (slides): MISSING {sa.name} or {sb.name}')
                continue
            body_clips.extend([sa, sb])
            log(f'  pair_{pair_idx:02d} -> SLIDES ({sa.name}, {sb.name})')
        else:
            src = avatar_dir / f'pair_{pair_idx:02d}_mono.mp4'
            if not src.exists():
                log(f'  pair_{pair_idx:02d} (avatar): MISSING {src.name}')
                continue
            body_clips.append(src)
            log(f'  pair_{pair_idx:02d} -> AVATAR ({src.name})')

    if not body_clips:
        log('NO BODY CLIPS; aborting hybrid concat')
        return None

    # Reuse the avatar-mode card directory so the cached lipsync intro/outro
    # mp4s built by stage_avatar_compose are reused (no rebuild needed).
    out_dir = avatar_dir
    cards_dir = out_dir / '_cards'
    cards_dir.mkdir(exist_ok=True, parents=True)
    log('STAGE 5b: Brand cards (intro / outro / closeout)')
    intro_clip = _make_card_clip(BRAND_DIR / 'deadair_intro-1080.png',
                                  cards_dir / 'intro.mp4', duration_sec=4.0, log=log)
    outro_clip = _make_card_clip(BRAND_DIR / 'deadair_outro-1080.png',
                                  cards_dir / 'outro.mp4', duration_sec=4.0, log=log)
    closeout_clip = _make_card_clip(BRAND_DIR / 'deadair_closeout-1024.png',
                                     cards_dir / 'closeout.mp4', duration_sec=2.0, log=log)

    final_clips = []
    if intro_clip: final_clips.append(intro_clip)
    final_clips.extend(body_clips)
    if outro_clip: final_clips.append(outro_clip)
    if closeout_clip: final_clips.append(closeout_clip)
    log(f'  final clip list: {len(final_clips)} clips ({len(body_clips)} body + branding)')

    list_file = out_dir / '_master_hybrid.txt'
    list_file.write_text('\n'.join(f"file '{s}'" for s in final_clips) + '\n')
    final = eval_dir / f'final-{bill_short}-cloud-hybrid-podcast.mp4'
    r = subprocess.run([FFMPEG, '-y', '-f', 'concat', '-safe', '0', '-i', str(list_file),
                        '-c', 'copy', str(final)], capture_output=True, text=True)
    if r.returncode == 0:
        log(f'\n*** HYBRID MASTER: {final}')
        log(f'    size = {final.stat().st_size/1024/1024:.1f} MB')
        log(f'    dur  = {dur(final):.1f}s ({dur(final)/60:.2f} min)')
        return final
    log(f'FINAL CONCAT FAIL: {r.stderr[-400:]}')
    return None


# ---- HELPERS for the UI to predict the output path / detect cached runs ----
def _resolve_eval_dir(bill_short: str, override_headline: str = None,
                      creative_direction: str = None):
    """Return (eval_dir, is_custom, auto_headline). Pure function — no I/O side
    effects beyond reading the canonical report. Used by run_full_pipeline AND
    by the Gradio handler to skip when a cached video already exists.
    """
    canonical_dir = REPO / 'eval' / 'canonical'
    merged = canonical_dir / f'{bill_short}-merged.json'
    ch01 = canonical_dir / f'{bill_short}-ch01.json'
    canonical = merged if merged.exists() else ch01
    auto_headline = '(unknown)'
    if canonical.exists():
        try:
            report = json.load(open(canonical))
            rankings = (report.get('agents', {})
                              .get('headline_ranker', {})
                              .get('output', {})
                              .get('rankings') or [])
            if rankings:
                auto_headline = rankings[0].get('headline', '(unknown)')
        except Exception:
            pass

    is_custom = bool(
        (override_headline and override_headline.strip() and override_headline.strip() != auto_headline)
        or (creative_direction and creative_direction.strip())
    )
    if is_custom:
        import hashlib
        key_src = (
            'H::' + (override_headline or '').strip() + '||' +
            'D::' + (creative_direction or '').strip()
        )
        custom_key = hashlib.sha1(key_src.encode('utf-8')).hexdigest()[:8]
        eval_dir = REPO / 'eval' / f'{bill_short}-cloud-custom-{custom_key}'
    else:
        eval_dir = REPO / 'eval' / f'{bill_short}-cloud'
    return eval_dir, is_custom, auto_headline


def expected_final_path(bill_short: str, override_headline: str = None,
                        creative_direction: str = None):
    """Return the Path where run_full_pipeline would write the master mp4 for
    this (bill, headline, direction) combo. Caller can stat() it to short-
    circuit if the video has already been rendered."""
    eval_dir, _is_custom, _auto = _resolve_eval_dir(bill_short, override_headline, creative_direction)
    # Prefer the hybrid master (Day 7.20 user-preferred pacing). Fall back to
    # slides-only if that's all that exists (older renders pre-Day-7.22.4).
    hybrid = eval_dir / f'final-{bill_short}-cloud-hybrid-podcast.mp4'
    slides = eval_dir / f'final-{bill_short}-cloud-podcast.mp4'
    if hybrid.exists():
        return hybrid
    if slides.exists():
        return slides
    return hybrid  # default expected path for fresh runs


# ---- TOP-LEVEL PIPELINE (importable) ----
def run_full_pipeline(bill_short: str, log=print,
                      skip_text=False, skip_slides=False, skip_wan=False, skip_tts=False,
                      override_headline: str = None, creative_direction: str = None):
    """Run the full bill->podcast pipeline. Returns Path to final mp4 or None on failure.

    `log` is called with progress strings. Defaults to print() for CLI use.
    Pass a custom callback (e.g. appending to a list) for Gradio streaming.

    `override_headline`: when set, used instead of the auto-ranked winner. Allows
        the user to drive the dialog around any of the 10 candidate headlines (or
        a fully custom one they typed).
    `creative_direction`: when set, prepended to the report context fed to the
        script writer as additional instructions (tone, angle, etc.).

    When EITHER override is provided, output is routed to a custom subfolder
    `{bill}-cloud-custom-{8-char hash}` so it doesn't overwrite the canonical
    cache. Multiple custom variants can coexist.
    """
    # Prefer multi-chunk merged report if it exists, else fall back to ch01
    canonical_dir = REPO / 'eval' / 'canonical'
    merged = canonical_dir / f'{bill_short}-merged.json'
    ch01 = canonical_dir / f'{bill_short}-ch01.json'
    canonical = merged if merged.exists() else ch01
    if not canonical.exists():
        log(f'ERROR: missing canonical report (looked for {merged.name} and {ch01.name})')
        return None
    report = json.load(open(canonical))
    log(f'  loaded canonical: {canonical.name}')

    # Resolve eval_dir + auto-headline via the shared helper so the UI's
    # short-circuit path uses the SAME folder name as the actual run.
    eval_dir, is_custom, auto_headline = _resolve_eval_dir(
        bill_short, override_headline, creative_direction
    )
    headline = override_headline.strip() if (override_headline and override_headline.strip()) else auto_headline
    if is_custom:
        log(f'  CUSTOM RUN: eval_dir = {eval_dir.name}')
    else:
        log(f'  default eval_dir = {eval_dir.name}')
    eval_dir.mkdir(exist_ok=True, parents=True)

    log(f'=== Bill: {bill_short} | Headline: {headline} ===')
    if creative_direction and creative_direction.strip():
        log(f'  creative_direction: {creative_direction.strip()[:120]}')
    big_t0 = time.time()

    log('')
    log('[STAGE 1/4] Text agents (script + slide prompts + motion prompts)')
    if skip_text and (eval_dir/'script.json').exists():
        script = json.load(open(eval_dir/'script.json'))
        slides_obj = json.load(open(eval_dir/'slides.json'))
        motions_obj = json.load(open(eval_dir/'motions.json'))
        log('  [skip] all text caches present')
    else:
        script, slides_obj, motions_obj = stage_text_agents(
            report, eval_dir, headline, log=log,
            creative_direction=creative_direction,
        )

    log('')
    log('[STAGE 2/4] Slide generation + critique (Qwen-Image + Vision)')
    if not skip_slides:
        stage_slides(slides_obj, eval_dir, log=log)

    log('')
    log('[STAGE 3/4] Wan i2v animations + Qwen3-TTS narration')
    if not skip_wan:
        stage_wan(slides_obj, motions_obj, eval_dir, log=log)
    if not skip_tts:
        stage_tts(script, eval_dir, log=log)

    log('')
    log('[STAGE 4/5] InfiniteTalk avatar pairs (lipsync)')
    # 3090 fork: stage_avatar_render is the real fn name (line 576). The
    # pre-fork pipeline called it as stage_avatar and the try/except masked
    # the NameError, so the avatar stage silently skipped even when ComfyUI
    # was up. With this fix it will actually attempt InfiniteTalk and only
    # soft-fail on legitimate errors (ComfyUI down, missing mask asset, etc.)
    try:
        stage_avatar_render(script, eval_dir, log=log)
    except Exception as _e:
        log(f'  WARN: stage_avatar_render failed: {_e!r} -- continuing with slides-only')

    log('')
    log('[STAGE 5/5] FFMPEG compose -- slides + avatar + hybrid masters')
    slides_master = stage_compose(eval_dir, bill_short, log=log)
    try:
        avatar_master = stage_avatar_compose(eval_dir, bill_short, log=log)
    except Exception as _e:
        log(f'  WARN: stage_avatar_compose failed: {_e!r}')
        avatar_master = None
    try:
        hybrid_master = stage_hybrid_compose(eval_dir, bill_short, log=log)
    except Exception as _e:
        log(f'  WARN: stage_hybrid_compose failed: {_e!r}')
        hybrid_master = None
    # Hybrid is the user-preferred output (Day 7.20 pacing). Fall through
    # to avatar-only or slides-only if any lipsync stage failed.
    final = hybrid_master or avatar_master or slides_master

    log('')
    log(f'=== TOTAL: {time.time()-big_t0:.1f}s ===')
    if final:
        log(f'WATCH: {final}')
    return final


# ---- CLI WRAPPER ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bill', default='border25')
    ap.add_argument('--skip-text', action='store_true')
    ap.add_argument('--skip-slides', action='store_true')
    ap.add_argument('--skip-wan', action='store_true')
    ap.add_argument('--skip-tts', action='store_true')
    args = ap.parse_args()
    final = run_full_pipeline(
        args.bill,
        skip_text=args.skip_text,
        skip_slides=args.skip_slides,
        skip_wan=args.skip_wan,
        skip_tts=args.skip_tts,
    )
    if final is None:
        sys.exit(1)


if __name__ == '__main__':
    main()
