"""Cloud podcast pipeline. All compute on MI300X. Idempotent + resumable."""
from __future__ import annotations
import argparse, base64, copy, json, sys, time
from pathlib import Path
import httpx

REPO = Path(r"B:\\amd-hackathon-bill-analyzer")
sys.path.insert(0, str(REPO))

from src.agents.podcast_script_writer import PodcastScriptWriter
from src.agents.slide_prompt_generator import SlidePromptGenerator
from src.agents.wan_motion_prompt_generator import WanMotionPromptGenerator
from src.agents.slide_critic import SlideCritic
from src.agents.youtube_metadata_generator import YouTubeMetadataGenerator

COMFY = "http://165.245.134.1:8188"
VOICE_MAP = {'Alex': 'Ryan', 'Jordan': 'Ono_anna'}
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
        # Up to 7 retry attempts: SlideCritic dual-call (OCR + judgment) is
        # strict about typos and visual quality. With 4-step Lightning the
        # tail of the seed distribution still produces ~10-20% slides that
        # need a re-roll, so 7 attempts gives ~99.99% pass probability.
        for attempt in range(7):
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

def stage_tts(script, eval_dir, log=print):
    tts_dir = eval_dir / 'tts'
    tts_dir.mkdir(exist_ok=True, parents=True)
    for line in script['dialog']:
        scene = line['scene']
        speaker = line['speaker']
        voice = VOICE_MAP.get(speaker, 'Ryan')
        text = line['line']
        out_path = tts_dir / f'scene-{scene:02d}.flac'
        if out_path.exists() and out_path.stat().st_size > 10_000:
            log(f'  [{scene:02d}] cached tts ({out_path.stat().st_size//1024}KB)')
            continue
        prefix = f'border-tts/scene-{scene:02d}-{voice}'
        wf = tts_workflow(text, voice, prefix)
        pid = submit(wf, f'tts-{scene}')
        outs, elapsed = wait_for(pid, f'tts-{scene}', timeout=180)
        auds = outs.get('2', {}).get('audio', [])
        if not auds:
            log(f'  [{scene:02d}] no audio! outs={outs}')
            continue
        sz = download(auds[0]['filename'], auds[0]['subfolder'], 'output', out_path)
        log(f'  [{scene:02d}] {voice}: {elapsed:.1f}s -> {sz//1024}KB | "{text[:60]}..."')

# ---- COMPOSE ----
import subprocess
FFMPEG = r"C:\\Users\\solti\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-7.1.1-full_build\\bin\\ffmpeg.exe"
FFPROBE = r"C:\\Users\\solti\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-7.1.1-full_build\\bin\\ffprobe.exe"

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
    return eval_dir / f'final-{bill_short}-cloud-podcast.mp4'


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
    log('[STAGE 4/4] FFMPEG compose')
    final = stage_compose(eval_dir, bill_short, log=log)

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
