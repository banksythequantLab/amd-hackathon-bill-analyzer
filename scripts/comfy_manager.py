"""ComfyUI process manager for the 3090-fork bill-analyzer pipeline.

Spawns / monitors / stops a ComfyUI subprocess so the orchestrator can
manage the 24 GB VRAM ceiling on Johnson's RTX 3090. Different stages
of make_podcast_cloud need different ComfyUI workflows (Qwen-Image for
slides, Wan-i2v for video, Wan + InfiniteTalk for avatar lipsync); each
loads ~12-18 GB of models. ComfyUI itself handles per-workflow model
swap internally, but we need an external manager to:

  1. Boot ComfyUI from a chosen install root (B: or E:)
  2. Wait for /system_stats to return 200 before returning control
  3. Hold the PID for clean shutdown
  4. Validate required model files exist BEFORE submitting a workflow
     (prevents 5-minute hangs when a model is missing)
  5. Optionally pause FreeClone first to free 7 GB of VRAM

Architecture: one long-lived ComfyUI process for the whole pipeline run.
ComfyUI's own model cache handles intra-process swaps. We don't try to
kill+restart between workflows -- ComfyUI's offload-to-RAM is more
efficient than cold-loading a 19 GB model from disk every stage.

Usage:
    from scripts.comfy_manager import ComfyManager

    with ComfyManager(install_root=r'E:\ComfyUI-Easy-Install\ComfyUI-Easy-Install',
                       port=8188,
                       attention='sage') as mgr:
        # ComfyUI is up; submit workflows via httpx as usual
        ...
    # ComfyUI is cleanly stopped on context exit
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import httpx


logger = logging.getLogger(__name__)


COMFY_DEFAULT_PORT = 8188

# Required model files keyed by workflow. Missing-model pre-flight check
# fails fast instead of letting ComfyUI return a cryptic node error
# 5 minutes into a render.
REQUIRED_MODELS = {
    'qwen_image': {
        'diffusion_models': ['qwen_image_2512_fp8_e4m3fn_scaled.safetensors',
                              'qwen_image_2512_fp8_e4m3fn.safetensors'],  # either name OK
        'text_encoders':    ['qwen_2.5_vl_7b_fp8_scaled.safetensors'],
        'vae':              ['qwen_image_vae.safetensors'],
        'loras':            ['Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors',
                              'Qwen-Image-Lightning-4steps-V2.0.safetensors'],  # either OK
    },
    'wan_i2v': {
        'diffusion_models': ['wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors'],
        'text_encoders':    ['umt5_xxl_fp8_e4m3fn_scaled.safetensors'],
        'vae':              ['wan_2.1_vae.safetensors'],
        'loras':            ['wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors'],
    },
    'infinitetalk': {
        'diffusion_models': ['Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors'],
        'text_encoders':    ['umt5_xxl_fp8_e4m3fn_scaled.safetensors'],
        'vae':              ['Wan2_1_VAE_bf16.safetensors'],
        'audio_encoders':   ['wav2vec2-chinese-base_fp16.safetensors'],
        # InfiniteTalk model sometimes lives in diffusion_models or
        # model_patches (we saw it under model_patches in B: scan).
        'model_patches':    ['wan2.1_infiniteTalk_multi_fp16.safetensors'],
    },
}


class ComfyError(Exception):
    """Raised on launch failures, health-check timeouts, or missing models."""


class ComfyManager:
    """Manages a single ComfyUI subprocess.

    Args:
        install_root: dir containing 'ComfyUI/main.py' + 'python_embeded/python.exe'
        port: ComfyUI HTTP port (default 8188)
        attention: 'sage', 'flash', or 'default' -- picks the corresponding
                   Start ComfyUI bat-equivalent CLI flags
        models_root: optional override for models dir (defaults to install_root/ComfyUI/models)
        startup_timeout_s: max wait for /system_stats to respond
        log_path: where to capture ComfyUI stdout (stderr goes to .err)
    """

    def __init__(
        self,
        install_root: str | Path,
        port: int = COMFY_DEFAULT_PORT,
        attention: str = 'sage',
        models_root: str | Path | None = None,
        startup_timeout_s: float = 180.0,
        log_path: str | Path | None = None,
    ):
        self.install_root = Path(install_root)
        self.port = port
        self.attention = attention
        self.models_root = Path(models_root) if models_root else self.install_root / 'ComfyUI' / 'models'
        self.startup_timeout_s = startup_timeout_s
        self.log_path = Path(log_path) if log_path else None

        self.python_exe = self.install_root / 'python_embeded' / 'python.exe'
        self.main_py = self.install_root / 'ComfyUI' / 'main.py'

        if not self.python_exe.exists():
            raise ComfyError(f'python_embeded not found at {self.python_exe}')
        if not self.main_py.exists():
            raise ComfyError(f'ComfyUI main.py not found at {self.main_py}')
        if not self.models_root.exists():
            raise ComfyError(f'models dir not found at {self.models_root}')

        self._proc: subprocess.Popen | None = None

    # ---- model pre-flight ----

    def check_models(self, workflows: Iterable[str]) -> dict:
        """Verify required model files exist for each named workflow.

        Returns a dict {workflow: {'missing': [...], 'found': [...]}}.
        Doesn't raise -- caller decides whether to fail or continue with
        a subset of workflows.
        """
        report = {}
        for wf_name in workflows:
            if wf_name not in REQUIRED_MODELS:
                report[wf_name] = {'missing': ['UNKNOWN WORKFLOW'], 'found': []}
                continue
            missing = []
            found = []
            for subdir, names in REQUIRED_MODELS[wf_name].items():
                dir_path = self.models_root / subdir
                hit = False
                for n in names:
                    candidates = list(dir_path.rglob(n))
                    if candidates:
                        found.append(f'{subdir}/{n}  -> {candidates[0]}')
                        hit = True
                        break
                if not hit:
                    missing.append(f'{subdir}/{names[0]}  (or alternatives: {names[1:] if len(names) > 1 else "none"})')
            report[wf_name] = {'missing': missing, 'found': found}
        return report

    def assert_models(self, workflows: Iterable[str]) -> None:
        """Like check_models() but raises ComfyError if any are missing."""
        report = self.check_models(workflows)
        all_missing = []
        for wf, status in report.items():
            for m in status['missing']:
                all_missing.append(f'{wf}: {m}')
        if all_missing:
            raise ComfyError(
                'Missing models:\n  ' + '\n  '.join(all_missing) +
                f'\n\n(Searched under {self.models_root})'
            )

    # ---- lifecycle ----

    def is_running(self) -> bool:
        """True if /system_stats returns 200 within 2s."""
        try:
            r = httpx.get(f'http://127.0.0.1:{self.port}/system_stats', timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def start(self) -> int:
        """Launch ComfyUI. Returns PID once /system_stats responds 200,
        else raises ComfyError on timeout."""
        if self.is_running():
            logger.warning('ComfyUI already running on :%d; not starting another', self.port)
            return -1

        args = [
            str(self.python_exe),
            '-I', '-W', 'ignore::FutureWarning',
            str(self.main_py),
            '--port', str(self.port),
            '--disable-dynamic-vram',
            '--windows-standalone-build',
        ]
        if self.attention == 'sage':
            args.append('--use-sage-attention')
        elif self.attention == 'flash':
            args.append('--use-flash-attention')

        stdout = open(self.log_path, 'wb') if self.log_path else subprocess.DEVNULL
        stderr = open(str(self.log_path) + '.err', 'wb') if self.log_path else subprocess.DEVNULL

        logger.info('Launching ComfyUI: %s', ' '.join(args))
        self._proc = subprocess.Popen(
            args,
            cwd=str(self.install_root),
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )

        # Poll /system_stats until ready
        t0 = time.time()
        while time.time() - t0 < self.startup_timeout_s:
            if self.is_running():
                elapsed = time.time() - t0
                logger.info('ComfyUI ready in %.1fs (PID %d)', elapsed, self._proc.pid)
                return self._proc.pid
            if self._proc.poll() is not None:
                rc = self._proc.returncode
                raise ComfyError(
                    f'ComfyUI exited with code {rc} during startup '
                    f'(check {self.log_path}.err)'
                )
            time.sleep(2.0)
        # Timeout
        self.stop()
        raise ComfyError(f'ComfyUI did not respond on :{self.port} within {self.startup_timeout_s}s')

    def stop(self, timeout_s: float = 15.0) -> None:
        """Gracefully stop the ComfyUI process. SIGTERM, then kill on timeout."""
        if not self._proc:
            # We didn't launch it. Try to find by port and kill anyway.
            self._stop_external()
            return
        if self._proc.poll() is not None:
            return  # Already exited

        try:
            self._proc.terminate()
            self._proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            logger.warning('ComfyUI did not exit gracefully; killing')
            self._proc.kill()
            self._proc.wait(timeout=5.0)
        logger.info('ComfyUI stopped (PID %d)', self._proc.pid)
        self._proc = None

    def _stop_external(self) -> None:
        """Kill any process listening on self.port (when we didn't launch it
        ourselves). Useful for re-running scripts."""
        try:
            import psutil
            for conn in psutil.net_connections(kind='tcp'):
                if conn.laddr.port == self.port and conn.status == 'LISTEN':
                    p = psutil.Process(conn.pid)
                    logger.info('Killing external ComfyUI (PID %d)', conn.pid)
                    p.terminate()
                    p.wait(timeout=10.0)
                    return
        except ImportError:
            logger.warning('psutil not installed; cannot kill external ComfyUI')

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False  # don't swallow exceptions


@contextmanager
def comfy_for(workflows: Iterable[str], **kwargs):
    """Convenience: pre-flight check models then start/stop ComfyUI.

    Usage:
        with comfy_for(['qwen_image', 'wan_i2v', 'infinitetalk'],
                        install_root=r'E:\\ComfyUI-Easy-Install\\ComfyUI-Easy-Install') as mgr:
            ...submit workflows...
    """
    mgr = ComfyManager(**kwargs)
    mgr.assert_models(workflows)
    with mgr:
        yield mgr