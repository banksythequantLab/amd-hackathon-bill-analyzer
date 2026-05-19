"""FreeClone process manager for the 3090-fork pipeline orchestrator.

FreeClone holds ~7 GB of VRAM (Whisper-large-v3 + VoxCPM2) while
running. When ComfyUI needs the GPU for heavy workflows (Qwen-Image at
peak uses ~18 GB; Wan-i2v ~14 GB; InfiniteTalk ~15 GB), keeping
FreeClone resident can OOM the 24 GB card. The orchestrator pauses
FreeClone before the ComfyUI block and resumes it before stage_tts.

This module wraps that pause/resume around B:\\freeclone-backend\\
START_BFORK.bat which the TODO #5 commit established as the canonical
launcher.

Usage:
    from scripts.freeclone_manager import FreeCloneManager

    fc = FreeCloneManager(launcher_bat=r'B:\\freeclone-backend\\START_BFORK.bat',
                           port=8300, work_dir=r'B:\\freeclone-backend')
    fc.pause()        # SIGTERM the process; port :8300 freed
    # ...heavy ComfyUI work...
    fc.resume()       # respawn from .bat; wait for /health
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

import httpx


logger = logging.getLogger(__name__)


class FreeCloneError(Exception):
    """Raised on launch / health-check failures."""


class FreeCloneManager:
    def __init__(
        self,
        launcher_bat: str | Path = r'B:\freeclone-backend\START_BFORK.bat',
        port: int = 8300,
        work_dir: str | Path = r'B:\freeclone-backend',
        startup_timeout_s: float = 120.0,
        log_path: str | Path | None = None,
    ):
        self.launcher_bat = Path(launcher_bat)
        self.port = port
        self.work_dir = Path(work_dir)
        self.startup_timeout_s = startup_timeout_s
        self.log_path = Path(log_path) if log_path else None

        if not self.launcher_bat.exists():
            raise FreeCloneError(f'launcher .bat not found at {self.launcher_bat}')
        if not self.work_dir.exists():
            raise FreeCloneError(f'work dir not found at {self.work_dir}')

        self._proc: subprocess.Popen | None = None

    def is_running(self) -> bool:
        try:
            r = httpx.get(f'http://127.0.0.1:{self.port}/health', timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def pause(self, timeout_s: float = 10.0) -> None:
        """Stop the FreeClone process if running. Frees its VRAM (~7 GB)."""
        if not self.is_running():
            logger.info('FreeClone not running; nothing to pause')
            return

        # We may not have spawned it ourselves; find by port and kill.
        try:
            import psutil
            for conn in psutil.net_connections(kind='tcp'):
                if conn.laddr.port == self.port and conn.status == 'LISTEN':
                    try:
                        p = psutil.Process(conn.pid)
                        # Walk up to find the python.exe (the launcher .bat
                        # spawns cmd.exe -> python.exe).
                        py_pid = conn.pid
                        # Also kill children just in case.
                        children = p.children(recursive=True)
                        logger.info('FreeClone pause: killing PID %d + %d children',
                                    conn.pid, len(children))
                        for c in children:
                            try: c.terminate()
                            except Exception: pass
                        p.terminate()
                        p.wait(timeout=timeout_s)
                    except Exception as e:
                        logger.warning('FreeClone pause: psutil error: %s', e)
                    break
        except ImportError:
            logger.warning('psutil not installed; cannot pause FreeClone cleanly')
            return

        # Wait for port to actually clear
        t0 = time.time()
        while time.time() - t0 < 10:
            if not self.is_running():
                logger.info('FreeClone paused (port :%d freed)', self.port)
                return
            time.sleep(0.5)
        logger.warning('FreeClone still responding on :%d after pause attempt', self.port)

    def resume(self) -> int:
        """Spawn FreeClone via the launcher .bat. Returns PID once /health
        responds 200. Raises FreeCloneError on timeout."""
        if self.is_running():
            logger.info('FreeClone already running on :%d', self.port)
            return -1

        stdout = open(self.log_path, 'wb') if self.log_path else subprocess.DEVNULL
        stderr = open(str(self.log_path) + '.err', 'wb') if self.log_path else subprocess.DEVNULL

        logger.info('Resuming FreeClone via %s', self.launcher_bat)
        self._proc = subprocess.Popen(
            [str(self.launcher_bat)],
            cwd=str(self.work_dir),
            stdout=stdout,
            stderr=stderr,
            shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )

        t0 = time.time()
        while time.time() - t0 < self.startup_timeout_s:
            if self.is_running():
                elapsed = time.time() - t0
                logger.info('FreeClone resumed in %.1fs', elapsed)
                # The .bat launches a child python.exe; our self._proc is the
                # cmd.exe wrapper which will exit quickly. The real server is
                # one of its descendants and is now serving on :port. Just
                # return the cmd PID for record-keeping.
                return self._proc.pid
            if self._proc.poll() is not None:
                # The .bat exited; the real python child is detached.
                # Still wait for /health since it's the canonical signal.
                pass
            time.sleep(2.0)

        raise FreeCloneError(
            f'FreeClone did not respond on :{self.port} within {self.startup_timeout_s}s '
            f'(check {self.log_path}.err)'
        )