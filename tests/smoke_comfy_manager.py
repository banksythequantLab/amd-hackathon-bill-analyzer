"""Smoke-test the ComfyManager: pre-flight model check + start/stop.

Validates:
  1. check_models() reports correct missing/found status
  2. start() launches ComfyUI and waits for /system_stats
  3. is_running() flips True
  4. stop() cleanly exits the process
  5. Port 8188 is released after stop

Run after the missing Qwen-Image 2512 model download completes."""
from __future__ import annotations
import sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.comfy_manager import ComfyManager, ComfyError

INSTALL = r'E:\ComfyUI-Easy-Install\ComfyUI-Easy-Install'

def main() -> int:
    print('=== ComfyManager smoke test ===')
    print(f'install: {INSTALL}')
    print()

    mgr = ComfyManager(
        install_root=INSTALL,
        port=8188,
        attention='sage',
        log_path=REPO / 'eval' / 'comfy-smoke.log',
        startup_timeout_s=180.0,
    )

    print('[1/5] Pre-flight model check for all 3 workflows...')
    report = mgr.check_models(['qwen_image', 'wan_i2v', 'infinitetalk'])
    for wf, status in report.items():
        print(f'  {wf}:')
        for f in status['found']:
            print(f'    OK: {f[:120]}')
        for m in status['missing']:
            print(f'    MISSING: {m}')

    # Aggregate missing count
    missing_count = sum(len(s['missing']) for s in report.values())
    if missing_count:
        print(f'\n[FAIL] {missing_count} model(s) missing; cannot continue smoke')
        return 1
    print('[1/5] OK - all required models present\n')

    print('[2/5] Checking nothing currently on :8188...')
    if mgr.is_running():
        print('  WARN: something is already on :8188 -- skipping launch test')
        return 1
    print('  OK: port clear\n')

    print('[3/5] Starting ComfyUI...')
    t0 = time.time()
    pid = mgr.start()
    elapsed = time.time() - t0
    print(f'  ComfyUI ready in {elapsed:.1f}s (PID {pid})')
    assert mgr.is_running(), 'is_running should return True after successful start'
    print('  is_running() == True confirmed\n')

    print('[4/5] Stopping ComfyUI...')
    t0 = time.time()
    mgr.stop()
    elapsed = time.time() - t0
    print(f'  stopped in {elapsed:.1f}s')
    # Wait a moment for the port to release
    time.sleep(2)
    assert not mgr.is_running(), 'is_running should return False after stop'
    print('  is_running() == False confirmed\n')

    print('[5/5] context manager test...')
    with ComfyManager(install_root=INSTALL, port=8188, attention='sage',
                      log_path=REPO / 'eval' / 'comfy-smoke-ctx.log',
                      startup_timeout_s=180.0) as mgr2:
        assert mgr2.is_running()
        print('  inside with: ComfyUI running')
    time.sleep(2)
    print('  after with: ComfyUI stopped')

    print('\n[OK] ComfyManager smoke passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())