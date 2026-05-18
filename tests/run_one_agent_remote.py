"""
Linux-side runner for run_one_agent.py — eliminates the Windows httpx hang.

Background:
  Day 2 evening burned ~2 hours on a recurring Python-side hang where spine
  returned 200 OK but the Windows Python client never returned to the script.
  Diagnosed as a Windows + venv-launcher + httpx response-body interaction.
  See commit 32bf7e0 for the full diagnosis.

This module is the Day 3 fix: instead of running the agent test on Vesper
(Windows) and having it call across the public internet to spine, SSH into
the cloud instance and run the agent test ON the instance, where:
  - Linux Python doesn't have the venv-launcher quirk
  - The httpx call is to localhost:8001, no public internet
  - No Windows in the loop at all

Usage from Vesper (or wherever):
    python tests/run_one_agent_remote.py --agent summarizer --bill bbb --chunk-id ch01
    python tests/run_one_agent_remote.py --agent xref       --bill hr1 --chunk-id ch01
    python tests/run_one_agent_remote.py --agent pork       --bill bbb --chunk-id ch01
    python tests/run_one_agent_remote.py --agent conflict   --bill bbb --chunk-id ch01

Prerequisites on the cloud instance (one-time setup, see Day 3 runbook):
    /root/repo/                          - this repo, cloned
    /root/repo/.venv/                    - Python 3.12 venv with deps
    /root/usc/usc.lmdb/                  - the USC LMDB (already there)
    /root/bills/chunks-bbb-full.json     - chunks files (uploaded via scp once)
    /root/bills/chunks-hr1-full.json
    /root/bills/chunks-ndaa-full.json

The script will scp the chunks files up automatically if they're missing.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

# Cloud-side paths (on the AMD instance). NOT used by the 3090 fork --
# this whole script tunnels SSH to the AMD droplet for canonical runs.
# On Johnson, run tests/run_one_agent.py directly instead.
INSTANCE_HOST = "165.245.134.1"  # Update if droplet IP changes (AMD only)
SSH_KEY = Path.home() / ".ssh" / "amd_hackathon"
SSH_USER = "root"

REMOTE_REPO_DIR = "/root/repo"
REMOTE_VENV_PY  = "/root/repo/.venv/bin/python"
REMOTE_USC_LMDB = "/root/usc/usc.lmdb"
REMOTE_BILLS_DIR = "/root/bills"
REMOTE_OUT_DIR  = "/root/agent-smoke"

# Local-side paths (on Vesper). Comment header says Vesper but this script
# also runs on Johnson (3090 fork). LOCAL_REPO derived so SCP+SSH commands
# upload from THIS fork, not the side-by-side AMD-baseline copy.
LOCAL_BILLS_DIR = Path(r"B:\hackathon-build")
LOCAL_OUT_DIR   = Path(r"B:\hackathon-build\agent-smoke")
LOCAL_REPO      = Path(__file__).resolve().parents[1]  # was r"B:\amd-hackathon-bill-analyzer"

CHUNK_FILE_NAMES = {
    "bbb":  "chunks-bbb-full.json",
    "hr1":  "chunks-hr1-full.json",
    "ndaa": "chunks-ndaa-full.json",
}

AGENT_VALID = {"summarizer", "xref", "pork", "conflict", "fiscal", "stakeholder", "podcast", "relay"}

# Use Git's bundled SSH on Windows (Windows OpenSSH stderr is broken under subprocess)
GIT_SSH = r"C:\Program Files\Git\usr\bin\ssh.exe"
GIT_SCP = r"C:\Program Files\Git\usr\bin\scp.exe"


def _ssh_cmd(*, ssh_args: list[str] | None = None) -> list[str]:
    base = [GIT_SSH, "-o", "StrictHostKeyChecking=accept-new", "-i", str(SSH_KEY)]
    if ssh_args:
        base.extend(ssh_args)
    base.append(f"{SSH_USER}@{INSTANCE_HOST}")
    return base


def _scp_cmd(local: Path, remote_path: str) -> list[str]:
    return [
        GIT_SCP, "-o", "StrictHostKeyChecking=accept-new",
        "-i", str(SSH_KEY),
        str(local),
        f"{SSH_USER}@{INSTANCE_HOST}:{remote_path}",
    ]


def run_remote(remote_cmd: str, *, timeout: int = 900) -> tuple[int, str, str]:
    """Run a shell command on the instance via SSH; return (rc, stdout, stderr)."""
    full = _ssh_cmd() + [remote_cmd]
    p = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def ensure_remote_chunks(bill: str) -> None:
    """If the chunks file for `bill` is missing on the instance, upload it."""
    fname = CHUNK_FILE_NAMES[bill]
    remote_path = f"{REMOTE_BILLS_DIR}/{fname}"
    rc, _, _ = run_remote(f"test -s {shlex.quote(remote_path)}", timeout=30)
    if rc == 0:
        return  # already present and non-empty
    print(f"[remote] uploading {fname} (one-time setup)...", flush=True)
    run_remote(f"mkdir -p {shlex.quote(REMOTE_BILLS_DIR)}", timeout=30)
    local = LOCAL_BILLS_DIR / fname
    if not local.exists():
        raise SystemExit(f"local chunks file missing: {local}")
    cmd = _scp_cmd(local, remote_path)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        raise SystemExit(f"scp failed: {p.stderr}")
    print(f"[remote] uploaded {local.stat().st_size:,} bytes", flush=True)


def ensure_remote_repo() -> None:
    """Verify repo + venv exist on instance. If venv is missing, instruct user."""
    rc, _, _ = run_remote(f"test -x {shlex.quote(REMOTE_VENV_PY)}", timeout=30)
    if rc == 0:
        return
    raise SystemExit(
        f"\nERROR: remote venv not found at {REMOTE_VENV_PY}\n\n"
        f"On the instance, run once:\n"
        f"  cd {REMOTE_REPO_DIR}\n"
        f"  git pull\n"
        f"  python3.12 -m venv .venv\n"
        f"  ./.venv/bin/pip install -q pdfplumber lmdb orjson httpx pydantic tiktoken\n"
        f"\nThen re-run this command.\n"
    )


def fetch_remote_outputs(agent: str, bill: str, chunk_id: str) -> list[Path]:
    """Pull the result file(s) the remote agent wrote back to the local dir."""
    LOCAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    fetched = []
    # The remote run writes:
    #   {agent_internal_name}-{chunk_id}.json
    #   {agent_internal_name}-metric-{chunk_id}.json
    # The agent_internal_name is set on each agent class (e.g. plain_english_summarizer).
    # We don't know it from the CLI label alone — pull anything matching the chunk_id
    # from the remote out dir that's newer than 60s, prefixed with bill for archive.
    rc, out, _ = run_remote(
        f"find {REMOTE_OUT_DIR} -type f -newer {REMOTE_OUT_DIR}/.last_pull -name '*-{chunk_id}.json' 2>/dev/null || "
        f"find {REMOTE_OUT_DIR} -type f -name '*-{chunk_id}.json'",
        timeout=30,
    )
    files = [line.strip() for line in out.strip().split("\n") if line.strip()]
    for remote_file in files:
        local_name = f"{Path(remote_file).stem}-{bill}.json"  # tag with bill on local
        local_path = LOCAL_OUT_DIR / local_name
        cmd = [
            GIT_SCP, "-o", "StrictHostKeyChecking=accept-new",
            "-i", str(SSH_KEY),
            f"{SSH_USER}@{INSTANCE_HOST}:{remote_file}",
            str(local_path),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if p.returncode == 0:
            fetched.append(local_path)
            print(f"[remote] pulled {local_path.name} ({local_path.stat().st_size:,} bytes)", flush=True)
        else:
            print(f"[remote] WARN: scp back failed for {remote_file}: {p.stderr.strip()}", flush=True)
    # mark the timestamp for next pull
    run_remote(f"touch {REMOTE_OUT_DIR}/.last_pull", timeout=10)
    return fetched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True, choices=sorted(AGENT_VALID))
    ap.add_argument("--bill", default="bbb", choices=list(CHUNK_FILE_NAMES))
    ap.add_argument("--chunk-id", default="ch01")
    ap.add_argument("--report-file", type=str, default=None,
                    help="For podcast agent: path to a LOCAL report JSON; will be uploaded to instance.")
    ap.add_argument("--no-fetch", action="store_true",
                    help="Don't pull result files back to local")
    ap.add_argument("--git-pull", action="store_true",
                    help="git pull on the instance before running (default: skip)")
    args = ap.parse_args()

    print(f"[remote] target: {SSH_USER}@{INSTANCE_HOST}", flush=True)
    print(f"[remote] agent={args.agent} bill={args.bill} chunk_id={args.chunk_id}", flush=True)

    ensure_remote_repo()
    ensure_remote_chunks(args.bill)

    if args.git_pull:
        print(f"[remote] git pull...", flush=True)
        rc, out, err = run_remote(f"cd {REMOTE_REPO_DIR} && git pull --ff-only 2>&1", timeout=60)
        print(out.strip() or err.strip(), flush=True)

    # Build the remote run_one_agent invocation.
    # The remote run_one_agent.py needs path overrides to point at /root/usc and /root/bills
    # rather than the Windows defaults baked into the file. We pass the overrides via
    # environment variables so we don't have to fork the script.
    # Endpoint env vars route the cloud-side run to localhost (no NAT, no public network hop).
    # Path env vars override the Windows defaults baked into run_one_agent.py.
    env_vars = (
        f"BILL_ANALYZER_USC_LMDB={REMOTE_USC_LMDB} "
        f"BILL_ANALYZER_CHUNKS_DIR={REMOTE_BILLS_DIR} "
        f"BILL_ANALYZER_OUT_DIR={REMOTE_OUT_DIR} "
        f"BILL_ANALYZER_SPINE_URL=http://localhost:8001/v1 "
        f"BILL_ANALYZER_REASONER_URL=http://localhost:8003/v1 "
        f"BILL_ANALYZER_VISION_URL=http://localhost:8002/v1"
    )
    remote_cmd = (
        f"mkdir -p {REMOTE_OUT_DIR} && "
        f"cd {REMOTE_REPO_DIR} && "
        f"{env_vars} {REMOTE_VENV_PY} -u tests/run_one_agent.py "
        f"--agent {shlex.quote(args.agent)} "
        f"--bill {shlex.quote(args.bill)} "
        f"--chunk-id {shlex.quote(args.chunk_id)}"
    )

    # If podcast agent, upload the report file first
    remote_report_path = ""
    if args.agent in ("podcast", "relay"):
        if not args.report_file:
            print("[remote] ERROR: --report-file required for podcast agent", flush=True)
            return 1
        from pathlib import Path as _Path
        local_rf = _Path(args.report_file)
        if not local_rf.exists():
            print(f"[remote] ERROR: local report file not found: {local_rf}", flush=True)
            return 1
        remote_report_path = f"/root/agent-smoke/{local_rf.name}"
        print(f"[remote] uploading {local_rf} -> {remote_report_path}", flush=True)
        scp_cmd = ["scp", "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=accept-new", str(local_rf), f"{SSH_USER}@{INSTANCE_HOST}:{remote_report_path}"]
        scp_rc = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=120)
        if scp_rc.returncode != 0:
            print(f"[remote] scp failed: {scp_rc.stderr}", flush=True)
            return 1
        # Append --report-file to the remote command
        remote_cmd += f" --report-file {shlex.quote(remote_report_path)}"

    print(f"[remote] running agent (cold prefill ~5min, APC-warm ~30-90s)...", flush=True)
    t0 = time.perf_counter()
    rc, out, err = run_remote(remote_cmd, timeout=900)
    elapsed = time.perf_counter() - t0
    print(f"[remote] elapsed {elapsed:.1f}s rc={rc}", flush=True)

    if out.strip():
        print("--- remote stdout ---", flush=True)
        print(out, flush=True)
    if err.strip():
        print("--- remote stderr ---", flush=True)
        print(err, flush=True)

    if rc != 0:
        print(f"[remote] FAILED (rc={rc})", flush=True)
        return rc

    if not args.no_fetch:
        fetched = fetch_remote_outputs(args.agent, args.bill, args.chunk_id)
        if not fetched:
            print(f"[remote] WARN: no result files pulled back", flush=True)
            return 2

    print(f"[remote] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())