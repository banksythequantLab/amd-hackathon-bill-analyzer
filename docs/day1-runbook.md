# Day 1 Runbook — Mon May 4, 2026

**Cloud-side work, paste-and-go.** Everything below is intended to be run in
sequence on the AMD MI300X cloud instance via SSH. Pre-flight pieces
(`vllm_serve.sh`, `apc_benchmark.py`, `usc.lmdb`) already exist on Johnson's
B:\ from Day 0.

---

## 0. Spin the instance (UI work — Derek)

In the AMD Developer Cloud console:

1. Create GPU Droplet → MI300X
2. **OS image:** `ROCm 7.2.0 + vLLM 0.17.1 Quick Start` (we want vLLM pre-baked)
3. SSH key: `banksy-johnson` (already registered)
4. Confirm the SSH command (looks like `ssh root@<IP>`)
5. Click create

Note the public IP. Pass it to me; I'll take over from there.

**Cost gate:** From this moment, the meter runs at $1.99/hr. Don't go to lunch.

---

## 1. First-touch verification (5 min)

```bash
# Confirm hardware + OS
rocminfo | head -30
nvidia-smi 2>/dev/null || echo "(no nvidia, expected)"
rocm-smi --showmeminfo vram --showuse
python3 --version  # expect 3.12+
ldd --version | head -1   # expect glibc >= 2.35
df -h /scratch     # expect lots of free space, this is where we put models + logs
```

**Acceptance:** rocminfo lists "Agent 1: AMD Instinct MI300X". Anything else = STOP, wrong instance type.

---

## 2. Upload Day-0 artifacts from Johnson (10 min)

From Vesper or your laptop (where you have B:\ mounted):

```powershell
# Replace <INSTANCE_IP> with the actual IP
$IP = "<INSTANCE_IP>"
$KEY = "$env:USERPROFILE\.ssh\amd_hackathon"

# Create the scratch layout on the instance
ssh -i $KEY root@$IP "mkdir -p /scratch/{usc,bills,vllm-logs,vllm-pids,hf-cache,repo}"

# Upload USC LMDB (~379 MB compact)
scp -i $KEY -r 'B:\amd-hackathon-bill-analyzer\data\usc.lmdb' root@${IP}:/scratch/usc/

# Upload the three demo bills
scp -i $KEY 'B:\amd-hackathon-bill-analyzer\tests\fixtures\*.pdf' root@${IP}:/scratch/bills/

# Clone the repo onto the instance (public, so no creds needed)
ssh -i $KEY root@$IP "cd /scratch/repo && git clone https://github.com/banksythequantLab/amd-hackathon-bill-analyzer.git ."
```

**Acceptance on the instance:**
```bash
ls -la /scratch/usc/usc.lmdb/      # should show data.mdb (~379 MB), lock.mdb
ls /scratch/bills/                 # 3 PDFs
ls /scratch/repo/infra/            # vllm_serve.sh, apc_benchmark.py, monitor.sh, usc_corpus_build.py
```

---

## 3. Install Python deps (10 min)

```bash
cd /scratch/repo
python3 -m venv /scratch/venv
source /scratch/venv/bin/activate
pip install --upgrade pip
pip install lmdb orjson pdfplumber httpx Pillow tenacity pydantic
# vLLM should already be in the Quick Start image — confirm:
which vllm && vllm --version
```

**Acceptance:** `vllm --version` returns 0.17.x or later, no import errors.

---

## 4. Smoke #1: Spine BF16 (load test, ~10 min)

```bash
# Make scripts executable
chmod +x /scratch/repo/infra/*.sh

# Launch spine in BF16 first to confirm full-precision works
PRECISION=bf16 \
LOG_DIR=/scratch/vllm-logs \
PID_DIR=/scratch/vllm-pids \
HF_HOME=/scratch/hf-cache \
/scratch/repo/infra/vllm_serve.sh spine

# Watch download + load progress in another tmux pane
tail -f /scratch/vllm-logs/spine.log
```

First load downloads ~70GB of BF16 weights from HF — expect ~5-8 min.

**Acceptance:** `curl http://localhost:8001/health` returns 200.

```bash
# Quick generation smoke
curl -s http://localhost:8001/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"spine","prompt":"USC Title 26 covers","max_tokens":20,"temperature":0}' \
    | python3 -m json.tool
```

Should return a coherent completion mentioning taxes / IRS / etc.

```bash
# Log the BF16 VRAM peak
rocm-smi --showmeminfo vram | tee /scratch/vllm-logs/vram-bf16.txt
```

---

## 5. Smoke #2: Spine FP8 (precision swap, ~5 min)

```bash
# Stop BF16 spine
/scratch/repo/infra/vllm_serve.sh stop

# Start FP8 — uses pre-quantized weights, no on-the-fly quant
PRECISION=fp8 /scratch/repo/infra/vllm_serve.sh spine
tail -f /scratch/vllm-logs/spine.log
```

FP8 weights are ~35GB — faster download than BF16. Healthy in <5 min on the second cold start.

**Acceptance:**
```bash
# Same generation smoke
curl -s http://localhost:8001/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"spine","prompt":"USC Title 26 covers","max_tokens":20,"temperature":0}' \
    | python3 -m json.tool

# Compare VRAM
rocm-smi --showmeminfo vram | tee /scratch/vllm-logs/vram-fp8.txt
diff /scratch/vllm-logs/vram-bf16.txt /scratch/vllm-logs/vram-fp8.txt
```

FP8 should use roughly half the VRAM of BF16. Log the delta.

---

## 6. Smoke #3: APC benchmark (the headline) (15 min)

This is the BiP Post #1 hook. Spine should still be running in FP8 from Smoke #2.

```bash
cd /scratch/repo
source /scratch/venv/bin/activate

# Run the benchmark against the smallest demo bill (HR1, ~125K tokens)
python infra/apc_benchmark.py \
    --bill /scratch/bills/one_big_beautiful_bill_2025_hr1.pdf \
    --endpoint http://localhost:8001/v1 \
    --model spine \
    --target-prefix-tokens 100000 \
    | tee /scratch/vllm-logs/apc-benchmark.json
```

**Acceptance gate:**
- `passes_3x_floor: true` → continue, post BiP #1
- `passes_3x_floor: false` → trigger SGLang escape valve (see docs/escape-valves.md)

The output JSON has the exact numbers for the BiP post:

```json
{
  "prefix_tokens": 100123,
  "request_a_cold": {"ttft_ms": <X>, "total_ms": <Y>},
  "request_b_warm": {"ttft_ms": <Z>, "total_ms": ...},
  "speedup_ttft": <ratio>
}
```

Save those numbers. They go into the BiP #1 post.

---

## 7. USC LMDB smoke (5 min)

```bash
cd /scratch/repo
source /scratch/venv/bin/activate

python3 -c "
import lmdb, orjson, time
env = lmdb.open('/scratch/usc/usc.lmdb', readonly=True, lock=False, subdir=True)
with env.begin() as txn:
    print('Total entries:', txn.stat()['entries'])
    # Hot-cache lookup latency
    iters = 10000
    t0 = time.perf_counter()
    for _ in range(iters):
        txn.get(b'26:401')
    dt = time.perf_counter() - t0
    print(f'{iters:,} lookups in {dt:.3f}s = {dt*1e6/iters:.2f} us/lookup')
    # Sample fetch
    print('---')
    rec = orjson.loads(txn.get(b'26:401'))
    print('26:401 heading:', rec['heading'])
    print('text length :', len(rec['text']))
"
```

**Acceptance:** 60,187 entries, <50µs/lookup, real text returned for 26:401.

---

## 8. Idle watchdog (1 min, then leave running)

In a fresh tmux pane:

```bash
tmux new -s monitor
/scratch/repo/infra/monitor.sh   # watch-only mode
# Ctrl+B, D to detach. Reattach with: tmux attach -t monitor
```

This is now your low-budget meter. If you walk away from the build for >5 minutes,
this prints a loud warning. Pass `--auto-stop` if you want auto-power-off
(don't do this until you've verified the warning logic looks right).

---

## 9. BiP Post #1 — publish (15 min, off-cloud work)

Open `docs/build-in-public-posts.md` in the repo. Section "Post #1 — Day 1".

Fill in the placeholders with the actual numbers from `/scratch/vllm-logs/apc-benchmark.json`:
- `[X]` = `request_a_cold.ttft_ms`
- `[Y]` = `request_b_warm.ttft_ms`
- `[Z]` = `speedup_ttft`
- `[REPO_URL]` = `https://github.com/banksythequantLab/amd-hackathon-bill-analyzer`

Optional: terminal screenshot of the JSON output. Even better: an asciinema clip.

Post on X + LinkedIn. Tag `@AMDDeveloper @lablabai`. Hashtag `#AMDDevHackathon`.

---

## 10. End-of-day shutdown discipline

If you're done for the day:

```bash
# Stop vLLM (frees GPU but keeps weights in HF cache for next time)
/scratch/repo/infra/vllm_serve.sh stop
```

Then in the AMD UI, **power off the droplet**. Snapshot before destroying if
you want to skip re-downloading weights tomorrow (saves ~$2 in download time).

If you're continuing tomorrow without rebooting: leave vLLM stopped, `monitor.sh`
running. The instance idles at "GPU 0% but instance billed" — that's still $1.99/hr.
**Power off if you're walking away for >2 hours.**

---

## Day 1 success criteria (mark Done in Notion)

- [ ] Instance spun, rocminfo confirms MI300X
- [ ] Smoke #1: spine BF16 healthy, generation works, VRAM logged
- [ ] Smoke #2: spine FP8 healthy, VRAM delta vs BF16 logged
- [ ] Smoke #3: APC benchmark produces JSON, speedup ≥ 3x
- [ ] USC LMDB <50µs lookups confirmed on instance
- [ ] monitor.sh running in tmux
- [ ] BiP Post #1 published with real numbers

If APC speedup ≥ 5x: **strong day, on track for ambitious 14-agent scope.**
If 3x ≤ speedup < 5x: **OK, but the demo punchline is weaker — workshop the BiP framing.**
If speedup < 3x: **fire SGLang escape valve, do not pass go.**
