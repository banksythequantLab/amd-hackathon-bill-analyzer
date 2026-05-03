#!/usr/bin/env bash
# vllm_serve.sh — launch the three-Qwen serving stack on a single MI300X
#
# Usage:
#   ./infra/vllm_serve.sh <command>
#
# Commands:
#   spine        Launch only the long-context spine (Qwen3.6-35B-A3B)
#   vision       Launch only the vision pre-processor (Qwen3-VL-8B-Thinking)
#   reasoner     Launch only the reasoner (Qwen3-32B)
#   all          Launch all three (default)
#   stop         Kill all running vLLM processes started by this script
#   status       Show what's running on which port
#
# Environment overrides:
#   PRECISION           bf16 | fp8       Default: fp8 for spine + vision
#   PRECISION_REASONER  bf16 | fp8       Default: bf16 for reasoner (full-precision citations)
#   MAX_LEN             max-model-len    Default: 262144 (Qwen3.6 native context)
#   GPU_MEM_UTIL        0.0-1.0          Default: 0.92
#
# Why three separate vLLM processes:
#   - Different precision per model (BF16 reasoner, FP8 spine + vision)
#   - Independent prefix cache per model (APC keyed per-model anyway)
#   - Independent restart for the BF16<->FP8 demo swap
#
# Port allocation:
#   8001 = spine        (Qwen3.6-35B-A3B)
#   8002 = vision       (Qwen3-VL-8B-Thinking)
#   8003 = reasoner     (Qwen3-32B)
#
# Day 1 TODO before first launch:
#   - Resolve EXACT HuggingFace ids for SPINE_MODEL and VISION_MODEL via Context7
#   - Confirm FP8 weights exist on HF (else use --quantization fp8 for on-the-fly)
#   - Re-test if Qwen3.6 endpoint name is "Qwen3.6" or some variant

set -euo pipefail

# --- Config ----------------------------------------------------------------
# CONFIRM these HF ids on Day 1 — Qwen3.6 was released ~April 2026 and
# the canonical id may differ from this guess. Context7 vLLM docs first.
SPINE_MODEL="${SPINE_MODEL:-Qwen/Qwen3.6-35B-A3B-Instruct}"
VISION_MODEL="${VISION_MODEL:-Qwen/Qwen3-VL-8B-Thinking}"
REASONER_MODEL="${REASONER_MODEL:-Qwen/Qwen3-32B}"

PRECISION="${PRECISION:-fp8}"
PRECISION_REASONER="${PRECISION_REASONER:-bf16}"
MAX_LEN="${MAX_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

LOG_DIR="${LOG_DIR:-/scratch/vllm-logs}"
PID_DIR="${PID_DIR:-/scratch/vllm-pids}"
mkdir -p "$LOG_DIR" "$PID_DIR"

# --- Helpers ---------------------------------------------------------------

quant_flag() {
    case "$1" in
        fp8)  echo "--quantization fp8" ;;
        bf16) echo "" ;;
        *)    echo "ERROR: unknown precision='$1' (expected fp8|bf16)" >&2; exit 1 ;;
    esac
}

dtype_flag() {
    case "$1" in
        fp8)  echo "--dtype auto" ;;
        bf16) echo "--dtype bfloat16" ;;
    esac
}

launch() {
    local name="$1" model="$2" port="$3" precision="$4" extra="${5:-}"
    local log_file="$LOG_DIR/$name.log"
    local pid_file="$PID_DIR/$name.pid"

    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "[$name] already running (pid $(cat "$pid_file"))"
        return 0
    fi

    echo "[$name] launching $model on port $port (precision=$precision)"
    nohup vllm serve "$model" \
        --host 0.0.0.0 \
        --port "$port" \
        --served-model-name "$name" \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --enable-prefix-caching \
        --trust-remote-code \
        $(dtype_flag "$precision") $(quant_flag "$precision") $extra \
        > "$log_file" 2>&1 &
    echo $! > "$pid_file"
    echo "[$name] pid=$(cat "$pid_file")  log=$log_file"
}

stop_one() {
    local name="$1"
    local pid_file="$PID_DIR/$name.pid"
    if [[ -f "$pid_file" ]]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[$name] killing pid $pid"
            kill "$pid"
            sleep 2
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pid_file"
    fi
}

status() {
    for name in spine vision reasoner; do
        local pid_file="$PID_DIR/$name.pid"
        if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
            local pid
            pid=$(cat "$pid_file")
            echo "[$name] RUNNING pid=$pid"
        else
            echo "[$name] not running"
        fi
    done
    echo
    echo "--- ROCm GPU status ---"
    rocm-smi --showuse --showmemuse 2>/dev/null || echo "(rocm-smi not available)"
}

wait_healthy() {
    local port="$1" name="$2" max_wait=600
    echo "[$name] waiting for /health on port $port (up to ${max_wait}s)..."
    local elapsed=0
    while ! curl -sf "http://localhost:$port/health" >/dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [[ $elapsed -ge $max_wait ]]; then
            echo "[$name] FAILED to become healthy in ${max_wait}s — see $LOG_DIR/$name.log"
            return 1
        fi
    done
    echo "[$name] healthy after ${elapsed}s"
}

# --- Commands --------------------------------------------------------------

cmd="${1:-all}"

case "$cmd" in
    spine)
        launch spine "$SPINE_MODEL" 8001 "$PRECISION"
        wait_healthy 8001 spine
        ;;
    vision)
        # VL-8B is small — keep at FP8 even if reasoner runs BF16
        launch vision "$VISION_MODEL" 8002 fp8 "--limit-mm-per-prompt image=10"
        wait_healthy 8002 vision
        ;;
    reasoner)
        # Reasoner runs BF16 by default to preserve citation/math fidelity.
        # The demo swaps to FP8 by setting PRECISION_REASONER=fp8.
        launch reasoner "$REASONER_MODEL" 8003 "$PRECISION_REASONER"
        wait_healthy 8003 reasoner
        ;;
    all)
        # Launch order matters: spine first (biggest KV-cache footprint),
        # then vision, then reasoner.
        launch spine "$SPINE_MODEL" 8001 "$PRECISION"
        wait_healthy 8001 spine || exit 1

        launch vision "$VISION_MODEL" 8002 fp8 "--limit-mm-per-prompt image=10"
        wait_healthy 8002 vision || exit 1

        launch reasoner "$REASONER_MODEL" 8003 "$PRECISION_REASONER"
        wait_healthy 8003 reasoner || exit 1

        echo
        echo "=== All three Qwen endpoints up ==="
        status
        ;;
    stop)
        stop_one spine
        stop_one vision
        stop_one reasoner
        echo "All vLLM processes stopped."
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {spine|vision|reasoner|all|stop|status}" >&2
        exit 1
        ;;
esac
