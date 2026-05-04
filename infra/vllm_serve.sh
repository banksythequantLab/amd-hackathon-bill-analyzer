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
# Models verified on Hugging Face Hub, May 4 2026:
#   Qwen/Qwen3.6-35B-A3B           BF16 weights, 2.7M+ downloads
#   Qwen/Qwen3.6-35B-A3B-FP8       FP8 pre-quantized, 2.5M+ downloads
#   Qwen/Qwen3-VL-8B-Thinking      BF16 weights, 721K+ downloads
#   Qwen/Qwen3-VL-8B-Thinking-FP8  FP8 pre-quantized, 31K+ downloads
#   Qwen/Qwen3-32B                 BF16 weights, 4.7M+ downloads
#   Qwen/Qwen3-32B-FP8             FP8 pre-quantized, 277K+ downloads
#
# Pre-quantized FP8 weights mean no on-the-fly quant step — cold start
# saves ~10-15 min vs. quantizing BF16 at launch.

set -euo pipefail

# --- Model selector (BF16 vs pre-quantized FP8) ----------------------------
# When PRECISION=fp8, prefer the pre-quantized HF id to skip on-the-fly quant.

resolve_model() {
    local role="$1" precision="$2"
    case "$role:$precision" in
        spine:bf16)     echo "Qwen/Qwen3.6-35B-A3B" ;;
        spine:fp8)      echo "Qwen/Qwen3.6-35B-A3B-FP8" ;;
        vision:bf16)    echo "Qwen/Qwen3-VL-8B-Thinking" ;;
        vision:fp8)     echo "Qwen/Qwen3-VL-8B-Thinking-FP8" ;;
        reasoner:bf16)  echo "Qwen/Qwen3-32B" ;;
        reasoner:fp8)   echo "Qwen/Qwen3-32B-FP8" ;;
        *) echo "ERROR: unknown role:precision '$role:$precision'" >&2; exit 1 ;;
    esac
}

# --- Config ----------------------------------------------------------------
PRECISION="${PRECISION:-fp8}"
PRECISION_REASONER="${PRECISION_REASONER:-bf16}"
MAX_LEN="${MAX_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

# Default scratch paths — override via env if your instance uses a different mount
LOG_DIR="${LOG_DIR:-/scratch/vllm-logs}"
PID_DIR="${PID_DIR:-/scratch/vllm-pids}"
HF_HOME="${HF_HOME:-/scratch/hf-cache}"
mkdir -p "$LOG_DIR" "$PID_DIR" "$HF_HOME"
export HF_HOME

# --- Helpers ---------------------------------------------------------------

dtype_flag() {
    case "$1" in
        fp8)  echo "--dtype auto" ;;
        bf16) echo "--dtype bfloat16" ;;
        *)    echo "ERROR: unknown precision='$1'" >&2; exit 1 ;;
    esac
}

# Note: with pre-quantized FP8 weights, --quantization is NOT needed.
# vLLM auto-detects from the model config. Only set it for on-the-fly quant.

launch() {
    local name="$1" role="$2" port="$3" precision="$4" extra="${5:-}"
    local model
    model=$(resolve_model "$role" "$precision")
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
        $(dtype_flag "$precision") $extra \
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
    local port="$1" name="$2" max_wait=900
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
        launch spine spine 8001 "$PRECISION"
        wait_healthy 8001 spine
        ;;
    vision)
        # VL-8B is small — keep at FP8 even if reasoner runs BF16
        launch vision vision 8002 fp8 "--limit-mm-per-prompt image=10"
        wait_healthy 8002 vision
        ;;
    reasoner)
        # Reasoner runs BF16 by default to preserve citation/math fidelity.
        # The demo swaps to FP8 by setting PRECISION_REASONER=fp8.
        launch reasoner reasoner 8003 "$PRECISION_REASONER"
        wait_healthy 8003 reasoner
        ;;
    all)
        # Launch order matters: spine first (biggest KV-cache footprint),
        # then vision, then reasoner.
        launch spine spine 8001 "$PRECISION"
        wait_healthy 8001 spine || exit 1

        launch vision vision 8002 fp8 "--limit-mm-per-prompt image=10"
        wait_healthy 8002 vision || exit 1

        launch reasoner reasoner 8003 "$PRECISION_REASONER"
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
