#!/usr/bin/env bash
# vllm_serve_docker.sh — launch the three-Qwen serving stack on MI300X
# using the pre-pulled vllm/vllm-openai-rocm:v0.17.1 image.
#
# Usage:
#   ./infra/vllm_serve_docker.sh <command>
#
# Commands:
#   spine        Launch only the long-context spine (Qwen3.6-35B-A3B)
#   vision       Launch only the vision pre-processor (Qwen3-VL-8B-Thinking)
#   reasoner     Launch only the reasoner (Qwen3-32B)
#   all          Launch all three (default)
#   stop         Stop and remove all vllm containers
#   status       Show container + GPU status
#   logs <name>  Tail a container log (spine|vision|reasoner)
#
# Environment overrides:
#   PRECISION           bf16 | fp8       Default: fp8 for spine + vision
#   PRECISION_REASONER  bf16 | fp8       Default: bf16 for reasoner
#   MAX_LEN             max-model-len    Default: 262144 (Qwen3.6 native)
#   GPU_MEM_UTIL        0.0-1.0          Default: 0.92
#   HF_CACHE            host path        Default: /root/hf-cache (persistent)
#   IMG                 docker image     Default: vllm/vllm-openai-rocm:v0.17.1
#
# Port allocation:
#   8001 = spine
#   8002 = vision
#   8003 = reasoner

set -euo pipefail

IMG="${IMG:-vllm/vllm-openai-rocm:v0.17.1}"
HF_CACHE="${HF_CACHE:-/root/hf-cache}"
PRECISION="${PRECISION:-fp8}"
PRECISION_REASONER="${PRECISION_REASONER:-bf16}"
MAX_LEN="${MAX_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

mkdir -p "$HF_CACHE"

resolve_model() {
    local role="$1" precision="$2"
    case "$role:$precision" in
        # Spine: pivoted from Qwen3.6-35B-A3B (vLLM 0.17.1's transformers
        # 4.57.6 doesn't know qwen3_5_moe yet) to Qwen3-30B-A3B-Instruct-2507.
        # Same A3B MoE topology, 30B/3B-active, 256K context, FP8 prequantized.
        spine:bf16)     echo "Qwen/Qwen3-30B-A3B-Instruct-2507" ;;
        spine:fp8)      echo "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8" ;;
        vision:bf16)    echo "Qwen/Qwen3-VL-8B-Thinking" ;;
        vision:fp8)     echo "Qwen/Qwen3-VL-8B-Thinking-FP8" ;;
        reasoner:bf16)  echo "Qwen/Qwen3-32B" ;;
        reasoner:fp8)   echo "Qwen/Qwen3-32B-FP8" ;;
        *) echo "ERROR: unknown role:precision '$role:$precision'" >&2; exit 1 ;;
    esac
}

dtype_flag() {
    case "$1" in
        fp8)  echo "auto" ;;
        bf16) echo "bfloat16" ;;
    esac
}

launch() {
    local name="$1" role="$2" port="$3" precision="$4" extra="${5:-}"
    local model
    model=$(resolve_model "$role" "$precision")

    if docker ps --format '{{.Names}}' | grep -qx "vllm-$name"; then
        echo "[$name] already running"
        return 0
    fi

    # Remove any stopped container with the same name
    docker rm -f "vllm-$name" 2>/dev/null || true

    echo "[$name] launching $model on port $port (precision=$precision)"
    # shellcheck disable=SC2086
    docker run -d \
        --name "vllm-$name" \
        --device /dev/kfd \
        --device /dev/dri \
        --group-add video \
        --security-opt seccomp=unconfined \
        --shm-size 16g \
        --ipc host \
        --network host \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -e HF_HUB_ENABLE_HF_TRANSFER=1 \
        -e GLOO_SOCKET_IFNAME=lo \
        -e NCCL_SOCKET_IFNAME=lo \
        -e VLLM_HOST_IP=127.0.0.1 \
        --restart no \
        "$IMG" \
        --model "$model" \
        --host 0.0.0.0 \
        --port "$port" \
        --served-model-name "$name" \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --enable-prefix-caching \
        --trust-remote-code \
        --dtype "$(dtype_flag "$precision")" \
        $extra

    echo "[$name] container id: $(docker ps -qf "name=vllm-$name")"
}

stop_one() {
    local name="$1"
    if docker ps -a --format '{{.Names}}' | grep -qx "vllm-$name"; then
        echo "[$name] stopping + removing container"
        docker rm -f "vllm-$name" >/dev/null
    fi
}

status() {
    echo "--- vLLM containers ---"
    docker ps --filter "name=vllm-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo
    echo "--- ROCm GPU ---"
    rocm-smi --showuse --showmemuse 2>/dev/null | head -25
}

wait_healthy() {
    local port="$1" name="$2" max_wait=900
    echo "[$name] waiting for /health on port $port (up to ${max_wait}s)..."
    local elapsed=0
    while ! curl -sf "http://localhost:$port/health" >/dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [[ $elapsed -ge $max_wait ]]; then
            echo "[$name] FAILED to become healthy in ${max_wait}s"
            echo "Last 30 log lines:"
            docker logs --tail 30 "vllm-$name" 2>&1 || true
            return 1
        fi
        if (( elapsed % 60 == 0 )); then
            echo "  [$name] still waiting (${elapsed}s)..."
        fi
    done
    echo "[$name] healthy after ${elapsed}s"
}

cmd="${1:-all}"

case "$cmd" in
    spine)
        launch spine spine 8001 "$PRECISION"
        wait_healthy 8001 spine
        ;;
    vision)
        launch vision vision 8002 fp8 "--limit-mm-per-prompt image=10"
        wait_healthy 8002 vision
        ;;
    reasoner)
        launch reasoner reasoner 8003 "$PRECISION_REASONER"
        wait_healthy 8003 reasoner
        ;;
    all)
        launch spine spine 8001 "$PRECISION"
        wait_healthy 8001 spine || exit 1
        launch vision vision 8002 fp8 "--limit-mm-per-prompt image=10"
        wait_healthy 8002 vision || exit 1
        launch reasoner reasoner 8003 "$PRECISION_REASONER"
        wait_healthy 8003 reasoner || exit 1
        echo
        status
        ;;
    stop)
        stop_one spine
        stop_one vision
        stop_one reasoner
        ;;
    status)
        status
        ;;
    logs)
        name="${2:-}"
        [[ -z "$name" ]] && { echo "Usage: $0 logs {spine|vision|reasoner}" >&2; exit 1; }
        docker logs -f "vllm-$name"
        ;;
    *)
        echo "Usage: $0 {spine|vision|reasoner|all|stop|status|logs <name>}" >&2
        exit 1
        ;;
esac
