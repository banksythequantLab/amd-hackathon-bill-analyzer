#!/usr/bin/env bash
set -e

echo "=== Pre-download config + tokenizer ==="
docker run --rm \
    -v /root/hf-cache:/root/.cache/huggingface \
    --entrypoint python3 \
    vllm/vllm-openai-rocm:v0.17.1 \
    -c "from huggingface_hub import snapshot_download; p = snapshot_download('Qwen/Qwen3-30B-A3B-Instruct-2507-FP8', allow_patterns=['*.json', '*.txt', 'tokenizer*']); print('Pre-DL OK:', p)" 2>&1 | tail -3

echo
echo "=== Load config via transformers ==="
docker run --rm \
    -v /root/hf-cache:/root/.cache/huggingface \
    --entrypoint python3 \
    vllm/vllm-openai-rocm:v0.17.1 \
    -c "from transformers import AutoConfig; cfg = AutoConfig.from_pretrained('Qwen/Qwen3-30B-A3B-Instruct-2507-FP8', trust_remote_code=True); print('arch:', cfg.architectures); print('model_type:', cfg.model_type); print('max_position_embeddings:', cfg.max_position_embeddings)" 2>&1 | tail -10

echo
echo "=== Container HF env ==="
docker run --rm --network host --entrypoint bash vllm/vllm-openai-rocm:v0.17.1 \
    -c 'echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"; echo "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"; curl -sI -m 5 https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8/resolve/main/config.json 2>&1 | head -2' 2>&1
