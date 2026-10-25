#!/usr/bin/env bash
set -euo pipefail
MODEL_ID="${1:?usage: serve_vllm.sh MODEL_ID PORT MAX_MODEL_LEN GPU_MEMORY_UTILIZATION}"
PORT="${2:-8000}"
MAX_MODEL_LEN="${3:-8192}"
GPU_MEMORY_UTILIZATION="${4:-0.85}"
export VLLM_USE_FLASHINFER_SAMPLER=0
exec /home/ubuntu/venvs/vllm/bin/vllm serve "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --seed 42 \
  --enforce-eager
