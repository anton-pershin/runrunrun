#!/bin/bash

# vLLM Benchmark Script for Qwen3.5-9B
# Mirrors the structure used for other models in this repo.

set -euo pipefail

# Configuration
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3.5-9B"}
CONTAINER_IMAGE=${CONTAINER_IMAGE:-"qwen_3_5_9b:latest"}
BENCHMARK_OUTPUT_DIR=${BENCHMARK_OUTPUT_DIR:-"./benchmark_results"}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Default benchmark parameters (override via env vars as needed)
# 262144
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-128}
NUM_PROMPTS=${NUM_PROMPTS:-64}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-20480}
DTYPE=${DTYPE:-"bfloat16"}
REASONING_PARSER=${REASONING_PARSER:-"qwen3"}
LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY:-"true"}

echo "=== vLLM Benchmark Script ==="
echo "Model: $MODEL_NAME"
echo "Container Image: $CONTAINER_IMAGE"
echo "Input Length: $INPUT_LEN"
echo "Output Length: $OUTPUT_LEN"
echo "Number of Prompts: $NUM_PROMPTS"
echo "Batch Size: $BATCH_SIZE"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "dtype: $DTYPE"
echo "Reasoning Parser: $REASONING_PARSER"
echo "Language Model Only: $LANGUAGE_MODEL_ONLY"
echo "Timestamp: $TIMESTAMP"
echo "================================"

mkdir -p "$BENCHMARK_OUTPUT_DIR"

echo "Starting benchmark..."
sudo docker run --gpus all --rm \
    -v "$PWD/$BENCHMARK_OUTPUT_DIR:/app/results" \
    "$CONTAINER_IMAGE" \
    vllm bench throughput \
    --model "$MODEL_NAME" \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --max-num-seqs "$BATCH_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype "$DTYPE" \
    --reasoning-parser "$REASONING_PARSER" \
    $( [ "$LANGUAGE_MODEL_ONLY" = "true" ] && echo "--language-model-only" ) \
    --output-json "/app/results/benchmark_${TIMESTAMP}.json"

#sudo docker run --gpus all --rm \
#    -v "$PWD/$BENCHMARK_OUTPUT_DIR:/app/results" \
#    "$CONTAINER_IMAGE" \
#    vllm --version

echo "Benchmark completed!"
echo "Results saved to: $BENCHMARK_OUTPUT_DIR/benchmark_${TIMESTAMP}.json"

if command -v jq &> /dev/null; then
    echo ""
    echo "=== Benchmark Summary ==="
    RESULT_FILE="$BENCHMARK_OUTPUT_DIR/benchmark_${TIMESTAMP}.json"
    if [ -f "$RESULT_FILE" ]; then
        echo "Throughput (tokens/s): $(jq -r '.throughput' "$RESULT_FILE" 2>/dev/null || echo 'N/A')"
        echo "Latency (ms): $(jq -r '.latency' "$RESULT_FILE" 2>/dev/null || echo 'N/A')"
        echo "Total time (s): $(jq -r '.total_time' "$RESULT_FILE" 2>/dev/null || echo 'N/A')"
    fi
else
    echo "Install 'jq' to see benchmark summary: sudo apt-get install jq"
fi
