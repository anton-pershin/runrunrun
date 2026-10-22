#!/bin/bash

# vLLM Benchmark Script for Jamba2 Model
# This script runs performance benchmarks using vllm bench with random sequences

set -e

# Configuration
MODEL_NAME="ai21labs/AI21-Jamba-Reasoning-3B"
CONTAINER_IMAGE="sha256:1f112bd2489a9f606e44ea3b23e8787a8a32b7175441f19a16b43fdc88394490"
BENCHMARK_OUTPUT_DIR="./benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Default benchmark parameters
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-128}
NUM_PROMPTS=${NUM_PROMPTS:-100}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}

echo "=== vLLM Benchmark Script ==="
echo "Model: $MODEL_NAME"
echo "Input Length: $INPUT_LEN"
echo "Output Length: $OUTPUT_LEN"
echo "Number of Prompts: $NUM_PROMPTS"
echo "Batch Size: $BATCH_SIZE"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Timestamp: $TIMESTAMP"
echo "================================"

# Create output directory
mkdir -p "$BENCHMARK_OUTPUT_DIR"

# Run benchmark in Docker container
echo "Starting benchmark..."
sudo docker run --gpus all --rm \
    -v "$PWD/$BENCHMARK_OUTPUT_DIR:/app/results" \
    "$CONTAINER_IMAGE" \
    vllm bench throughput \
    --model "$MODEL_NAME" \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --max-num-seqs "$BATCH_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --mamba-ssm-cache-dtype float32 \
    --output-json "/app/results/benchmark_${TIMESTAMP}.json"

echo "Benchmark completed!"
echo "Results saved to: $BENCHMARK_OUTPUT_DIR/benchmark_${TIMESTAMP}.json"

# Display summary if jq is available
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
