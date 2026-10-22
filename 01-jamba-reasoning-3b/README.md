# AI21-Jamba-Reasoning-3B

## Links

Huggingface: https://huggingface.co/ai21labs/AI21-Jamba-Reasoning-3B

GGUF: https://huggingface.co/ai21labs/AI21-Jamba-Reasoning-3B-GGUF#quickstart

Blog: https://www.ai21.com/blog/introducing-jamba-reasoning-3B

## Inference performance

See `benchmark_vllm.sh` for settings.

| Input len | Output len | Max seq len | Throughput (RPS) | Throughput (tokens per sec) |
| 512 | 128 | 8 | 3.5 | 2221 |
| 32000 | 1024 | 8 | 0.2 | 6379 |
