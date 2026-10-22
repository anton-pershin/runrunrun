# Qwen-3 8B

## Links

**Huggingface:** https://huggingface.co/Qwen/Qwen3-8B

**Blog:** https://qwenlm.github.io/blog/qwen3/

## Hardware

* vCPU 8
* RAM 32 GB
* HDD 240 GB
* NVidia RTX 4090 (24 GB HBM, bandwidth 1 TB/s, FP16 @ 82 TFLOPS)

## Inference performance

See `benchmark_vllm.sh` for settings. The same docker has been used as for Qwen-3.5 9B

| Input len | Output len | Max num seqs (batch size) | Number of prompts | Throughput (RPS) | Throughput (total tokens per sec) | Output throughput (output tokens per sec)
|-|-|-|-|-|-|-|
| 512 | 128 | 4 | 64 | 1.4 | 902 | 181 |
| 2048 | 512 | 4 | 64 | 0.33 | 855 | 171 |
| 16000 | 1024 | 4 | 64 | 0.04 | 692 | 41.6 |

**Issues**
- Default context length (40960) does not fit with default GPU memory utilization so I halved it to 20480. This leads to 1.15x concurrency so that's the limit

