# Qwen-3.5 9B

## Links

**Huggingface:** https://huggingface.co/Qwen/Qwen3.5-9B

**Blog:** https://qwen.ai/blog?id=qwen3.5

## Hardware

* vCPU 8
* RAM 32 GB
* HDD 240 GB
* NVidia RTX 4090 (24 GB HBM, bandwidth 1 TB/s, FP16 @ 82 TFLOPS)

## Inference performance

See `benchmark_vllm.sh` for settings. Note that the default context length 262144 fits the memory with the default GPU memory utilization. I checked whether the configured context length affects the throughput just in case (it doesn't).

| Input len | Output len | Max num seqs (batch size) | Number of prompts | Throughput (RPS) | Throughput (tokens per sec) |
|-|-|-|-|-|-|
| 512 | 128 | 4 | 64 | 0.56 | 360 | 71.8 |
| 2048 | 512 | 4 | 64 | 0.24 | 622 | 125 |
| 16000 | 1024 | 4 | 1 | 0.011 | 191 | 11.4 |

**Issues**
- Given that the response time for one request with 16000/1024 setting was about 88 seconds, I decided not to run 32000/1024 test and run 16000/1024 for one request only
