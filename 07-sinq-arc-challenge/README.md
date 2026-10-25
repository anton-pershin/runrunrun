# SINQ vs AWQ on Qwen3.5-9B: ARC-Challenge comparison

## Status
The full 300-item ARC-Challenge comparison is complete for all three models. Results use the same deterministic first 300 items from the pinned `ARC-Challenge` test split and the same letter-only, no-thinking protocol.

## Research question
Does SINQ 4-bit weight-only quantization preserve Qwen3.5-9B accuracy on a non-mathematical science-reasoning benchmark compared with BF16 and a recognized 4-bit AWQ weight-only checkpoint?

## Models
- **BF16:** [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B), revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- **AWQ:** [`cyankiwi/Qwen3.5-9B-AWQ-4bit`](https://huggingface.co/cyankiwi/Qwen3.5-9B-AWQ-4bit), revision `e4d2d39cede8c5d1716d824cd843e2889c648083`. The live config specifies 4-bit weights and null input/output activation quantization fields. It is weight-only, not W-A.
- **SINQ:** BF16 source quantized at load time with Transformers `SinqConfig(nbits=4, group_size=32, tiling_mode="1D")`. The same 160 module names excluded by the approved AWQ recipe were excluded from SINQ conversion. SINQ is weight-only (`sinq`), not activation-aware ASINQ.

## Dataset and protocol
- Dataset: [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc), config `ARC-Challenge`, test split, revision `210d026faf9955653af8916fad021475a3f00453`.
- Full test split: 1,172 items. This experiment evaluates the deterministic first 300 items in dataset order. The selected IDs are retained in the JSONL outputs.
- Prompt: answer the four-option science question with only `A`, `B`, `C`, or `D`; no explanation.
- Decoding: temperature 0, top-p 1, seed 42, Qwen thinking disabled, maximum 16 generated tokens.
- Scoring: exact letter match. Unparsed outputs count incorrect. All 900 records have two generated tokens and zero truncation. All three models evaluated the same 300 IDs.

## Results
- **BF16:** 278/300 = **92.67%**; 1 unparsed output; mean request time 0.1375 s; aggregate request time 41.26 s.
- **AWQ 4-bit:** 278/300 = **92.67%**; 1 unparsed output; mean request time 0.1389 s; aggregate request time 41.68 s.
- **SINQ 4-bit:** 280/300 = **93.33%**; 1 unparsed output; mean request time 0.4698 s; aggregate request time 140.93 s.

The raw output records are in:
- `results/bf16-arc300.jsonl`
- `results/awq-arc300.jsonl`
- `results/sinq-arc300.jsonl`

Paired outcomes on the same 300 items:
- BF16 vs AWQ: 276 both correct; 2 BF16-only; 2 AWQ-only.
- BF16 vs SINQ: 277 both correct; 1 BF16-only; 3 SINQ-only.
- AWQ vs SINQ: 276 both correct; 2 AWQ-only; 4 SINQ-only.

## Interpretation
On this screening set, SINQ did not reduce accuracy relative to BF16 or AWQ. It scored two more correct answers than BF16 and AWQ, but the difference is small: 0.67 percentage points, or two questions. It should not be treated as evidence that SINQ is intrinsically more accurate; sampling uncertainty, benchmark composition, and the no-reasoning protocol matter.

The results support **considering SINQ for further commercial testing**, not immediate production adoption. The main measured disadvantage in this setup is serving path and latency: BF16 and AWQ used vLLM, while SINQ used Transformers + GemLite because vLLM SINQ support was not available in this environment. SINQ request time was about 3.4x BF16/AWQ in aggregate, although its timing includes first-use kernel effects and is not a controlled production-throughput benchmark.

Before a commercial decision, measure: production-shaped throughput and concurrency, cold-start time, peak memory, power/cost per token, long-context behavior, target-domain accuracy, robustness, safety, and the licenses of Qwen, AWQ checkpoint, SINQ, GemLite, and the serving stack.

## Reproduction
On `rtx3090-oct-22`, study directory `/home/ubuntu/reps/runrunrun/06-sinq-amc23`:

1. Start vLLM with `scripts/serve_vllm.sh MODEL_ID 8000 18432 0.88`.
2. Run BF16 or AWQ:
   `PYTHONPATH=/home/ubuntu/venvs/vllm/lib/python3.12/site-packages:/home/ubuntu/venvs/slam/lib/python3.12/site-packages /home/ubuntu/venvs/sinq-aime/bin/python scripts/run_benchmark.py --backend vllm --label LABEL --model MODEL_ID --datasets arc_challenge --thinking off --max-new-tokens 16 --limit 300`
3. Stop vLLM and run SINQ:
   `PYTHONPATH=/home/ubuntu/venvs/vllm/lib/python3.12/site-packages:/home/ubuntu/venvs/slam/lib/python3.12/site-packages /home/ubuntu/venvs/sinq-aime/bin/python scripts/run_benchmark.py --backend sinq --label sinq-arc300 --model Qwen/Qwen3.5-9B --datasets arc_challenge --thinking off --max-new-tokens 16 --limit 300`

## Software and hardware
- RTX 3090 24 GiB.
- BF16/AWQ: vLLM 0.30.0, Transformers 5.17.0, Torch 2.13.0+cu130.
- SINQ: separate environment with SINQ 0.1.7.post1, GemLite 0.5.1.post1, Accelerate 1.15.0, SciPy 1.18.1; it shares the required model/runtime packages through `PYTHONPATH`.
- SINQ used the reference PyTorch causal-convolution fallback because `causal_conv1d` was not installed.

## Commercial-use limits
This is a 300-item, single-benchmark accuracy comparison. It does not establish commercial suitability. The result is a screening signal: SINQ preserved accuracy in this test, but production latency, cost, reliability, licensing, and target-domain quality remain open.
