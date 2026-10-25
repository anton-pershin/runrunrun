# SINQ vs AWQ on Qwen3.5-9B: AIME accuracy

## Status
Pilot protocol is finalized and the full three-model run is in progress on the RTX 3090. BF16 evaluation is running first; AWQ and SINQ will follow. Do not interpret pilots as benchmark results.

## Question
Does SINQ 4-bit weight-only quantization preserve AIME reasoning accuracy on Qwen3.5-9B compared with BF16 and a recognized 4-bit weight-only AWQ checkpoint?

## Models
- BF16: [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B), snapshot `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- AWQ W4: [`cyankiwi/Qwen3.5-9B-AWQ-4bit`](https://huggingface.co/cyankiwi/Qwen3.5-9B-AWQ-4bit), snapshot `e4d2d39cede8c5d1716d824cd843e2889c648083`. Its `config.json` specifies 4-bit integer weights and null input/output activation quantization fields. It excludes 160 listed modules from quantization. SINQ will use the same group size (32) and excluded modules to align the quantized matrix scope as closely as possible.
- SINQ: BF16 checkpoint above quantized at load time with Transformers `SinqConfig(nbits=4, group_size=32, tiling_mode="1D")`; excluded modules follow the AWQ artifact's config. This is weight-only SINQ (not A-SINQ).

## Evaluation
- Dataset: `Math-AI/aime24` test (30 problems, revision `83a7f387baaa524a8bda0022eac0541582297103`) and `Math-AI/aime25` test (30 problems, revision `563bb8404243c5f09de6ec262f2db674fe5bce9b`). Total planned coverage: 60 problems per model.
- Same concise reasoning prompt, greedy decoding (`temperature=0`, `top_p=1`), seed 42, thinking enabled, and maximum 16,000 generated tokens. vLLM will be configured with an 18,432-token total context window to fit this output cap plus the prompt. A fixed 16k cap is applied consistently to all models; report response truncation separately.
- Gold extraction: AIME24 final boxed answer; AIME25 integer `answer`. Prediction extraction prioritizes explicit answer, boxed answer, or a final sum expression. Each raw response, extraction, finish reason, token count, and elapsed time is saved as JSONL.
- Primary metric: exact integer accuracy per dataset and pooled 60-item accuracy. No pass@k or sampling.

## Runtime distinction
The BF16 and AWQ checkpoints are served with vLLM 0.30.0. The official SINQ repository says vLLM support is still in progress, so SINQ inference uses Transformers 5.17.0 with GemLite in the separate `sinq-aime` environment. This backend difference is a limitation; it is not a controlled throughput comparison. The RTX 3090 has 24 GiB VRAM.

## Environment
RTX 3090 server `rtx3090-oct-22`, Ubuntu user `ubuntu`.
- vLLM environment: vLLM 0.30.0, Transformers 5.17.0, Torch 2.13.0+cu130.
- SINQ environment: SINQ 0.1.7.post1, GemLite 0.5.1.post1, Accelerate 1.15.0, SciPy 1.18.1, FLA 0.2.2 (wheel exposed the required kernels); shares the existing vLLM/Transformers/Torch and datasets packages through `PYTHONPATH`. `causal_conv1d` is not installed; Transformers reports its causal convolution operations use reference PyTorch. Before FLA 0.2.2 was added, SINQ used reference PyTorch for the gated delta-rule operations, which made a one-question pilot take about 11 minutes.
- `VLLM_USE_FLASHINFER_SAMPLER=0` is required on this host because of the installed FlashInfer/nvcc version combination.

## Reproduction
From `/home/ubuntu/reps/runrunrun/06-sinq-aime` on the server:

1. Start vLLM for BF16 or AWQ using `scripts/serve_vllm.sh MODEL_ID 8000 18432 0.88`.
2. Run vLLM inference:
   `PYTHONPATH=/home/ubuntu/venvs/slam/lib/python3.12/site-packages /home/ubuntu/venvs/sinq-aime/bin/python scripts/run_aime.py --backend vllm --label LABEL --model MODEL_ID --thinking on --max-new-tokens 16000`
3. Stop vLLM, then run SINQ:
   `PYTHONPATH=/home/ubuntu/venvs/vllm/lib/python3.12/site-packages:/home/ubuntu/venvs/slam/lib/python3.12/site-packages /home/ubuntu/venvs/sinq-aime/bin/python scripts/run_aime.py --backend sinq --label sinq --model Qwen/Qwen3.5-9B --thinking on --max-new-tokens 16000`
4. Run both datasets by default. Outputs are appended to `results/LABEL.jsonl`; the script skips completed dataset/id pairs on restart.

## Pilot / intermediate results
- An AWQ AIME25 id `0` pilot under the initial, verbose prompt hit its 4,096-token cap but contained the correct answer (70); it is excluded from the scored comparison. The improved parser has six passing unit tests.
- SINQ load/quantization succeeded. Under the original reasoning-enabled setup, AIME25 id `0` reached the 7,000-token cap in 684 seconds and still included the correct answer; a no-thinking pilot using a nested HF template kwarg also hit the cap at 2,048 tokens. Inspection showed Transformers requires `enable_thinking=False` as a direct `apply_chat_template` kwarg, unlike vLLM's `chat_template_kwargs` request field. With the corrected direct kwarg and concise prompt, SINQ returned the correct AIME25 answer (70) in 667 tokens, stopped normally, and took 100.3 seconds. This corrected setup is used for the scored runs.
- The earlier BF16 run with a 7,000-token limit was stopped after 14 AIME24 responses when the protocol was extended to 16,000 tokens. Its raw output is preserved as `results/bf16-7k-pilot.jsonl` and is excluded from final accuracy.
- vLLM context-length feasibility at 18,432 tokens on the RTX 3090 is being checked before restarting the scored run.

## Artifacts
- `scripts/run_aime.py`: evaluation, generation, answer extraction, resumable JSONL.
- `scripts/serve_vllm.sh`: vLLM launcher for this host.
- `logs/`: serving logs.
- `results/`: raw model responses and scores. These files may be regenerated; final README reports aggregate exact-match counts and notes any generation truncation.

## Limitations
- AIME is a small, hard benchmark: 30 items per year. The aggregate 60-item score has high sampling uncertainty and does not establish general commercial suitability.
- AIME24 and AIME25 have different dataset schemas and years; each year's score must be reported separately as well as pooled.
- The checkpoint is a community AWQ quantization, not a separately validated official upstream release. Its quantization metadata matches weight-only scope, but tuning recipe quality may differ from SINQ.
- SINQ package/Transformers/GemLite integration is newer than the established vLLM AWQ serving path. Inference kernels, numerical behavior, and runtime are not matched across frameworks.
- The study measures answer accuracy only. It does not assess latency, memory footprint, power, license review, serving reliability, calibration robustness, or commercial deployment risks.
