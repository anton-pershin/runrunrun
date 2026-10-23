# Huginn vs dense Qwen3 at matched forward-pass compute

Status: completed as a pilot with follow-up trace analysis; this identifies a likely cause for the 4-step failure, not a general result about looped models.

## Question

At approximately matched model-forward FLOPs per generated token, how do Huginn-0125 and dense Qwen3-8B compare on exact accuracy, per-example end-to-end generation latency, and peak GPU memory on one RTX 3090?

## Comparison design

- Huginn-0125: bf16, `num_steps=4`. The model card estimates materialized parameters as `2B + 1.5B × num_steps`, or about 8B at four steps.
- Qwen3-8B: bf16, one ordinary forward pass per generated token, about 8B parameters.
- The models ran on the same RTX 3090 and PyTorch build with greedy decoding and the same generated arithmetic examples/token cap, but required different Transformers versions (Huginn's custom model code fails under 5.17). The models also have different training, architecture, and tokenizer; this is a checkpoint comparison, not a controlled causal claim about looping.
- This is a small proof-of-concept task (seeded 3-digit addition), not a general capability benchmark. The metric is exact integer accuracy.

## Reproduction

Runs were made on `rtx3090-oct-22` (RTX 3090, 24 GB) using two separate environments because Huginn's custom model code fails under Transformers 5.17:

- Qwen3: `~/venvs/vllm` (PyTorch 2.13.0+cu130, Transformers 5.17.0, tokenizers 0.23.2)
- Huginn: `~/venvs/huginn-cdr` (Transformers 4.44.2; PyTorch is exposed from the existing vLLM environment through `vllm_shared.pth`)

The Huginn environment is host-specific; do not recreate it by blindly installing a second PyTorch build. Model revisions are pinned in `study.py`. The Huginn checkpoint was removed from HF cache after the run to restore disk space; it will need to be downloaded again.

```bash
cd ~/reps/runrunrun/04-huginn-equal-compute
source ~/venvs/vllm/bin/activate
python study.py qwen3 --n 20 --seed 20261022 --out results
source ~/venvs/huginn-cdr/bin/activate
python study.py huginn --n 20 --seed 20261022 --out results
```

The script writes per-example JSON under `results/`; do not commit model weights. It records the pinned revision, exact answers, token counts, per-example timing, and PyTorch peak memory.

## Pilot results

RTX 3090, bf16, greedy decoding, 20 identical seeded addition prompts, maximum 16 generated tokens. Accuracy is exact integer match.

- **Huginn-0125, 4 recurrent steps:** 0/20 correct; mean generation E2E 0.521 s/example; mean output 10.25 tokens; peak PyTorch memory 13.40 GiB allocated / 13.44 GiB reserved. Model load took 98.91 s because its checkpoint was downloaded during this cold run.
- **Qwen3-8B:** 20/20 correct; mean generation E2E 0.749 s/example (**1.34 examples/s**); mean output 4.50 tokens; peak PyTorch memory 15.27 GiB allocated / 15.43 GiB reserved. Model load took 4.53 s from cache.

These numbers do **not** show that the looped architecture is more accurate or faster. The arithmetic prompt is a poor fit for this pretrained Huginn checkpoint: on the first three cases it answered `796`, `$12$`, and `$1000$` for gold answers `1257`, `1082`, and `1250`; it often repeats part of the prompt or returns an unrelated number. Since output lengths differ and Huginn produced no correct answers, the generation latency is not an equal-work quality comparison. The cold Huginn load time is also not comparable with Qwen's cached load time.

## Trace diagnosis: recurrence depth was too low

I reran the same 20 prompts with the same Huginn checkpoint and generation settings at `num_steps=32`, then compared its decoded outputs. This isolates recurrence depth within Huginn more directly than the original cross-model comparison.

- At **4 steps**, Huginn scored **0/20**. Typical outputs copied an operand (`796` for `796 + 461`), gave unrelated short numbers (`$12$`, `$1000$`), or began a verbal explanation and stopped before calculating (`The sum ... is 149 +`).
- At **32 steps**, Huginn scored **19/20**. The answers usually state the operation and correct sum, e.g. `The sum of 796 and 461 is 1257.` The one error was `160 + 303 = 313` instead of `463`.
- Mean generation time rose from **0.521 s/example** at 4 steps to **4.480 s/example** at 32 steps (8.60×); peak PyTorch allocated memory rose from **13.40 GiB** to **13.97 GiB**.

There was also an **evaluation bug** in the original scoring code: the regex rejected an integer immediately followed by a period. So a correct output ending `... is 1257.` was parsed as `461`, the preceding operand. The raw 32-step traces were correct on the first five examples, but the original scorer incorrectly called all five wrong. I fixed the parser to take the final integer and recalculated the saved JSON. This bug did not explain the 4-step failure: none of its 20 outputs contained the correct sum as the final integer.

**Best-supported explanation:** four recurrent steps gave this checkpoint too little latent computation for this task; more recurrence let it produce a valid arithmetic continuation. This matches the model card's warning that fewer than four steps gives coarse answers and that benchmark quality can improve with more depth, sometimes up to about 64 steps. The 32-step result supports that explanation for this pilot, but does not establish a general causal claim or explain what happens inside the hidden states. I inspected generated text, not per-iteration latent activations/logits.

The original comparison of Qwen3-8B against 4-step Huginn was therefore misleading as an accuracy comparison. At 32 steps Huginn is much closer to Qwen on this tiny task (19/20 vs 20/20), but it uses more recurrent compute and was about 6× slower per example than Qwen's measured 0.749 s on this setup. This is only a small pilot, not evidence of a broad accuracy/efficiency frontier.

Trace and per-example evidence is stored in `results/huginn-4steps.json` and `results/huginn-32steps.json`. The updated `study.py` parser accepts a `--num-steps` override; the recorded depth-32 run used:

```bash
python study.py huginn --n 20 --seed 20261022 --num-steps 32 --out results
```

## Conclusion and limitations

The initial four-step setting was a compute-matched choice based on the model card's approximate materialized-parameter estimate (`2B + 1.5B × 4 ≈ 8B`)—not a sensible quality operating point for this task. At 32 steps, that estimate implies about 50B materialized parameters, so it is no longer compute-matched to Qwen3-8B. A follow-up should test several depths (for example 4, 8, 16, 32, 64) on a benchmark that fits Huginn's trained behavior, validate answer extraction, and compare accuracy/latency/VRAM at equal measured or carefully estimated compute. PyTorch memory stats exclude non-PyTorch allocations; latency is single-request, not serving throughput.
