## 01-jamba-reasoning-3b

Tested Jamba deployment via vLLM and measured throughput.

NOTE: this study followed an obsolete format and should not be used as an example

## 02-qwen-3-5-9b

Tested Qwen3.5-9B deployment via vLLM and measured throughput.

NOTE: this study followed an obsolete format and should not be used as an example

## 03-qwen-3-8b

Tested Qwen3-8B deployment via vLLM and measured throughput.

NOTE: this study followed an obsolete format and should not be used as an example

## 04-huginn-equal-compute

Pilot comparison of Huginn-0125 at four recurrent steps with dense Qwen3-8B at approximately matched forward-pass compute, measuring exact accuracy, E2E generation latency, and peak VRAM on an RTX 3090.

## 05-ouro-26b-eval

Evaluation of ByteDance/Ouro-2.6B on the prior addition control and a deterministic GSM8K subset, with dense Qwen3-8B as reference.

## 06-mk-sh-v-visual-decisions

A three-image local CPU probe of MK-SH-V. It answered a visible-number question correctly, but gave confident wrong answers to an island-location question and a people-counting task. Includes the user-provided images, exact questions, model outputs, limitations, and reproduction notes.

## 07-sinq-arc-challenge

Comparison of Qwen3.5-9B BF16, 4-bit AWQ weight-only, and 4-bit SINQ weight-only models on ARC-Challenge and MMLU-Redux. Includes inline Mermaid quantization schemas and final results.

## 08-reasoning-states-pddl

Interactive Blocks World pilot for action/outcome trajectories: generated and solved 12 PDDL tasks, tested a structured execution wrapper with a state-independent full action catalog, collected reference-controller traces, and ran Pydantic AI tool-agent rollouts for `gpt-6-luna` and `qwen/qwen3.5-9b`. The Qwen action-catalog dataset has 24 episodes, 514 events, and 19 goal completions. Proxy-state analysis remains pending; independent VAL validation remains outstanding.
