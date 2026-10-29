# Sampled operational commitment

## Status

The generic controller and matched experiment harness are implemented. All 18 offline tests pass. Both 16-episode comparisons are complete and replay-valid. On screening-selected tasks, commit solved 6/8 versus recede 4/8, using 45 versus 99 model calls. On tasks 007/012, neither used in plan screening, both solved 7/8; commit used 27 calls versus recede 120. The accuracy advantage did not replicate. Fewer calls are demonstrated; improved determinism or planning accuracy is not. See `RESULTS.md`. No GPU hardware was used.

## Recovery without hardware

`original-defaults-02` ran the unchanged historical runner on task-005: goal reached in 32 attempts, no error, 34 responses with no length truncation. `original-defaults-01` was a logging serialization error before any model request; it is not a model failure.

`settings-probe-01` replayed the same post-catalog request with original defaults, explicit temperature/top-p, an explicit 8,192-token cap, and both overrides. All four responses returned one tool call, without truncation. The sampling settings and cap alone therefore do not reproduce the earlier problem.

`tool-plan-screen-01` uses a submit_plan tool instead of plain JSON output and a concise tool-oriented prompt, at temperature 0.7 and top-p 0.9, provider-default thinking, and the same 8,192-token cap. It solved task-005 in 25 attempts with six plan calls and task-011 in 17 attempts with two plan calls. All replay checks passed. Output format and prompt changed together, so these results do not isolate the cause of recovery.

`tool-plan-eval-01` is a separate frozen matched evaluation: two tasks, both conditions, two repeats, two arms, one episode at a time under a 512 MiB virtual-memory limit. See PROTOCOL.md. Do not merge screening results into efficacy counts.

## Mechanism

Sample an ordered operational plan at temperature 0.7 and top-p 0.9. Retain its cursor after tool success rather than sampling a new action each step. Retry one reported transient failure. Revise the plan on an observed rejection or exhaustion, using the ordinary observation and failure memory. The matched `recede` arm uses the same plan prompt and sampler but requests a new plan after each successful operation.

The controller does not import PDDL, filter applicable actions, inspect reference plans, or search future states. The experiment executor uses PDDL to apply attempted operations; the verifier replays them offline. Both arms receive the same ordinary operation descriptions. Operational plans are explicit outputs, not hidden reasoning.

## Memory fault and recovery

The interrupted implementation had an unbounded loop: it ignored budgets and repeatedly appended plans when a transient failure was not handled. A memory-limited reproduction failed with MemoryError. The corrected controller checks action and plan budgets and implements bounded transient retry. The formerly failing test and additional budget regression checks pass. This is evidence of a plausible cause of the earlier RAM spike, not a recovered operating-system OOM record.

Subsequent experiments ran one at a time with `ulimit -v 524288` and wall-time limits. Final system checks showed 12 GiB available. No local model was loaded.

## Live screening

Results are under `../results/control/`:

- `commit-screen-01`: default provider thinking setting; two completed episodes. Task-011 solved in 14 actions from one sampled plan. Task-005 produced no plan before consuming the 8,192-token completion limit.
- `paired-screen-02`: requested disabled thinking through `chat_template_kwargs`; four completed episodes, no goals. Six requests returned six responses. Four responses consumed the completion limit; the two usable plans contained rejected operations. The provider reported 8,192 reasoning tokens and no visible output on each length-truncated response. The requested flag therefore did not reliably produce non-thinking completions.
- `commit-screen-03`: increased completion cap to 32,768. Incomplete: two requests were logged, one connection error was recorded as an episode, and the other request has no terminal log event after the command transport timed out. Preserve it as interrupted, not a completed experiment.
- `commit-screen-04`: requested disabled thinking through top-level `enable_thinking`; two episodes, both completion-limit failures with all 8,192 completion tokens reported as reasoning tokens.

All completed episode replay checks passed. These early probes did not establish that the endpoint honors requested thinking controls or a fixed backend revision. They also did not establish that commitment solves planning errors. The initial request for a controlled deployment was premature: the subsequent unchanged-baseline and tool-plan tests above recovered usable API behavior without hardware.

## Reproduction

Start in the parent study directory using its existing environment. No installation is required for the harness.

```sh
ulimit -v 524288
timeout 20s .venv/bin/python -m unittest discover -s control-study -v
```

API execution uses `HERMES_CUSTOM_CAILA_IO_API_KEY` and refuses an existing result directory. API logs remove private reasoning fields and think-tag content. Sources, protocol, sent settings, model labels, usage, errors and episode replay results are saved with each run. `baseline_probe.py` contains tested request/context/scoring helpers from the interrupted diagnosis; it is not a complete CLI runner.

```sh
ulimit -v 524288
timeout 340s .venv/bin/python control-study/experiment.py \
  --tasks task-005 task-011 --repeats 1 --modes commit \
  --concurrency 1 --timeout 120 --output results/control/new-screen
```

`PROTOCOL.md` defines the matched comparison. Independent repeated evaluation is now running using the recovered tool-call interface, with a separate source snapshot and result namespace. Larger batches must preserve all adverse results and distinguish action-budget failures from interrupted or unusable model responses.
