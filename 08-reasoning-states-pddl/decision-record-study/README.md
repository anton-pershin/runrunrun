# Decision-record stabilization

This follow-up tests an explicit working-state record as a tool-agent intervention. It does not use a planner, legal-action filtering, reference plans, or symbolic lookahead to select actions. PDDL is the executor and an offline measurement instrument.

The record describes declared working state. It is not a direct observation of neural state or hidden reasoning. The main target is correct, stable outcomes, not identical action sequences.

## Status

The historical audit, both three-arm API smoke tests, and the balanced 24-episode pilot are complete. Pilot-01 was stopped as an interface diagnostic; pilot-02 completed with 0/8 successes in each arm. All 24 episode replays passed, with no missing or duplicate jobs. The record intervention has not demonstrated an accuracy gain. See `REPORT.md` for the results and interpretation, and `PROTOCOL.md` for scope, settings, confounds, and protocol revisions. Do not merge smoke tests or pilot-01 into the pilot-02 efficacy counts.

The successful record smoke episode contains 33 record versions and 251 confirmed-fact claims. Offline checks found no unsupported or currently false claims, no objective drift, and no wrong last-operation fields. It still attempted seven actions with false preconditions. Thus factual corruption of the record is not necessary for action errors. This does not establish whether the model followed its declared strategy or whether the intervention improves outcomes.

A readable version of that episode is in `../results/decision-record/smoke-02/record-timelines/task-005__baseline__record__r00.txt`. It links every active record to the action and observed outcome. The record episode solved the task in 33 actions; the reset smoke episode solved it in 50. The history smoke episode returned an empty response after two actions. These single episodes are not a valid accuracy comparison.

## What the records already make visible

These are exploratory examples from pilot-02, not an efficacy estimate. The confirmed-fact list, objective, and last-operation fields can pass their checks while the proposed subgoal is wrong. The quality checks do not certify a strategy or a subgoal.

- `task-005__baseline__record__r00`, decision 2: observed facts include `(on b2 b3)`, but the record says `Unstack b3 from b2 before moving b2`. This reverses the relation in the observation. The selected action is a different unstack operation, `(unstack b5 b2)`, which fails because b5 is not clear. This record exposes a derived-state interpretation error and an action/subgoal mismatch, not a corrupted copied fact list.
- `task-005__baseline__record__r01`, decision 2: the record says `Place b4 on b2` and the action is `(stack b4 b2)`. The latest observation shows b4 is held, but b5 is still on b2. The action fails because b2 is not clear. Here the selected action agrees with the declared target, but the target is premature.

These cases suggest separating observation copying, derived interpretation, subgoal feasibility, and action/subgoal agreement. They do not prove that the declared subgoal caused the action. A matched-context record intervention is needed to test causal influence.

## Files

- `runner.py`: native OpenAI-compatible tool agent, record writer, append-only API logs, and episode replay checks.
- `analyze.py`: offline outcome, cycle, failure repetition, usage, and record-grounding measurements.
- `test_runner.py`: tests of prompt boundaries, sampling parameters, record use, termination, disturbance scheduling, replay validation, and metric edge cases.
- `historical_audit.py`: complete replay and behavioral audit of the original four Qwen runs.
- `PROTOCOL.md`: question, intervention boundary, arms, scope, revisions, and limitations.
- `../results/decision-record/`: historical audit, raw requests/responses, source snapshots, manifests, episodes, and summaries. Each run has a separate directory. Credentials and provider-specific hidden reasoning fields are not saved.

## Historical findings

The audit replayed all 96 episodes and 2,061 action events with zero mismatches. Goals were true at the final state in 88/96 episodes: 45/48 baseline and 43/48 disturbance-condition episodes. The disturbance actually occurred in 44/48 intended episodes.

There are 24 task/condition groups with four rollouts each. All groups have four distinct action/status traces, but 17 groups succeed in every rollout. Only seven groups have mixed success outcomes. Trace diversity therefore overstates harmful outcome instability.

Before first goal attainment, there were 311 state-changing returns to an earlier complete state across 41 episodes, and 66 repeated failures across 16 episodes. A repeated failure here has the same complete pre-action state, action string, and non-transient failure status. A state return is descriptive: some rearrangement is necessary, so a return alone is not proof of a bad decision.

Termination is a separate source of instability. Of the 88 episodes that reached a goal, 73 continued afterward. There were 187 attempts that began in an already solved state, including all 51 malformed calls in the dataset. Four successful actions in two episodes destroyed an achieved goal; later actions repaired it. Eleven episodes hit a usage-limit exception, including four with a solved final state. One unsolved episode returned normally and falsely claimed success.

Examples use the stored one-based `attempt` number:

- Root run, task-005, baseline: events 13–14 unstack and restack b3 on b1, returning to the event-12 state. The failed `(stack b5 b1)` at event 5 recurs in the same complete pre-action state at 20, 36, and 48. The episode is unsolved at its limit.
- Batch-02, task-004, baseline: the goal is true after event 2. Events 9, 26, and 41 execute `(unstack b1 b5)` and destroy it; 10, 27, and 42 restore it. The episode reaches a usage-limit exception after 49 action attempts.
- Root run, task-012, baseline: the final tool observation says the goal is false. The completion text claims `(on b6 b3)`, but the required relation is `(on b3 b6)`. This is a false completion, not an exception.

The original data do not contain full model messages or explicit sampling settings. These findings establish behavioral patterns, not a causal explanation based on hidden states or message-history drift. Baseline and disturbance episodes are independent stochastic runs, not shared-prefix counterfactuals. Their aggregate success difference is not an isolated estimate of the disturbance effect.

The new runner stops on an observed true goal in every arm. This removes a measured termination problem, but it is a shared executor change, not an improvement caused by the record.

## Exact offline progress check

The separate exhaustive evaluation found no environmental dead ends. Every reachable state has a path to each corresponding task goal, and every legal transition has a one-action inverse. The five-block graph has 866 states; the six-block graph has 7,057. Initial shortest distances for tasks 001–012 are 10, 10, 12, 2, 16, 6, 14, 8, 2, 6, 14, and 14. The supplied reference plans are all shortest.

Before first goal attainment, successful applied actions in the eight unsuccessful runs decreased exact goal distance 132 times and increased it 108 times. Thus 45% of their applied actions moved away from the goal. They also had 127 precondition failures and 170 state-changing returns to earlier states. The corresponding distance-increasing share in the 88 successful runs was 13.24%. These are descriptive associations, not an estimate of a causal effect.

Of the 88 successful runs, 44 followed a shortest legal path, including eight whose sequence differs from the reference. This is direct evidence that action-sequence diversity need not mean a reasoning failure.

See `exact-offline/README.md` for verified reproduction, and `../results/decision-record/historical-exact/` for the generated summaries, exhaustive graphs, distances, and annotated historical traces. Exact distances remain offline measurements and are never shown to the policy.

## Reproduction

Use the existing study environment. No installation is needed. Commands below start in the parent study directory, `08-reasoning-states-pddl`.

```sh
.venv/bin/python -m unittest discover -s decision-record-study -v
.venv/bin/python decision-record-study/historical_audit.py
```

For new API runs, set `HERMES_CUSTOM_CAILA_IO_API_KEY` in the process environment. The runner does not write it. Select a new output directory for every run; existing output directories are refused.

```sh
.venv/bin/python decision-record-study/runner.py \
  --tasks task-005 task-011 --repeats 2 --concurrency 6 \
  --temperature 0.7 --top-p 0.9 \
  --output results/decision-record/pilot-new
.venv/bin/python decision-record-study/analyze.py \
  results/decision-record/pilot-new \
  --output results/decision-record/pilot-new/summary.json
```

The defaults select `qwen/qwen3.5-9b` through Caila, all three arms, both disturbance conditions, and a 50-action budget. The record writer uses the same temperature and top-p as the action selector. It adds a model call before each action; this is not a compute-matched experiment. Model revision and backend sampling implementation are not independently verified by the API model label.
