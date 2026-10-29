# Trajectory stability: findings and interpretation

## Status and scope

The historical audit is complete and independently reproduced. The balanced 24-episode pilot is complete, with no missing or duplicate jobs and all episode replays valid. All three arms scored 0/8 successes. This report separates measured behavior from proposed mechanisms. It does not claim an accuracy improvement.

The intervention uses only ordinary observations, the task, tool descriptions, and an explicit active decision record. There is no online planner, applicable-action filter, reference plan, or exact-distance signal. Sampling remains at temperature 0.7 and top-p 0.9 in every new arm, including record updates. The historical sampling settings were not saved, so the historical and new runs are not a matched causal comparison.

## What the original four runs establish

All 96 episodes and 2,061 action events replay exactly. Final success is 88/96: 45/48 in baseline and 43/48 in the disturbance condition. All 24 task/condition groups have four distinct action/status traces, but 17 groups always succeed. Only seven groups mix success and failure. Sequence divergence is therefore not the main problem by itself.

The exhaustive offline evaluation rules out environmental dead ends. Every reachable state has a finite path to the corresponding goal; every legal action has a one-action inverse. The eight unsuccessful final states are still 4–15 legal actions from a goal.

Before first goal attainment, the eight unsuccessful runs have:

- 132 successfully applied actions that reduce exact goal distance;
- 108 successfully applied actions that increase it;
- 127 precondition failures;
- 170 successful transitions back to a previously seen state.

Thus 45% of applied actions in those unsuccessful traces move away from a goal, compared with 13.24% in successful traces. These are descriptive associations, not proof that a particular history or hidden state causes failure. Seven failures end at the usage limit. The remaining failure falsely treats `(on b6 b3)` as the required `(on b3 b6)`.

There is also harmless diversity: 44 successful runs take shortest legal paths, and eight of those paths differ from the reference sequence.

Termination is a separate failure source. Of 88 goal-attaining episodes, 73 continue afterward. All 51 malformed calls occur after goal attainment. The new executor stops on observed success in every arm; any benefit from that shared rule must not be attributed to the record.

## A more useful mechanism than sampling alone

The agent does not sample a complete plan once and then execute it. It repeatedly interprets the observation, chooses a subgoal and operation, receives an outcome, and makes another decision. A stochastic difference can change both the world and the evidence available to the next decision. Returning to the same world state does not return a history-based agent to the same input.

This gives a candidate feedback mechanism: an early interpretation or commitment changes later choices; those choices create new observations; later decisions can retain, revise, or forget the commitment. A finite budget turns reversible detours and repeated rejected actions into outcome failures. This explains how divergence can matter without implying that every different path is wrong.

The data support the behavioral parts: reversible cycling, repeated rejection, relation reversal, and incorrect stopping. They do not yet establish history-dependent policy attractors, neural-state drift, or the causal role of a record.

## What the record exposes

The record is a declared external working state, not a transcript of hidden neural reasoning. Its versions link observations, confirmed facts, subgoals, commitments, chosen actions, and outcomes.

In pilot-02, `task-005__baseline__record__r00`, decision 2 copies the true fact `(on b2 b3)` but declares `Unstack b3 from b2 before moving b2`. It then selects `(unstack b5 b2)`, which fails because b5 is not clear. The record exposes both a reversed derived relation and an action/subgoal mismatch.

In `task-005__baseline__record__r01`, decision 2 declares `Place b4 on b2` and selects `(stack b4 b2)`. The observation says b4 is held, but b5 is still on b2. The action agrees with the declared subgoal, yet is premature.

Therefore, correct copied facts do not imply correct interpretation, feasible subgoals, or suitable actions. Stabilizing an incorrect commitment can stabilize failure. The offline fact checker must not be presented as a validator of strategy quality.

These examples are inspectable diagnostics, not evidence that the declared subgoal caused the selected action. The saved record timelines and raw request logs permit that question to be tested separately.

## Completed pilot: negative result with reliability limits

The final `../results/decision-record/pilot-02/summary.json` contains all 24 planned episodes, with no missing, duplicate, or unexpected keys. Every replay passes. All returned API model labels are `qwen/qwen3.5-9b`; this does not establish a fixed backend revision. Logs contain 821 requests and 814 responses. The sent sampling settings are identical across arms: temperature 0.7, top-p 0.9, and an 8,192-token completion cap.

| Arm | Successes | Action attempts | Precondition failures | Repeated failed state/action pairs | Successful state revisits |
|---|---:|---:|---:|---:|---:|
| Full history | 0/8 | 353 | 157 | 70 | 86 |
| Fresh complete state | 0/8 | 300 | 129 | 60 | 65 |
| Active record | 0/8 | 76 | 31 | 2 | 7 |

Each arm has four baseline and four disturbance-condition episodes. Neither condition has a success in this pilot. A consistent failure is not useful determinism. The record arm's smaller loop count is not evidence of stabilization: it stops much earlier, and two episodes execute no action.

The terminations are:

- History: five action-budget limits, one API timeout, one completion-token limit, and one empty response.
- Fresh state: four action-budget limits, two API timeouts, and two completion-token limits.
- Record: four record-writing API timeouts, one record schema error, one record-writing completion-token limit, one empty action response, and one record returned as ordinary text instead of the required tool call. The last case has the generic stored label `model_stopped`; it is not evidence of deliberate abandonment.

All record episodes therefore end on a protocol, format, or resource boundary rather than reaching the action budget. Nevertheless, the observed record-conditioned actions contain planning errors before those boundaries. There are 77 saved records and 673 fact claims: no unsupported or currently false fact claims, no objective drift, three last-operation mismatches, and 29 precondition failures following records that pass all these fact/metadata checks.

This is a negative result for the implemented intervention under this protocol. It does not establish that all external records are ineffective, or that history compression is the main cause of failure. Both complete-state reset and full history also fail. Frequent timeouts and format failures limit the efficacy comparison, while nine action-budget failures show that not every failure is infrastructure-only.

The gap between these results and the historical 88/96 successes is unresolved. Tasks were selected for historical difficulty; prompts, action-catalog delivery, stopping, and explicitly fixed sampling also differ. The model label does not certify a constant provider backend. It would be incorrect to attribute that gap to the record or to claim a provider bug from these data alone.

The 96-episode expansion is not justified. First establish a working, repeatable baseline with the original interface and the new interface on the same tasks and current endpoint, preserving sampling and budget settings. A controlled local endpoint is an option if API behavior cannot be made reliable; hidden states or log probabilities are not yet necessary to identify this problem.

## Next control to test, not a demonstrated solution

The target should be stable task commitments and successful outcomes, not identical text or one fixed action sequence. Non-zero sampling can still choose among compatible ways to carry out the same commitment. There is no guarantee of identical outcomes while incompatible decisions retain non-zero probability.

A stronger transferable record would separate four things:

1. Observation-backed facts, with source IDs and explicit uncertainty.
2. The current commitment and the observation that justified it.
3. The proposed operation, its expected observable result, and any assumptions not yet supported by observations.
4. Persistent counterexamples: a rejected operation, the observed reason, and what evidence must change before trying it again. An execution timeout or transient failure must not be treated as evidence that an operation is infeasible.

Before execution, a model-based consistency check would compare the proposal with the cited observations and counterexamples. It would not query the simulator, enumerate legal actions, supply a replacement plan, or evaluate future world states. Commitments would persist until an observed result justifies revision. Failed checks would become visible record revisions rather than silently changing the task interpretation.

This is a hypothesis for the next intervention. The present pilot implements the simpler record, not this additional check. A consistency check can itself be wrong and adds computation. Its effect must be compared with a compute-matched control.

The most informative causal experiment is a matched-checkpoint test. Hold the current observation, task, tools, action prompt, and sampling settings fixed. Compare continuations with the original record, no record, an observation-only record, and a record whose unsupported subgoal has been revised using only the same observations. Measure action/subgoal agreement, rejected-operation repetition, and eventual goal success. Use several checkpoint types and sample independent continuations. Do not select only examples where a revised record already works.

Only after that test should a larger repeated-rollout evaluation or an AppWorld transfer claim be attempted. In a partially observed environment, records must use ordinary tool evidence and uncertainty, not a complete hidden world state.

## Verification and review

The final unit run passed 14 tests; two paid integration probes were skipped by default. All 24 pilot episodes replay correctly, with complete expected job coverage. All seven episodes that contain a valid saved record have a readable timeline; an episode can save a record and then fail before executing its action.

The exhaustive historical evaluator and its separate artifact verifier were re-executed successfully from repository-local source copies in scratch: 96 trajectories, 2,061 transitions, graph completeness and reversibility, and exact-distance equations checked. `git diff --check` passed.

A separate read-only scientific review found no material error in the exploratory interpretation, historical artifact figures, smoke metrics, or the two record/action examples. That review inspected a 20-episode checkpoint, not the final 24-episode dataset. The parent process subsequently checked final coverage, counts, replay results, failure categories, and timelines. The review also noted that error-terminated episodes must not be described as demonstrated planning failures; this report preserves that distinction.

## Reproducible evidence

- `PROTOCOL.md`: intervention boundary, fixed settings, arm definitions, and separately preserved protocol revisions.
- `README.md`: reproduction commands and historical examples.
- `historical_audit.py`: replay, outcome diversity, repeated errors, cycles, and stopping behavior.
- `exact-offline/README.md`: independently exercised exhaustive graph and distance evaluation.
- `../results/decision-record/historical-exact/`: exact summaries and annotated original traces.
- `../results/decision-record/pilot-01/ABORTED.md`: why the earlier partial pilot is not an efficacy comparison.
- `../results/decision-record/pilot-02/`: raw API logs, source snapshots, manifest, completed episodes, and readable record timelines.

No packages were installed. No original trajectory file was changed. No commit or push was made for this follow-up.
