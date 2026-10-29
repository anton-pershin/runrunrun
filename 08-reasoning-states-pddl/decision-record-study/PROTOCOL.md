# Decision-record stabilization: exploratory protocol

## Question and boundary

Can an explicit, observation-grounded working record reduce harmful instability in a stochastic tool-using agent? Stable outcomes, not identical action strings, are the main target. This is an exploratory continuation of study 08, not a powered evaluation.

The policy can use the task, tool descriptions, the full fixed action vocabulary, and observations returned to an ordinary agent. It cannot query applicable actions, action preconditions outside tool results, a planner, reference plans, or future states. PDDL internals can be used offline for measurement only. The executor applies actions and returns observations, as in the original study.

The record is a declared working-state representation, not a measurement of hidden neural states or a request for chain-of-thought.

## Arms

- `history`: native tool conversation with the complete action/observation history.
- `reset`: a fresh decision context with the complete current state and goal, in sorted canonical form. This is a privileged complete-state diagnostic, not a proposed solution for partially observable applications.
- `record`: first update a compact structured record from the previous record and latest observation. Then choose an action in a fresh context using the active record, goal, and latest observation. Save all record versions for analysis, but do not include their full history in the action context. Confirmed facts must cite observation IDs. Fields cover uncertainty, subgoal, last operation/outcome, and a strategy revision condition. Do not supply a symbolic plan or filter actions.

All arms use the same action prompt, native action tool, model, explicit temperature 0.7 and top-p 0.9, and 50 attempted-action budget. No seed is supplied. The record writer uses the same temperature and top-p. Thus the sampling algorithm/settings are fixed, but the record arm has an extra model call per action and a different prompt; it is not a compute-matched comparison. The historical runs have unrecorded sampling settings and cannot be used as a matched causal baseline.

All arms receive the full fixed grounded action catalog upfront. This avoids unrecorded catalog calls and is an explicit interface change from the historical runner. The executor stops immediately when the observed goal is true in every arm. Do not ask the model to infer successful termination after that point. Record model refusal or premature text termination as an unsuccessful episode, not as an execution exception.

## Execution

1. Unit-test record structure, prompt boundaries, sampling parameters, goal termination, and replay verification using the existing Python environment. Do not install packages.
2. Run a three-arm API smoke test on task-005, one baseline rollout per arm, in a separate output namespace. Verify returned API model IDs, raw messages, record parsing, and executor replay.
3. Run a first-stage pilot: tasks 005 and 011; two independent repeats per arm; baseline and one-transient-failure conditions. Total: 24 episodes. This scope was reduced before the pilot because smoke record updates can take roughly a minute each, and the endpoint sometimes returns empty assistant messages. These tasks are selected because historical runs include failures; they are not a representative accuracy benchmark. Preserve every episode, including API errors, parse errors, empty responses, and failed goals. A 96-episode expansion (tasks 005, 007, 011, and 012; four repeats) is deferred until this first stage establishes a useful signal and acceptable interface reliability. Do not call the first stage a robustness estimate.
4. Use a fixed shuffled job order and bounded concurrency. Persist each request before sending, each response immediately, and each episode on completion. Never overwrite a previous output directory. Resume only unfinished episodes and preserve previous attempts separately.
5. In the disturbed condition, inject one failure into the first legal action after at least two attempted actions, matching the original rule. This executor-only intervention is not disclosed in advance to the policy. Record whether it occurred.

## Measurements and verification

- Episode counts by task, repeat, arm, and condition; goal success and clean termination separately.
- Task/condition cells with mixed outcomes across repeats. Report uncertainty and small-sample limits.
- Action counts, precondition/format errors, repeated failed state-action pairs, and successful transitions that return to an earlier world state. Do not label every state revisit harmful: rearrangement can be necessary.
- First-goal action index and post-goal actions (must be zero in this runner).
- First-action agreement and action/state prefixes as descriptive diversity metrics, not primary success measures.
- Record confirmed-fact claims checked against cited observed states; distinguish stale but once-supported claims from unsupported claims. This is offline measurement only and does not correct the record during execution.
- Replay every action with the recorded disturbance. Verify initial state, every before/after state, status, final state, goal, and action count. Check the action contexts contain no reference plans or state-dependent legal-action lists.
- Save exact sampling parameters, prompts/messages, tool schemas, response model IDs, usage, latency, dependency versions, code hashes, and Git state. Do not save credentials or provider-specific hidden reasoning fields.

## Smoke-test revision before the pilot

Smoke-01 used a 2,048-token completion cap. The provider reported reasoning-token usage and returned length-truncated responses without tool calls. Preserve smoke-01 as an engineering check, not an efficacy result. Increase the cap to 8,192 tokens equally in all arms for smoke-02 and the pilot. Do not change temperature, top-p, or the model's thinking mode. Classify length truncation as an explicit completion-limit error, not voluntary model termination. Requests and returned usage establish what was sent and reported; they do not independently prove that the provider honors every sampling setting.

Smoke-02 and later results preserve complete source snapshots. Before the first-stage pilot, review fixes made the disturbance parser consistent with execution, added independent schedule/budget/termination checks, and separated record-writer refusals and empty responses. No empty-response retries are used. Record claims are measured but never repaired by the evaluator. The action sampler and record sampler both retain temperature 0.7 and top-p 0.9.

## Protocol revision after empty-response probes

Pilot-01 was stopped after 12 of 24 episodes completed because native history calls were dominated by empty assistant responses. Its incomplete logs and episode files are retained; do not compare its arm success rates. An opt-in probe of the original Pydantic AI runner also stopped after one attempted action and exhausted its output retries.

At one saved task-011 checkpoint, four `tool_choice=auto` calls and four `tool_choice=required` calls all returned no tool call. In a second probe at that checkpoint, four more auto calls returned no tool call, while four calls with an explicit final user request returned tool calls. These are small interface diagnostics at one checkpoint, not evidence of planning improvement or proof of a provider defect.

For pilot-02, every action context in every arm ends with the same explicit request: `The latest observation is above. Choose your next execute_action call.` This is unconditional and contains no task solution, legal-action filter, or failure-specific advice. Keep tool_choice auto, temperature 0.7, top-p 0.9, the 8,192-token cap, and the 50-action budget. Use six concurrent episodes. Retain all failures and empty responses; do not retry them silently. The 24-episode scope is unchanged.

Preserve native provider continuation fields in memory if supplied, but redact them from both request and response logs. The original-agent protocol probe did not report such fields in its assistant messages, so missing reasoning-field replay is not an established explanation for the observed empty responses.

## Interpretation limits and follow-ups

A record improvement would support a useful external-memory intervention, not prove that the record faithfully exposes neural state. Extra computation, structured prompting, and history compression remain confounds. A record failure can expose factual corruption, stale commitments, or correct records followed by poor actions.

If the pilot is promising, add compute-matched record-observation-only and unstructured-summary controls, matched-world-state continuation experiments, and held-out tasks before making causal or transfer claims. A later AppWorld test must construct records from ordinary observations without access to complete simulator state. No local model deployment is required for this pilot.
