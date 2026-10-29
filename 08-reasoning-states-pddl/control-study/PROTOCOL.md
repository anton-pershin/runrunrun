# Sampled plan commitment: exploratory protocol

## Target and boundary

Find a controller that improves repeatability of successful task completion at non-zero temperature. Identical action traces are secondary. A stable failure is not an improvement. The controller should apply to ordinary tool APIs, not require a symbolic environment.

The proposed controller samples a bounded operational plan, keeps a cursor into that plan, and revises it only when an actual tool observation contradicts it or the plan is exhausted. Tool success advances the cursor without asking the model to choose again. A reported transient failure permits one retry. A semantic rejection invalidates the remaining plan and triggers replanning using the actual observation and retained failure evidence. The controller does not decide which replacement action is legal or useful.

There are no reference plans, simulator calls for action selection, legal-action filters, offline distance estimates in prompts, or stored solutions reused across runs. The executor alone applies an attempted operation. Evaluation may replay transitions or inspect goal satisfaction. Model-generated action plans are public operational artifacts, not hidden chain-of-thought.

## Matched comparison

Both arms use the same model, API transport, operational plan output schema, ordinary tool documentation, current observation, task goal, and failure memory. Both retain temperature 0.7 and top-p 0.9. No seed is set. Every episode starts with empty memory. Native provider continuation fields are not needed because each planning request is fresh.

- `recede`: request a plan and execute its first operation. Request a new plan at the next decision, supplying the previous remaining plan as ordinary context.
- `commit`: request a plan and execute successive operations without resampling after each successful tool observation. Request a new plan after an observed semantic failure or exhaustion. Supply the previous remaining plan and failure memory when replanning.

This changes the times at which the policy is sampled, not the temperature or token sampling procedure. It is not a compute-matched experiment: receding plans require more calls. Report model calls, tokens, latency, and early protocol failures separately. Extra compute is not to be silently converted into a budget advantage for either arm.

The same natural-language tool contract is supplied to both arms. It describes the API operations and feedback, not a planner or a state-dependent set of applicable actions. This explicitly differs from the earlier catalog-only pilot; it must not be called a reproduction of the historical protocol.

## Execution and evidence

1. Use bounded live probes to establish a working response interface. Preserve sent settings, reported model, response status, usage, and all errors. Treat provider thinking-mode selection, if needed, as an explicit protocol setting used equally in both arms. Never store private reasoning text or credentials.
2. Test generic controller behavior before model runs. The controller must not import PDDL or inspect world facts to choose operations.
3. Screen on tasks 005 and 011. Preserve screening data and any adaptations separately.
4. Freeze the selected prompt, parser, transport, controller, budgets, and sampling before independent evaluation. Run both arms in shuffled balanced order on selected tasks with independent repeats. Report task selection and distinguish screening tasks from held-out tasks.
5. Include baseline and one-transient-failure conditions if the working interface supports a useful first comparison. The executor injects once into the first legal action after two attempts, never supplies eligibility to the controller, and records whether injection occurred.
6. Stop on observed complete goal satisfaction. Use the same action and request budgets. No silent retry of malformed responses; any bounded network retry policy must be identical across arms and logged.

## API recovery and evaluation freeze

The unchanged historical runner solved task-005 in 32 actions with provider defaults. At the same saved post-catalog checkpoint, four request variants (original defaults, explicit temperature/top-p, explicit 8,192-token cap, and both) all returned usable tool calls. Thus neither the cap nor explicit sampling alone is an established cause of the earlier failures.

The operational-plan interface was then changed from plain JSON completion to a `submit_plan` tool call, with a concise tool-oriented prompt. The default thinking mode, temperature 0.7, top-p 0.9, and completion cap 8,192 remained. Both screening tasks succeeded (005: 25 attempts and six plan calls; 011: 17 attempts and two plan calls). This changes output format and prompt together; it does not isolate which caused API recovery. It establishes a usable interface for the matched comparison, not a controller effect.

Freeze for `tool-plan-eval-01`: tasks 005 and 011, baseline and one-transient-failure conditions, two independent repeats, both recede and commit modes (16 episodes). Use tool output format, provider-default thinking, explicit temperature 0.7 and top-p 0.9, completion cap 8,192, timeout 120 seconds, no automatic API retries, 50 attempted operations, and one transient retry. Run one episode at a time under a 512 MiB virtual-memory limit. Source snapshots and fixed shuffled order are saved before execution. Screening episodes are excluded from evaluation totals. These are screening-selected tasks, not held-out evidence. Four repeats per task across conditions are not a precise robustness estimate.

## First matched result and held-out check

`tool-plan-eval-01` completed all 16 episodes with valid replay and complete distinct job coverage. Commit solved 6/8 (baseline 4/4, disturbance 2/4); recede solved 4/8 (baseline 3/4, disturbance 1/4). Commit used 45 planning calls versus 99 for recede. Both commit failures were completion truncations. Recede had two completion truncations and two responses without the required submit_plan call. Thus fewer opportunities for output failure can explain part or all of the outcome advantage; this is not evidence yet of improved planning quality or action-sequence determinism.

Without changing controller, prompt, parsing, or sampling, run `tool-plan-heldout-01`: tasks 007 and 012, both conditions, two repeats, both modes (16 episodes), with the same one-process resource limits. These tasks were not used in operational-plan screening or tuning. They are from the known original benchmark, not a previously unseen domain. Keep their results separate and report all failures.

## Agreement-controller follow-up

The held-out commitment check completed with 7/8 goals in each arm. Reduced call count replicated, but the accuracy advantage did not. Do not treat the user's stability goal as achieved.

Preserve `consensus-screen-01`: three independently sampled plans vote on their first operation, then the winning plan is committed. It solved 1/2 screening episodes; correlated bad suffixes and repeated rejections remain. Preserve `review-screen-01`: draft plus a separately sampled contract-review call before commitment. It solved 1/2, with the other reaching the action limit. Neither screen establishes efficacy.

`step-vote-screen-01` instead votes again after each observed success, never executing the unchecked suffix. Task 011 solved in 19 actions and 60 calls. The whole screening process hit its 340-second outer limit during task 005; its request log is preserved as incomplete evidence, not counted as a completed episode or rerun in place.

Freeze `step-vote-eval-01` before new requests: tasks 005 and 011, both conditions, two independent repeats, modes recede and step_vote (16 episodes). The same tool-plan prompts, model, temperature 0.7, top-p 0.9, completion cap 8192, and per-call timeout 60 seconds apply. No seed. Each step_vote decision uses three independent requests with identical visible context, requires support from at least two for the first operation, then executes only that operation. The majority operation is selected by count, with lexical tie breaking; the remaining plan is selected by full-list count then lexical order only within that first-operation group. Re-vote without execution if there is no quorum. Invalid responses are retained and count against the three requests. Failed semantic actions are supplied as observations, not filtered using world rules. Report this output-recovery behavior as a confound relative to recede.

All controllers have 50 executed-attempt limits. Recede has at most 51 requests; step_vote has at most 51 vote rounds / 153 requests. This is extra-compute inference, not compute-matched. Report actual calls, latency, tokens, failed output counts and no-quorum rounds, alongside success, mixed outcomes, rejected repeats and cycles. Use one process, 512 MiB, a new result directory, all expected jobs and source snapshots. Complete and retain adverse episodes. No across-run solution cache, legal-action filter, simulated rollout, or privileged evaluator feedback is permitted.

## Acceptance and limitations

A useful result requires successful independent repeated episodes and lower harmful variability or fewer regressions/rejected repeats without reducing accuracy. Report successes by task, condition, arm and repeat; first-action/trace diversity; calls; invalid attempts; repeated rejected state-action pairs; state-changing cycles; plan revisions; and infrastructure/format failures.

Do not claim transfer from PDDL results alone. The transfer argument is structural: persist a sampled operational commitment, advance only on tool evidence, and revise on contradiction. General applications need operation-specific success and retry semantics, and may need safety checks or confirmation before irreversible operations. Blindly replaying a long plan is not appropriate when tool success does not establish that its expected condition holds.

If no reliable API baseline or useful controller can be demonstrated, preserve the negative result and ask for a controlled local model deployment rather than claiming a mechanism works.
