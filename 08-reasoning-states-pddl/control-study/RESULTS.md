# Operational commitment: completed exploratory evaluation

## Result

No GPU hardware was used. Both evaluations ran one episode at a time with a 512 MiB virtual-memory limit, temperature 0.7, top-p 0.9, and identical tool-plan prompts and transport settings across arms. Every completed episode was retained. Both datasets contain exactly their 16 expected distinct jobs and all stored replay checks pass.

| Dataset | Commit goals | Recede goals | Commit model calls | Recede model calls |
|---|---:|---:|---:|---:|
| Screening-selected tasks 005/011, new repeats | 6/8 | 4/8 | 45 | 99 |
| Tasks 007/012, not used in plan screening | 7/8 | 7/8 | 27 | 120 |

Each dataset includes two tasks, two conditions, two independent repeats, and two modes. Screening episodes are not included. These small, sequentially collected datasets are not a powered robustness estimate. The second dataset is held out from prompt/controller screening, not from knowledge of the original benchmark.

The first outcome advantage did not replicate on the held-out task subset. Commitment used fewer model calls while matching accuracy there. This is a useful execution-efficiency result, not demonstrated general improvement in determinism or planning accuracy.

## Mechanism and limits

Both modes sample an ordered operational plan. Commit executes successive operations from that plan, advancing its cursor on tool success. Recede samples again after each successful operation, with the previous remaining plan supplied as context. Both permit one retry after reported transient failure, and revise after observed semantic rejection. Both retain failure evidence. No simulator-derived policy filtering, reference plans, exact distances, or solution caches are used.

Commitment reduces the number of opportunities to change a previously sampled intention. It also reduces exposure to model-output failures. These effects are not separated by this comparison. Model call count is not a token-cost estimate because response lengths can differ.

All eight unsuccessful evaluation episodes ended on unusable model output: four completion-token-limit failures and four responses without the required submit_plan call. None ended at the action budget. This prevents interpreting the success difference as better planning. Planning mistakes did occur before output failure or eventual success.

Commit had three repeated rejected state/action pairs and three state revisits in the first dataset; recede had none. In the second dataset, commit had one repeated rejection and no state revisits; recede had no repeated rejection and two state revisits. These small, exposure-dependent counts do not show a consistent reduction in harmful decision errors.

Across paired repeats within task/condition, mixed outcomes occurred in zero of four commit cells versus two of four recede cells in the first dataset, and one of four cells in each mode in the second dataset. A cell where both episodes fail must not be called useful stability. Complete action-trace identity is not a requirement and was not established by the intervention.

## Transfer argument

The controller itself is domain-independent: retain an operational commitment, advance using tool feedback, and revise on contradiction. The Blocks World operation descriptions and executor are task-specific. A practical tool environment would need explicit operation success, transient-failure, and invalidation semantics. Tool success alone can be too weak: plans must be invalidated when expected postconditions fail or external state changes. Irreversible or high-risk operations require appropriate safety and approval checks. This study does not establish transfer to AppWorld.

## Conclusion and next experiment

The existing API is usable without GPU deployment. Sampled operational commitment is implemented and exercised, preserves non-zero sampling, and substantially reduces model calls in both comparisons. It has not yet demonstrated a repeatable accuracy or determinism improvement over the matched baseline.

The next informative test is to separate output reliability from decision stability: apply an identical, logged bounded output-repair policy in both arms, then measure matched-checkpoint continuation diversity and independent repeated success. Repair must not execute operations before a valid output is received, invent actions, drop adverse attempts, or change sampling settings. Full episode retries must not be silently substituted for failed samples. Do not claim the user's accuracy-stability goal is achieved on current evidence.

## Artifacts

- `controller.py`: generic bounded controller.
- `transport.py`: public tool-plan requests and response parsing.
- `experiment.py`: matched harness, executor, replay and source snapshots.
- `PROTOCOL.md`: freeze and sequential evaluation decisions.
- `../results/control/tool-plan-eval-01/`: 16 episodes, request logs, source snapshots and summaries.
- `../results/control/tool-plan-heldout-01/`: 16 episodes under unchanged controller and prompt.

No original trajectory files were modified. No packages installed or commits made for this follow-up.
