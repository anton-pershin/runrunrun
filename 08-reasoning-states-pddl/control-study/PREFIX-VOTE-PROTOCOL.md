# Prefix voting: frozen exploratory test

## Hypothesis

Voting only at the start can reduce harmful sensitivity to early sampled decisions without the cost of voting throughout execution. This may fail when late errors dominate or early samples share the same mistake. No improvement is assumed.

## Controller

Use three independent proposals at each of the first five action decisions, execute only the plurality-supported first operation, then observe its actual result. After decision five, use ordinary single-sample receding planning. A semantic rejection consumes a decision; an explicitly transient failure permits one retry of the same operation without another vote. These are action decisions, not five successful state transitions.

One sampling round only per early decision. Ties use the first usable sample in sample order. Thus all-distinct responses fall back to the first usable proposal; this is not majority support and must be counted separately. Invalid samples are logged and count against the round; stop if none is usable. Do not silently retry. The full plan attached to the chosen operation is retained as context but its suffix is not blindly executed.

At most ten extra proposal requests over the same number of single-sample decisions. This does not guarantee lower total episode cost: trajectory lengths and output failures may differ. Three calls may tolerate unusable outputs that terminate a single-response arm; report this confound. Model, prompts, token sampling, and per-request cap remain unchanged. No simulator policy filtering or cross-run solution cache.

## Evaluation

Run a new shuffled balanced batch on tasks 005 and 011 with baseline and one-transient-failure conditions, three independent repeats, modes recede and prefix_vote: 24 episodes. Both arms use temperature 0.7, top-p 0.9, max_tokens 8192, timeout 60 seconds, 50 execution attempts, no seed or automatic network retries. Prefix length five is fixed before requests; do not tune four versus five against these results. Use concurrency one and 512 MiB virtual-memory limit. Preserve every episode and all interruptions.

Primary descriptive robustness measures: outcome disagreement across repeats within each task/condition, separately retaining consistently unsuccessful cells; baseline versus disturbance success rates and paired outcome changes (unseeded rollouts are independent, not matched random trajectories). Report first-action agreement and prefix diversity as behavioral measures, not correctness or robustness proof. Also report success, repeated rejected state/action pairs, state revisits, missing outputs, voting support/no-majority counts, actual requests, tokens and latency. With three repeats per cell these are exploratory estimates, not conclusive robustness certification. Stop early only for a verified implementation fault or resource limit, not adverse outcomes.

The test is limited to the known benchmark tasks; an apparent gain needs an unchanged-controller check on other tasks. No GPU hardware is used.
