# Five-decision prefix voting: exploratory results

All 24 planned distinct jobs completed; expected key coverage and stored replay checks were verified. Three independent repeats per task/condition/mode. Source snapshots and logs are in ../results/control/prefix-vote-eval-01/. Settings were frozen in PREFIX-VOTE-PROTOCOL.md.

| Measure | Single-sample replanning | Prefix voting |
|---|---:|---:|
| Goal reached | 9/12 | 11/12 |
| Baseline goal reached | 6/6 | 5/6 |
| Transient-disturbance goal reached | 3/6 | 6/6 |
| Task/condition cells with mixed outcomes | 2/4 | 1/4 |
| Different-outcome repeat pairs | 4/12 | 2/12 |
| Model requests | 296 | 400 |
| Action attempts | 300 | 285 |
| Repeated rejected state/action pairs | 24 | 15 |
| State revisits | 18 | 12 |

Prefix voting obtained majority support in 52 of 60 voting rounds. Remaining rounds used the documented first-usable-sample tie fallback. It voted only on the first five action decisions, not necessarily five successes. Later decisions used single-sample replanning.

Replanning failures: one action limit, one API timeout, one missing submit_plan tool call. Prefix failure: one API timeout after the voting prefix. Fewer errors are not necessarily evidence of better planning; early voting can tolerate malformed samples. Counts of rejection and revisit events have unequal exposure. Repeated-pair measurements within cells are dependent and must not be treated as independent statistical trials.

This is encouraging descriptive evidence for improved end-to-end robustness at much lower cost than full step-wise voting. It is not conclusive: only three repeats per cell, two previously used tasks, no paired random streams, and infrastructure failures affect outcomes. No stable failure cells occurred. Both arms remain stochastic; identical traces are not established or required. Generalization to other tasks or tools is not demonstrated. An unchanged-controller evaluation on other benchmark tasks is needed before claiming a reliable gain.

No GPU hardware, simulator policy guidance, package installation, or original-trajectory modification was used.
