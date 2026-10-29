# Repository artifact policy

This branch includes study source code, tests, task definitions, protocols, compact reports, and small visualizations. Raw experiment data remains on the original machine and is not included in Git: model trajectory JSON, request logs, episode dumps, exact-state graphs, and experiment source snapshots. Virtual environments, vendor checkouts, and Python caches are also excluded.

Reports can refer to these local-only artifacts. A fresh clone can run offline controller tests but cannot reproduce historical-data analyses without those artifacts. Paid API experiments require credentials supplied through the documented environment variables; no credentials are included.

Main follow-up reports:

- `decision-record-study/REPORT.md`: trajectory audit and decision-record experiments.
- `control-study/RESULTS.md`: plan-commitment comparisons.
- `control-study/PREFIX-VOTE-RESULTS.md`: voting on the first five decisions.
- `control-study/PREFIX-VOTE-PROTOCOL.md`: prefix-voting evaluation design.

The original artifacts remain under `results/` locally. Their exclusion does not delete or replace them.
