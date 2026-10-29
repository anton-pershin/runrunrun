# Exact offline evaluation

These scripts measure the original four Qwen runs. They do not select agent actions, call a model, or feed distance estimates into an agent prompt.

`evaluate_exact.py` enumerates each reachable state graph, checks transition reversibility, computes exact shortest distances to all states that satisfy each goal, and replays every historical transition. `verify_artifacts.py` checks the saved graphs, distances, annotations, and totals separately.

The scripts are preserved from the verified scratch evaluation. The evaluator restricts output to the Hermes scratch directory. The verifier reads data beside its own script. Use a scratch copy of both scripts for reproduction. Start in the parent study directory, `08-reasoning-states-pddl`, using the existing environment:

```sh
mkdir -p "$TMPDIR/pddl-exact-offline-reproduction"
cp decision-record-study/exact-offline/evaluate_exact.py \
   decision-record-study/exact-offline/verify_artifacts.py \
   "$TMPDIR/pddl-exact-offline-reproduction/"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  "$TMPDIR/pddl-exact-offline-reproduction/evaluate_exact.py" --repo "$PWD" \
  > "$TMPDIR/pddl-exact-offline-reproduction/run.log"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  "$TMPDIR/pddl-exact-offline-reproduction/verify_artifacts.py"
```

These commands were exercised on this machine. Verification checked 96 trajectories and 2,061 transitions. The complete graphs contain 866 states for five blocks and 7,057 states for six blocks. All reachable states have finite goal distance for each corresponding task.

The original generated artifacts are also preserved under `results/decision-record/historical-exact/`. They are generated data, not an additional policy input. The summary includes input SHA-256 hashes. Do not commit large generated artifacts without review. The scratch outputs are temporary; the repository-local scripts and original trajectories are sufficient to regenerate them.
