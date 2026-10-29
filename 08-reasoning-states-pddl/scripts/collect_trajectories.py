"""Collect matched baseline and one-disturbance trajectories.

No language model is used here. The controller is a deterministic reference
controller that follows the planner's valid plan. This isolates and verifies
the environment/trajectory contract before introducing an agent policy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "lib") not in sys.path:
    sys.path.insert(0, str(ROOT / "lib"))
from pddl_env import BlocksWorldEnv  # noqa: E402


def run_baseline(domain: Path, problem: Path, plan: list[str]) -> dict:
    env = BlocksWorldEnv(domain, problem)
    initial = env.snapshot()
    trace = [env.step(action) for action in plan]
    assert env.solved
    return {
        "controller": "reference_plan",
        "condition": "baseline",
        "initial": initial,
        "trace": trace,
        "final": env.snapshot(),
    }


def run_disturbed(
    domain: Path, problem: Path, plan: list[str], injection_index: int
) -> dict:
    env = BlocksWorldEnv(domain, problem)
    initial = env.snapshot()
    trace = [env.step(action) for action in plan[:injection_index]]
    checkpoint = env.snapshot()
    failed = env.step(plan[injection_index], transient_failure=True)
    retry = env.step(plan[injection_index])
    trace.extend([failed, retry])
    trace.extend(env.step(action) for action in plan[injection_index + 1 :])
    assert failed["status"] == "transient_failure"
    assert retry["status"] == "success"
    assert env.solved
    return {
        "controller": "reference_plan_with_retry",
        "condition": "one_transient_failure",
        "injection_index": injection_index,
        "initial": initial,
        "checkpoint": checkpoint,
        "trace": trace,
        "final": env.snapshot(),
    }


def main() -> None:
    domain = ROOT / "pddl" / "domain.pddl"
    tasks = json.loads((ROOT / "results" / "tasks.json").read_text())
    episodes = []
    for item in tasks:
        problem = ROOT / item["problem_file"]
        plan = item["reference_plan"]
        baseline = run_baseline(domain, problem, plan)
        index = max(0, len(plan) // 2)
        disturbed = run_disturbed(domain, problem, plan, index)

        # The prefix and post-retry continuation must be matched exactly.
        assert (
            baseline["trace"][index]["state_before"]
            == disturbed["checkpoint"]["state"]
        )
        assert disturbed["trace"][index + 1]["state"] == baseline["trace"][index]["state"]
        for offset, base_event in enumerate(baseline["trace"][index + 1 :], start=index + 2):
            assert disturbed["trace"][offset]["state"] == base_event["state"]
        episodes.extend(
            [
                {"task_id": item["task_id"], **baseline},
                {"task_id": item["task_id"], **disturbed},
            ]
        )

    out = ROOT / "results" / "trajectories.json"
    out.write_text(json.dumps(episodes, indent=2) + "\n")
    total_events = sum(len(ep["trace"]) for ep in episodes)
    print(f"episodes={len(episodes)} tasks={len(tasks)} events={total_events}")
    print(f"matched baseline/disturbed continuations verified for {len(tasks)} tasks")
    print(f"output={out}")


if __name__ == "__main__":
    main()
