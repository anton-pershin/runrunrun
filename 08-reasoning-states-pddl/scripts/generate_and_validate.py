"""Generate Blocks World tasks and validate reference plans.

Run from this directory with the vendored pyperplan import path handled by the
script. The generator is the published AI-Planning/pddl-generators program;
its random seed is recorded for every task.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
PYPERPLAN = VENDOR / "pyperplan"
if str(PYPERPLAN) not in sys.path:
    sys.path.insert(0, str(PYPERPLAN))
if str(ROOT / "lib") not in sys.path:
    sys.path.insert(0, str(ROOT / "lib"))

from pyperplan.heuristics.relaxation import hFFHeuristic  # noqa: E402
from pyperplan.planner import SEARCHES, search_plan  # noqa: E402

from pddl_env import BlocksWorldEnv  # noqa: E402

DOMAIN_SOURCE = VENDOR / "pddl-generators" / "blocksworld" / "4ops" / "domain.pddl"
GENERATOR = VENDOR / "pddl-generators" / "blocksworld" / "blocksworld"
PDDL_DIR = ROOT / "pddl"
TASK_DIR = PDDL_DIR / "tasks"
RESULTS_DIR = ROOT / "results"


def plan_for(domain: Path, problem: Path):
    return search_plan(
        str(domain),
        str(problem),
        SEARCHES["astar"],
        hFFHeuristic,
        successor_generator="tree",
    )


def replay_plan(env: BlocksWorldEnv, plan) -> list[dict]:
    transitions = []
    for op in plan or []:
        transition = env.step(op.name)
        transitions.append(transition)
        if transition["status"] != "success":
            raise AssertionError(f"reference plan action failed: {transition}")
    return transitions


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    if not GENERATOR.exists():
        raise SystemExit(f"missing built generator: {GENERATOR}; run make first")
    PDDL_DIR.mkdir(exist_ok=True)
    TASK_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    domain = PDDL_DIR / "domain.pddl"
    domain.write_text(DOMAIN_SOURCE.read_text())

    metadata = []
    # Small, inspectable tasks first; the seed is part of the task identity.
    configs = [(5, seed) for seed in range(101, 107)] + [
        (6, seed) for seed in range(107, 113)
    ]
    for index, (blocks, seed) in enumerate(configs, start=1):
        task_id = f"task-{index:03d}"
        problem = TASK_DIR / f"{task_id}.pddl"
        generated = subprocess.run(
            [str(GENERATOR), "4", str(blocks), str(seed)],
            check=True,
            capture_output=True,
            text=True,
        )
        problem.write_text(generated.stdout)

        env = BlocksWorldEnv(domain, problem)
        initial = env.snapshot()
        plan = plan_for(domain, problem)
        if plan is None:
            raise AssertionError(f"generator produced an unsolved task: {task_id}")
        reference = replay_plan(env, plan)
        if not env.solved:
            raise AssertionError(f"reference plan did not reach goal: {task_id}")
        metadata.append(
            {
                "task_id": task_id,
                "blocks": blocks,
                "seed": seed,
                "problem_file": str(problem.relative_to(ROOT)),
                "initial_state": initial["state"],
                "goal": initial["goal"],
                "reference_plan": [t["action"] for t in reference],
                "reference_plan_length": len(reference),
                "planner": "pyperplan A* + hFF",
                "optimality_proven": False,
            }
        )
        print(f"{task_id}: blocks={blocks} plan_length={len(reference)}")

    (RESULTS_DIR / "tasks.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"validated {len(metadata)} tasks; metadata={RESULTS_DIR / 'tasks.json'}")


if __name__ == "__main__":
    main()
