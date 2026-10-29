"""Executable smoke tests for the execution contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "lib") not in sys.path:
    sys.path.insert(0, str(ROOT / "lib"))
from pddl_env import BlocksWorldEnv  # noqa: E402


def main() -> None:
    domain = ROOT / "pddl" / "domain.pddl"
    metadata = json.loads((ROOT / "results" / "tasks.json").read_text())
    checks = 0
    for item in metadata:
        env = BlocksWorldEnv(domain, ROOT / item["problem_file"])
        before = tuple(env.snapshot()["state"])
        available = env.list_available_actions()
        assert "(putdown b1)" in available
        assert "(put-down b1)" not in available

        malformed = env.step("stack")
        assert malformed["status"] == "malformed_action"
        assert not malformed["world_changed"]
        assert tuple(malformed["state"]) == before

        unknown = env.step("(not-a-real-action b1)")
        assert unknown["status"] == "unknown_action"
        assert not unknown["world_changed"]
        assert tuple(unknown["state"]) == before

        invalid = next(
            action for action in env.task.operators
            if not action.applicable(env.state)
        )
        failed = env.step(invalid.name)
        assert failed["status"] == "precondition_failed"
        assert failed["details"]["missing_preconditions"]
        assert not failed["world_changed"]
        assert tuple(failed["state"]) == before

        valid = env.valid_actions()[0]
        transient = env.step(valid, transient_failure=True)
        assert transient["status"] == "transient_failure"
        assert not transient["world_changed"]
        assert tuple(transient["state"]) == before

        success = env.step(valid)
        assert success["status"] == "success"
        assert success["world_changed"]
        assert env.list_available_actions() == available
        checks += 6
    print(f"wrapper smoke checks passed: {checks}")


if __name__ == "__main__":
    main()
