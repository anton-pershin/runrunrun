"""A transparent Blocks World execution wrapper over pyperplan's STRIPS model.

The wrapper is deliberately separate from the planner. It exposes attempted
agent actions, deterministic outcomes, and the symbolic world after every
attempt. Invalid actions never mutate the world.
"""

from __future__ import annotations

import re
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYPERPLAN = ROOT / "vendor" / "pyperplan"
if str(PYPERPLAN) not in sys.path:
    sys.path.insert(0, str(PYPERPLAN))

from pyperplan.grounding import ground  # noqa: E402
from pyperplan.pddl.parser import Parser  # noqa: E402

_ACTION_RE = re.compile(r"^\(([a-z][a-z0-9_-]*(?:\s+[a-z0-9_-]+)*)\s*\)$")


class BlocksWorldEnv:
    """Execute grounded STRIPS actions and expose structured outcomes."""

    def __init__(self, domain_file: str | Path, problem_file: str | Path):
        self.domain_file = Path(domain_file)
        self.problem_file = Path(problem_file)
        parser = Parser(str(self.domain_file))
        domain = parser.parse_domain()
        parser.set_prob_file(str(self.problem_file))
        self.problem = parser.parse_problem(domain)
        self.task = ground(
            self.problem,
            remove_statics_from_initial_state=False,
            remove_irrelevant_operators=False,
        )
        self._operators = {op.name: op for op in self.task.operators}
        self._initial_state = frozenset(self.task.initial_state)
        self.state = set(self._initial_state)
        self.attempts = 0

    @property
    def goal(self) -> list[str]:
        return sorted(self.task.goals)

    @property
    def solved(self) -> bool:
        return self.task.goal_reached(self.state)

    def reset(self) -> dict[str, Any]:
        self.state = set(self._initial_state)
        self.attempts = 0
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": sorted(self.state),
            "goal": self.goal,
            "goal_reached": self.solved,
            "attempts": self.attempts,
        }

    def valid_actions(self) -> list[str]:
        return sorted(
            op.name for op in self.task.operators if op.applicable(self.state)
        )

    def list_available_actions(self) -> list[str]:
        """Return the fixed grounded action vocabulary, not currently valid moves.

        The result depends only on the PDDL task's objects and domain, never on
        the current world state. Actions with unsatisfied preconditions remain
        listed so the agent must reason about applicability itself.
        """
        return sorted(self._operators)

    def step(self, action: str, *, transient_failure: bool = False) -> dict[str, Any]:
        """Attempt an action and return a fully structured transition.

        ``transient_failure`` simulates a tool/execution failure. It is checked
        only after syntax and precondition validation, and never mutates state.
        """
        before = frozenset(self.state)
        self.attempts += 1
        parsed = _ACTION_RE.fullmatch(action.strip())
        if parsed is None:
            return self._result(action, "malformed_action", before, {
                "message": "expected '(action arg)' or '(action)'"
            })

        tokens = parsed.group(1).split()
        grounded = "(" + " ".join(tokens) + ")"
        op = self._operators.get(grounded)
        if op is None:
            return self._result(action, "unknown_action", before, {
                "message": "action is not a grounded action in this problem"
            })
        missing = sorted(op.preconditions - self.state)
        if missing:
            return self._result(action, "precondition_failed", before, {
                "missing_preconditions": missing
            })
        if transient_failure:
            return self._result(action, "transient_failure", before, {
                "message": "simulated execution failure; action was not applied"
            })

        self.state = set(op.apply(self.state))
        return self._result(action, "success", before, {})

    def _result(
        self,
        action: str,
        status: str,
        before: frozenset[str],
        details: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "attempt": self.attempts,
            "action": action,
            "status": status,
            "details": details,
            "state_before": sorted(before),
            "world_changed": frozenset(self.state) != before,
            "state": sorted(self.state),
            "goal": self.goal,
            "goal_reached": self.solved,
        }
