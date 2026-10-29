"""Unit tests with ordinary async fixtures; no model or simulator calls."""
import asyncio
import importlib
from pathlib import Path
import unittest


def run(initial, propose, execute, **kwargs):
    path = Path(__file__).with_name("controller.py")
    assert path.exists(), "controller implementation is missing"
    return asyncio.run(importlib.import_module("controller").run_controller(
        initial, propose, execute, **kwargs
    ))


class ControllerTests(unittest.TestCase):
    def test_commits_cursor_without_resampling_successful_operations(self):
        proposals = []
        actions = []
        observations = []

        async def propose(**context):
            proposals.append(context)
            return ["open", "write", "close", "never"]

        async def execute(action):
            actions.append(action)
            event = {"status": "success", "state": list(actions),
                     "goal_reached": action == "close"}
            observations.append(event)
            return event

        result = run({"state": [], "goal_reached": False}, propose, execute)
        self.assertEqual(actions, ["open", "write", "close"])
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["remaining_plan"], [])
        self.assertEqual(proposals[0]["failures"], [])
        self.assertEqual(result["trace"], observations)
        self.assertEqual(result["executed_actions"], actions)
        self.assertEqual(result["plans"][0]["actions"], ["open", "write", "close", "never"])
        self.assertEqual(result["plans"][0]["cursor"], 3)
        self.assertEqual(result["revisions"][0]["reason"], "initial")
        self.assertEqual(result["termination"], "goal_reached")
        self.assertIs(result["goal_reached"], True)

    def test_semantic_failure_revises_from_actual_observation_and_failed_suffix(self):
        proposals = []
        actions = []
        rejected = {"status": "precondition_failed", "state": {"opaque": "new"},
                    "goal_reached": False, "details": {"message": "locked"}}

        async def propose(**context):
            proposals.append(context)
            return ["open", "stale"] if len(proposals) == 1 else ["unlock"]

        async def execute(action):
            actions.append(action)
            return rejected if action == "open" else {
                "status": "success", "state": "done", "goal_reached": True}

        result = run({"state": "start", "goal_reached": False}, propose, execute)
        self.assertEqual(actions, ["open", "unlock"])
        self.assertEqual(proposals[1]["observation"], rejected)
        self.assertEqual(proposals[1]["remaining_plan"], ["open", "stale"])
        self.assertEqual(proposals[1]["failures"], [{"action": "open", "observation": rejected}])
        self.assertEqual(result["plans"][0]["cursor"], 0)
        self.assertEqual(result["plans"][1]["cursor"], 1)
        self.assertEqual(result["revisions"][1]["reason"], "failure")
        self.assertEqual(result["trace"][0], rejected)


    def test_transient_failure_retries_same_operation_without_discarding_plan(self):
        proposals = []
        actions = []

        async def propose(**context):
            proposals.append(context)
            return ["open", "close"] if len(proposals) == 1 else ["unwanted"]

        async def execute(action):
            actions.append(action)
            return {"status": "transient_failure" if len(actions) == 1 else "success",
                    "state": len(actions), "goal_reached": action == "close"}

        result = run({"goal_reached": False}, propose, execute)
        self.assertEqual(actions, ["open", "open", "close"])
        self.assertEqual(len(proposals), 1)
        self.assertEqual(result["plans"][0]["cursor"], 2)
        self.assertEqual(len(result["trace"]), 3)


class BudgetTests(unittest.TestCase):
    def test_action_budget_stops_an_unsuccessful_replanning_loop(self):
        async def propose(**kwargs):
            return ['repeat']
        async def execute(action):
            return {'status': 'success', 'goal_reached': False}
        result = run({'goal_reached': False}, propose, execute, max_actions=3)
        self.assertEqual(len(result['trace']), 3)
        self.assertEqual(result['termination'], 'action_limit')

    def test_planning_budget_stops_repeated_semantic_rejections(self):
        async def propose(**kwargs):
            return ['repeat']
        async def execute(action):
            return {'status': 'precondition_failed', 'goal_reached': False}
        result = run({'goal_reached': False}, propose, execute, max_replans=2)
        self.assertEqual(len(result['plans']), 2)
        self.assertEqual(result['termination'], 'planning_limit')

    def test_receding_mode_samples_again_after_success(self):
        contexts = []
        async def propose(**kwargs):
            contexts.append(kwargs)
            return ['open', 'close'] if len(contexts) == 1 else ['close']
        async def execute(action):
            return {'status': 'success', 'goal_reached': action == 'close'}
        result = run({'goal_reached': False}, propose, execute, mode='recede')
        self.assertEqual(len(contexts), 2)
        self.assertEqual(contexts[1]['remaining_plan'], ['close'])
        self.assertTrue(result['goal_reached'])


if __name__ == "__main__":
    unittest.main()
