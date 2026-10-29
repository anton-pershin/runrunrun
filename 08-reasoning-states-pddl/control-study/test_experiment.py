import asyncio
import copy
import importlib.util
import json
import unittest
from pathlib import Path

import experiment

ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = next(t for t in json.loads((ROOT / 'results/tasks.json').read_text()) if t['task_id'] == 'task-004')

    def test_executor_disturbance_is_observation_only_and_happens_once(self):
        env, execute = experiment.make_executor(self.task, 'one_transient_failure')
        operations = self.task['reference_plan']
        async def run():
            results = []
            for op in operations:
                results.append(await execute(op))
            # Task 004 is two operations, so it finishes before disturbance eligibility.
            return results
        results = asyncio.run(run())
        self.assertTrue(env.solved)
        self.assertEqual(sum(e['status'] == 'transient_failure' for e in results), 0)
        self.assertEqual(results[-1]['state'], env.snapshot()['state'])

    def test_episode_verifier_replays_transitions_and_rejects_corruption(self):
        env, execute = experiment.make_executor(self.task, 'baseline')
        initial = env.snapshot()
        async def run():
            return [await execute(op) for op in self.task['reference_plan']]
        trace = asyncio.run(run())
        episode = {'task_id': self.task['task_id'], 'condition': 'baseline', 'initial': initial,
                   'trace': trace, 'final': env.snapshot(), 'goal_reached': env.solved,
                   'max_actions': 50}
        self.assertTrue(experiment.verify_episode(episode)['valid'])
        broken = copy.deepcopy(episode)
        broken['trace'][0]['state'] = []
        self.assertFalse(experiment.verify_episode(broken)['valid'])

    def test_balanced_jobs_have_unique_keys_and_do_not_select_only_successes(self):
        jobs = experiment.make_jobs(['task-005', 'task-011'], ['baseline', 'one_transient_failure'], 2)
        self.assertEqual(len(jobs), 16)
        self.assertEqual(len({tuple(sorted(j.items())) for j in jobs}), 16)
        self.assertEqual(sum(j['mode'] == 'commit' for j in jobs), 8)
        self.assertEqual(experiment.make_jobs(['task-005', 'task-011'], ['baseline', 'one_transient_failure'], 2), jobs)


if __name__ == '__main__':
    unittest.main()
