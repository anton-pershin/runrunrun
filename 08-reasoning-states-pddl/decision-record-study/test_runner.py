"""Run with the study's existing Python: python -m unittest discover -s ..."""
import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class RunnerTests(unittest.TestCase):
    def test_record_context_excludes_history_but_contains_active_record(self):
        self.assertIsNotNone(importlib.util.find_spec('runner'), 'instrumented runner is missing')
        import runner
        observation = {'id': 'obs-3', 'state': ['(holding b2)'], 'goal': ['(on b2 b1)']}
        record = {'current_subgoal': 'place b2', 'confirmed_facts': []}
        history = [{'role': 'user', 'content': 'OLD_HISTORY_SENTINEL'}]
        messages = runner.action_messages('record', observation, record, history, ['(putdown b2)'])
        import json
        text = json.dumps(messages)
        self.assertIn('place b2', text)
        self.assertIn('obs-3', text)
        self.assertNotIn('OLD_HISTORY_SENTINEL', text)
        self.assertNotIn('valid_actions', text)

    def test_episode_uses_record_and_stops_on_observed_goal(self):
        import asyncio
        import json
        import runner
        self.assertTrue(hasattr(runner, 'run_episode'), 'episode runner is missing')
        root = HERE.parent
        task = next(t for t in json.loads((root / 'results/tasks.json').read_text()) if t['task_id'] == 'task-004')
        requests = []
        actions = iter(['(unstack b1 b4)', '(stack b1 b5)'])

        async def request(messages, tools, stage, tag):
            requests.append((stage, messages, tools))
            if stage == 'record':
                obs = json.loads(messages[-1]['content'])['latest_observation']
                record = {'objective': obs['goal'], 'confirmed_facts': [
                    {'fact': fact, 'observation_id': obs['id']} for fact in obs['state']],
                    'uncertainties': [], 'current_subgoal': 'TEST_SUBGOAL',
                    'last_operation': {'action': obs.get('action', ''), 'status': obs.get('status', 'initial'), 'observation_id': obs['id']},
                    'strategy': {'commitment': 'complete task', 'revise_if': 'observed failure'}}
                name, arguments = 'update_decision_record', record
            else:
                self.assertIn('TEST_SUBGOAL', json.dumps(messages))
                name, arguments = 'execute_action', {'action': next(actions)}
            return {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': f'call-{len(requests)}', 'type': 'function',
                 'function': {'name': name, 'arguments': json.dumps(arguments)}}]}

        episode = asyncio.run(runner.run_episode(task, 'record', 'baseline', 0, request, max_actions=50))
        self.assertTrue(episode['goal_reached'])
        self.assertEqual(episode['termination'], 'observed_goal')
        self.assertEqual(len(episode['trace']), 2)
        self.assertEqual([s for s, _, _ in requests], ['record', 'action', 'record', 'action'])
        self.assertEqual(len(episode['records']), 2)
        self.assertTrue(episode['records'][1]['previous_record'] == episode['records'][0]['record'])
        self.assertTrue(runner.verify_episode(episode)['valid'])
        episode['trace'][0]['state'] = []
        self.assertFalse(runner.verify_episode(episode)['valid'])

    def test_api_recorder_fixes_sampling_and_persists_requests(self):
        import asyncio
        import json
        import tempfile
        from types import SimpleNamespace
        import runner
        self.assertTrue(hasattr(runner, 'APIRecorder'), 'API recorder is missing')
        captured = []
        class Response:
            choices = [SimpleNamespace(message=SimpleNamespace())]
            def model_dump(self, **kwargs):
                return {'model': 'returned-model', 'choices': [{'message': {
                    'role': 'assistant', 'content': None, 'reasoning_content': 'DO_NOT_STORE',
                    'tool_calls': [{'id': 'c', 'type': 'function', 'function': {
                        'name': 'execute_action', 'arguments': '{"action":"(pickup b1)"}'}}]}}],
                    'usage': {'total_tokens': 12}}
        async def create(**kwargs):
            captured.append(kwargs)
            return Response()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with tempfile.TemporaryDirectory(dir='/home/tony/.hermes/cache/scratch') as tmp:
            path = Path(tmp) / 'requests.jsonl'
            request = runner.APIRecorder(client, path, 'test-model', 0.7, 0.9)
            message = asyncio.run(request([{'role': 'assistant', 'content': None,
                'reasoning_content': 'DO_NOT_STORE'}], [runner.ACTION_TOOL], 'action', 0))
            self.assertEqual(message.get('reasoning_content'), 'DO_NOT_STORE',
                             'native provider history must be preserved in memory, not on disk')
            self.assertEqual(captured[0]['temperature'], 0.7)
            self.assertEqual(captured[0]['top_p'], 0.9)
            self.assertEqual(captured[0]['max_tokens'], 8192)
            self.assertFalse(captured[0]['parallel_tool_calls'])
            self.assertNotIn('seed', captured[0])
            text = path.read_text()
            self.assertNotIn('DO_NOT_STORE', text)
            entries = [json.loads(line) for line in text.splitlines()]
            self.assertEqual([e['kind'] for e in entries], ['request', 'response'])
            self.assertEqual(message['tool_calls'][0]['function']['name'], 'execute_action')

    def test_completion_truncation_is_not_voluntary_stop(self):
        import asyncio
        import tempfile
        from types import SimpleNamespace
        import runner
        class TruncatedResponse:
            def model_dump(self, **kwargs):
                return {'choices': [{'finish_reason': 'length', 'message': {'role': 'assistant'}}]}
        async def create(**kwargs):
            return TruncatedResponse()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with tempfile.TemporaryDirectory(dir='/home/tony/.hermes/cache/scratch') as tmp:
            request = runner.APIRecorder(client, Path(tmp) / 'log.jsonl', 'm', 0.7, 0.9)
            with self.assertRaisesRegex(RuntimeError, 'completion token limit'):
                asyncio.run(request([], [], 'action', 0))

    def test_metrics_separate_record_corruption_from_action_errors(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('analyze'), 'analysis module is missing')
        import analyze
        observation = {'id': 'obs-0', 'state': ['(arm-empty)'], 'goal': ['(on b1 b2)']}
        record = {'objective': ['(on b1 b2)'], 'confirmed_facts': [
            {'fact': '(holding b1)', 'observation_id': 'obs-0'},
            {'fact': '(arm-empty)', 'observation_id': 'obs-99'}],
            'uncertainties': [], 'current_subgoal': 'place b1',
            'last_operation': {'action': '', 'status': 'initial', 'observation_id': 'obs-0'},
            'strategy': {'commitment': 'finish', 'revise_if': 'failure'}}
        metrics = analyze.record_quality(record, observation, {'obs-0': observation})
        self.assertEqual(metrics['unsupported_claims'], 2)
        self.assertEqual(metrics['false_current_claims'], 1)
        self.assertEqual(metrics['goal_drift'], 0)
        self.assertEqual(metrics['last_operation_mismatch'], 0)

    def test_metrics_count_failed_repeats_separately_from_successful_cycles(self):
        import analyze
        import runner
        self.assertTrue(hasattr(analyze, 'episode_metrics'), 'episode metrics are missing')
        env = runner.BlocksWorldEnv(HERE.parent / 'pddl/domain.pddl', HERE.parent / 'pddl/tasks/task-004.pddl')
        initial = env.snapshot()
        trace = [env.step(a) for a in ['(pickup b1)', '(pickup b1)', '(unstack b1 b4)',
                                       '(putdown b1)', '(pickup b1)', '(stack b1 b5)']]
        episode = {'initial': initial, 'trace': trace, 'records': [], 'goal_reached': env.solved,
                   'arm': 'history', 'termination': 'observed_goal'}
        metrics = analyze.episode_metrics(episode)
        self.assertEqual(metrics['actions'], 6)
        self.assertEqual(metrics['repeated_failed_state_action'], 1)
        self.assertEqual(metrics['successful_state_revisits'], 1)
        self.assertEqual(metrics['statuses']['precondition_failed'], 2)
        self.assertEqual(metrics['first_goal_action'], 6)

    def test_transient_failure_preserves_state_and_is_injected_once(self):
        import asyncio
        import json
        import runner
        task = next(t for t in json.loads((HERE.parent / 'results/tasks.json').read_text()) if t['task_id'] == 'task-001')
        plan = task['reference_plan']
        actions = iter(plan[:2] + [plan[2]] + plan[2:])
        async def request(messages, tools, stage, tag):
            return {'role': 'assistant', 'tool_calls': [{'id': f'c{tag}', 'type': 'function',
                'function': {'name': 'execute_action', 'arguments': json.dumps({'action': next(actions)})}}]}
        episode = asyncio.run(runner.run_episode(task, 'history', 'one_transient_failure', 0, request))
        self.assertTrue(episode['goal_reached'])
        self.assertTrue(episode['disturbance_injected'])
        failures = [x for x in episode['trace'] if x['status'] == 'transient_failure']
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]['attempt'], 3)
        self.assertEqual(failures[0]['state_before'], failures[0]['state'])
        self.assertTrue(runner.verify_episode(episode)['valid'])
        episode['disturbance_injected'] = False
        self.assertFalse(runner.verify_episode(episode)['valid'])

    def test_summary_reports_missing_episodes_without_claiming_verification(self):
        import json
        import tempfile
        import analyze
        with tempfile.TemporaryDirectory(dir='/home/tony/.hermes/cache/scratch') as tmp:
            directory = Path(tmp)
            job = {'task_id': 'task-001', 'arm': 'history', 'condition': 'baseline', 'repeat': 0}
            (directory / 'manifest.json').write_text(json.dumps({'jobs': [job]}))
            summary = analyze.summarize(directory)
            self.assertEqual(summary['planned_episodes'], 1)
            self.assertEqual(summary['completed_episodes'], 0)
            self.assertFalse(summary['all_replays_valid'])
            self.assertEqual(len(summary['missing_jobs']), 1)

    def test_verifier_enforces_schedule_budget_and_termination(self):
        import asyncio
        import copy
        import json
        import runner
        task = next(t for t in json.loads((HERE.parent / 'results/tasks.json').read_text()) if t['task_id'] == 'task-001')
        actions = iter(task['reference_plan'])
        async def request(messages, tools, stage, tag):
            return {'role': 'assistant', 'tool_calls': [{'id': f'c{tag}', 'type': 'function',
                'function': {'name': 'execute_action', 'arguments': json.dumps({'action': next(actions)})}}]}
        original = asyncio.run(runner.run_episode(task, 'history', 'baseline', 0, request))
        self.assertTrue(runner.verify_episode(original)['valid'])
        for update in [{'max_actions': 1}, {'termination': 'action_limit'}, {'condition': 'one_transient_failure'}]:
            episode = copy.deepcopy(original)
            episode.update(update)
            with self.subTest(update=update):
                self.assertFalse(runner.verify_episode(episode)['valid'])

    def test_whitespace_action_cannot_bypass_disturbance(self):
        import asyncio
        import json
        import runner
        task = next(t for t in json.loads((HERE.parent / 'results/tasks.json').read_text()) if t['task_id'] == 'task-001')
        plan = task['reference_plan']
        actions = iter([a.replace(' ', '  ') for a in plan[:2] + [plan[2]] + plan[2:]])
        async def request(messages, tools, stage, tag):
            return {'role': 'assistant', 'tool_calls': [{'id': f'c{tag}', 'type': 'function',
                'function': {'name': 'execute_action', 'arguments': json.dumps({'action': next(actions)})}}]}
        episode = asyncio.run(runner.run_episode(task, 'history', 'one_transient_failure', 0, request))
        self.assertTrue(episode['disturbance_injected'])
        self.assertTrue(episode['goal_reached'])
        self.assertTrue(runner.verify_episode(episode)['valid'])

    def test_record_writer_refusal_is_a_model_stop(self):
        import asyncio
        import json
        import runner
        task = json.loads((HERE.parent / 'results/tasks.json').read_text())[0]
        async def request(*args):
            return {'role': 'assistant', 'content': 'I cannot continue.'}
        episode = asyncio.run(runner.run_episode(task, 'record', 'baseline', 0, request))
        self.assertEqual(episode['termination'], 'model_stopped')
        self.assertEqual(episode['termination_stage'], 'record')
        self.assertEqual(episode['final_response'], 'I cannot continue.')
        self.assertIsNone(episode['error'])

    def test_no_action_is_not_first_action_agreement(self):
        import analyze
        self.assertTrue(hasattr(analyze, 'first_action_agreement'))
        metric = analyze.first_action_agreement([{'trace': []}, {'trace': []},
            {'trace': [{'action': '(pickup b1)'}]}, {'trace': [{'action': '(pickup b1)'}]}])
        self.assertEqual(metric['no_first_action'], 2)
        self.assertEqual(metric['agreement'], 1.0)
        self.assertIsNone(analyze.first_action_agreement([{'trace': []}, {'trace': []}])['agreement'])

    def test_all_arms_have_the_same_explicit_decision_request(self):
        import runner
        observation = {'id': 'obs-1', 'state': ['(arm-empty)'], 'goal': ['(on b1 b2)']}
        history = [{'role': 'tool', 'tool_call_id': 'c1', 'content': '{}'}]
        messages = [runner.action_messages(arm, observation, {}, history, []) for arm in ['history', 'reset', 'record']]
        self.assertTrue(all(m[-1]['role'] == 'user' for m in messages))
        self.assertTrue(all(m[-1]['content'] == messages[0][-1]['content'] for m in messages))
        self.assertIn('Choose your next execute_action call', messages[0][-1]['content'])

    def test_record_timeline_links_declared_state_to_actual_outcome(self):
        import analyze
        self.assertTrue(hasattr(analyze, 'render_record_timeline'))
        episode = {'task_id': 'task-001', 'arm': 'record', 'condition': 'baseline', 'repeat': 0,
            'goal_reached': False, 'termination': 'action_limit', 'initial': {'goal': ['(on b1 b2)']},
            'records': [{'before_action': 1, 'observation_id': 'obs-0', 'record': {'current_subgoal': 'TEST_SUBGOAL'}}],
            'trace': [{'action': '(pickup b1)', 'status': 'precondition_failed', 'details': {'missing_preconditions': ['(clear b1)']}}]}
        text = analyze.render_record_timeline(episode)
        for token in ['obs-0', 'TEST_SUBGOAL', '(pickup b1)', 'precondition_failed', '(clear b1)']:
            self.assertIn(token, text)


if __name__ == '__main__':
    unittest.main()
