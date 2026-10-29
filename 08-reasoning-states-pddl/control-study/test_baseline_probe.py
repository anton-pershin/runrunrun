"""Offline tests; run with the study's existing .venv/bin/python -m unittest."""
import importlib.util
import json
from pathlib import Path
import unittest

MODULE = Path(__file__).with_name('baseline_probe.py')
probe = None
if MODULE.exists():
    spec = importlib.util.spec_from_file_location('baseline_probe', MODULE)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    def test_request_matrix_changes_only_interface_and_thinking(self):
        self.assertIsNotNone(probe, 'baseline_probe implementation is missing')
        context = {'name': 'initial', 'task_id': 'task-005',
                   'observation': {'state': ['(holding b4)'], 'goal': ['(on b4 b2)']},
                   'catalog': ['(putdown b4)', '(pickup b1)']}
        for interface in ['tool', 'json_action', 'json_plan']:
            default = probe.build_request(context, interface, 'default')
            disabled = probe.build_request(context, interface, 'disabled')
            self.assertEqual(default, {k: v for k, v in disabled.items() if k != 'extra_body'})
            self.assertEqual(disabled['extra_body'], {'chat_template_kwargs': {'enable_thinking': False}})
            self.assertEqual(default['model'], 'qwen/qwen3.5-9b')
            self.assertEqual((default['temperature'], default['top_p'], default['max_tokens']), (0.7, 0.9, 8192))
            self.assertIn(json.dumps(context['catalog']), default['messages'][0]['content'])
            if interface == 'tool':
                self.assertEqual(default['tool_choice'], 'auto')
                self.assertFalse(default['parallel_tool_calls'])
                self.assertNotIn('enum', json.dumps(default['tools']))
            else:
                self.assertNotIn('tools', default)
                self.assertNotIn('response_format', default)
    def test_public_logs_strip_native_reasoning_and_think_blocks(self):
        raw = {'choices': [{'message': {'content': '<think>SECRET_ONE</think>{"action":"(putdown b4)"}',
                'reasoning_content': 'SECRET_TWO', 'reasoning_details': [{'text': 'SECRET_THREE'}]}}],
               'nested': [{'thinking': 'SECRET_FOUR'}], 'incomplete': '<think>SECRET_FIVE',
               'credential_example': 'key-for-test'}
        result = probe.public_data(raw, secrets=('key-for-test',))
        serialized = json.dumps(result)
        self.assertNotIn('SECRET_', serialized)
        self.assertNotIn('key-for-test', serialized)
        self.assertEqual(result['choices'][0]['message']['content'], '{"action":"(putdown b4)"}')
        self.assertIn('SECRET_TWO', raw['choices'][0]['message']['reasoning_content'])
        telemetry = probe.reasoning_metadata(raw)
        self.assertEqual(telemetry['native_reasoning_characters'], len('SECRET_TWO') + len(json.dumps([{'text': 'SECRET_THREE'}])))
        self.assertTrue(telemetry['content_has_think_tags'])
        self.assertNotIn('SECRET_', json.dumps(telemetry))
    def test_actual_contexts_replay_and_score_without_repair(self):
        self.assertTrue(hasattr(probe, 'load_contexts'), 'context loading and scoring missing')
        contexts = probe.load_contexts()
        self.assertEqual([c['name'] for c in contexts], ['initial', 'checkpoint'])
        initial, checkpoint = contexts
        self.assertEqual(initial['observation']['attempts'], 0)
        self.assertIn('(holding b4)', checkpoint['observation']['state'])
        for context in contexts:
            serialized = json.dumps(context)
            self.assertNotIn('reference_plan', serialized)
            self.assertEqual(len(context['catalog']), 60)
        good = {'choices': [{'finish_reason': 'stop', 'message': {'content':
                '{"actions":["(unstack b4 b5)","(putdown b4)"]}'}}]}
        result = probe.score_response(initial, 'json_plan', good)
        self.assertTrue(result['usable'])
        self.assertTrue(result['all_actions_feasible'])
        self.assertFalse(result['goal_reached'])
        bad = {'choices': [{'finish_reason': 'stop', 'message': {'content':
               '{"action":"(unstack b3 b1)"}'}}]}
        result = probe.score_response(initial, 'json_action', bad)
        self.assertTrue(result['usable'])
        self.assertFalse(result['first_action_feasible'])
        self.assertEqual(result['first_failure_index'], 0)
        tool = {'choices': [{'finish_reason': 'tool_calls', 'message': {'tool_calls': [
                {'function': {'name': 'execute_action', 'arguments': '{"action":"(putdown b4)"}'}}]}}]}
        self.assertTrue(probe.score_response(checkpoint, 'tool', tool)['first_action_feasible'])
        good['choices'][0]['finish_reason'] = 'length'
        self.assertFalse(probe.score_response(initial, 'json_plan', good)['usable'])
        empty = {'choices': [{'finish_reason': 'stop', 'message': {'content': '', 'reasoning_content': 'private'}}]}
        self.assertFalse(probe.score_response(initial, 'json_action', empty)['usable'])
        self.assertNotIn('private', json.dumps(probe.score_response(initial, 'json_action', empty)))


if __name__ == '__main__':
    unittest.main()
