import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import transport


class TransportTests(unittest.TestCase):
    def test_tool_plan_format_uses_function_arguments(self):
        from types import SimpleNamespace
        captured = {}
        class Endpoint:
            async def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(model_dump=lambda **kw: {'choices':[{'finish_reason':'tool_calls','message':{
                    'role':'assistant','tool_calls':[{'function':{'name':'submit_plan','arguments':'{"actions":["operation"]}'}}]}}]})
        client = SimpleNamespace(chat=SimpleNamespace(completions=Endpoint()))
        with tempfile.TemporaryDirectory() as tmp:
            p = transport.PlanProposer(client, Path(tmp)/'log.jsonl', output_format='tool')
            self.assertEqual(asyncio.run(p(observation={},failures=[],remaining_plan=[])), ['operation'])
        self.assertEqual(captured['tools'][0]['function']['name'], 'submit_plan')

    def test_review_mode_receives_contract_and_review_instruction(self):
        from types import SimpleNamespace
        captured={}
        class Endpoint:
            async def create(self, **kwargs):
                captured.update(kwargs)
                raise RuntimeError('capture')
        client=SimpleNamespace(chat=SimpleNamespace(completions=Endpoint()))
        with tempfile.TemporaryDirectory() as tmp:
            p=transport.PlanProposer(client,Path(tmp)/'review.jsonl',output_format='review')
            with self.assertRaises(RuntimeError):
                asyncio.run(p(observation={'state':['observed']},failures=[],remaining_plan=['draft']))
        self.assertIn('Audit the candidate',captured['messages'][0]['content'])
        self.assertIn('tools',captured)
        self.assertEqual(captured['temperature'],0.7)

    def test_native_thinking_flag_is_sent_explicitly(self):
        captured = {}
        class Endpoint:
            async def create(self, **kwargs):
                captured.update(kwargs)
                raise RuntimeError('probe endpoint')
        from types import SimpleNamespace
        client = SimpleNamespace(chat=SimpleNamespace(completions=Endpoint()))
        with tempfile.TemporaryDirectory() as tmp:
            p = transport.PlanProposer(client, Path(tmp)/'log.jsonl', thinking='off-native')
            with self.assertRaises(RuntimeError):
                asyncio.run(p(observation={}, failures=[], remaining_plan=[]))
        self.assertEqual(captured['extra_body'], {'enable_thinking': False})

    def test_plan_parser_preserves_operations_and_rejects_bad_payload(self):
        self.assertEqual(transport.parse_plan('{"actions":["(pickup b1)","(putdown b1)"]}'),
                         ['(pickup b1)', '(putdown b1)'])
        self.assertEqual(transport.parse_plan('```json\n{"actions":["(pickup b1)"]}\n```'), ['(pickup b1)'])
        for payload in ['{"actions":[]}', '{"actions":[1]}', '{"actions":[" "]}', 'not json']:
            with self.assertRaises(ValueError):
                transport.parse_plan(payload)

    def test_prompt_uses_only_observation_and_explicit_evidence(self):
        observation = {'state': ['fact-a'], 'goal': ['fact-b'], 'goal_reached': False}
        messages = transport.plan_messages(observation, [], ['operation-1'])
        context = json.loads(messages[-1]['content'])
        self.assertEqual(context['observation'], observation)
        self.assertEqual(context['remaining_plan'], ['operation-1'])
        self.assertEqual(context['failures'], [])
        self.assertNotIn('reference_plan', json.dumps(messages))
        self.assertNotIn('valid_actions', json.dumps(messages))

    def test_response_log_redacts_private_reasoning(self):
        raw = {'model': 'test', 'choices': [{'message': {'role': 'assistant',
               'content': '<think>private text</think>{"actions":["operation"]}',
               'reasoning_content': 'private text'}}]}
        safe = transport.public_response(raw)
        self.assertNotIn('private text', json.dumps(safe))
        self.assertEqual(safe['choices'][0]['message']['content'], '{"actions":["operation"]}')

    def test_api_request_uses_explicit_sampling_and_logs_actual_response(self):
        class Response:
            def model_dump(self, **kwargs):
                return {'model': 'test-model', 'usage': {'completion_tokens': 7},
                        'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant',
                                     'content': '{"actions":["operation"]}'}}]}
        class Endpoint:
            async def create(self, **kwargs):
                self.parameters = kwargs
                return Response()
        class Client:
            pass
        endpoint = Endpoint()
        client = Client()
        client.chat = Client()
        client.chat.completions = endpoint
        with tempfile.TemporaryDirectory() as tmp:
            proposer = transport.PlanProposer(client, Path(tmp) / 'requests.jsonl', model='test-model')
            plan = asyncio.run(proposer(observation={'goal': ['g']}, failures=[], remaining_plan=[]))
            self.assertEqual(plan, ['operation'])
            self.assertEqual(endpoint.parameters['temperature'], 0.7)
            self.assertEqual(endpoint.parameters['top_p'], 0.9)
            events = [json.loads(x) for x in (Path(tmp) / 'requests.jsonl').read_text().splitlines()]
            self.assertEqual([e['kind'] for e in events], ['request', 'response'])
            self.assertEqual(events[1]['data']['model'], 'test-model')


if __name__ == '__main__':
    unittest.main()
