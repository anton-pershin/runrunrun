"""Opt-in paid integration test of the original Pydantic AI runner.

RUN_PAID_PROTOCOL_PROBE=1 PROBE_OUTPUT=<new-directory> python -m unittest discover ...
Only protocol metadata and the original public episode are saved, not thinking text.
"""
import asyncio
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))


@unittest.skipUnless(os.environ.get('RUN_PAID_PROTOCOL_PROBE') == '1', 'opt-in paid API probe')
class OriginalAPIProbe(unittest.TestCase):
    def test_original_agent_protocol(self):
        import runner
        spec = importlib.util.spec_from_file_location('original_runner', ROOT / 'scripts/run_llm_agent.py')
        assert spec is not None and spec.loader is not None
        original = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = original
        spec.loader.exec_module(original)
        os.environ['REASONING_STATES_MODEL'] = 'qwen/qwen3.5-9b'
        out = Path(os.environ['PROBE_OUTPUT']).resolve()
        out.mkdir(parents=True, exist_ok=False)
        agent = original.make_agent()
        agent.model_settings = {'temperature': 0.7, 'top_p': 0.9, 'max_tokens': 8192, 'parallel_tool_calls': False}
        create = agent.model.client.chat.completions.create
        requests = []

        async def logged_create(**kwargs):
            protocol: dict = {k: v for k, v in kwargs.items() if k not in {'messages', 'tools'} and isinstance(v, (str, bool, int, float, type(None)))}
            protocol['message_shapes'] = [{'role': m['role'], 'keys': sorted(m),
                'content_length': len(m.get('content') or ''),
                'reasoning_present': any(k in m for k in ['reasoning', 'reasoning_content'])} for m in kwargs['messages']]
            protocol['tool_names'] = [t['function']['name'] for t in kwargs.get('tools', [])]
            response = await create(**kwargs)
            data = response.model_dump(mode='json', exclude_none=True)
            choice = data['choices'][0]
            message = choice['message']
            protocol['response'] = {'finish_reason': choice.get('finish_reason'), 'model': data.get('model'),
                'message_keys': sorted(message), 'content_length': len(message.get('content') or ''),
                'tool_calls': message.get('tool_calls', []), 'usage': data.get('usage')}
            requests.append(protocol)
            runner.append_json(out / 'protocol.jsonl', protocol)
            return response
        agent.model.client.chat.completions.create = logged_create
        task = next(t for t in json.loads((ROOT / 'results/tasks.json').read_text()) if t['task_id'] == 'task-005')
        episode = asyncio.run(original.run_episode(agent, task, 'baseline'))
        (out / 'episode.json').write_text(json.dumps(episode, indent=2) + '\n')
        print(json.dumps({'requests': len(requests), 'actions': len(episode['trace']),
                          'goal': episode['goal_reached'], 'error': episode['error'], 'output': str(out)}), flush=True)
        self.assertGreater(len(requests), 0)
        self.assertTrue(all(r.get('temperature') == 0.7 and r.get('top_p') == 0.9 for r in requests))


@unittest.skipUnless(os.environ.get('RUN_PAID_CHOICE_PROBE') == '1', 'opt-in paid tool-choice probe')
class ToolChoiceProbe(unittest.TestCase):
    def test_auto_and_required_on_identical_context(self):
        from openai import AsyncOpenAI
        import runner
        source = ROOT / 'results/decision-record/pilot-01/task-011__baseline__history__r00.requests.jsonl'
        entries = [json.loads(line) for line in source.read_text().splitlines()]
        parameters = [e['parameters'] for e in entries if e['kind'] == 'request'][-1]
        out = Path(os.environ['PROBE_OUTPUT']).resolve()
        out.mkdir(parents=True, exist_ok=False)
        (out / 'context.json').write_text(json.dumps(parameters, indent=2) + '\n')

        choices = ['auto', 'required'] if os.environ.get('PROBE_USER_NUDGE') != '1' else ['auto', 'user_nudge']
        async def run():
            client = AsyncOpenAI(api_key=os.environ['HERMES_CUSTOM_CAILA_IO_API_KEY'],
                                 base_url='https://caila.io/api/openai/v1', timeout=120, max_retries=0)
            semaphore = asyncio.Semaphore(4)
            async def call(choice, repeat):
                async with semaphore:
                    p = {**parameters, 'tool_choice': choice if choice != 'user_nudge' else 'auto'}
                    if choice == 'user_nudge':
                        p['messages'] = [*parameters['messages'], {'role': 'user', 'content':
                            'The tool result above is the latest observation. Choose your next execute_action call.'}]
                    item = {'choice': choice, 'repeat': repeat, 'parameters': p}
                    try:
                        response = await client.chat.completions.create(**p)
                        item['response'] = runner.public_data(response.model_dump(mode='json', exclude_none=True))
                    except Exception as exc:
                        item['error'] = f'{type(exc).__name__}: {exc}'
                    runner.append_json(out / 'responses.jsonl', item)
                    return item
            try:
                return await asyncio.gather(*(call(c, r) for c in choices for r in range(4)))
            finally:
                await client.close()
        results = asyncio.run(run())
        counts = {}
        for choice in choices:
            subset = [r for r in results if r['choice'] == choice]
            counts[choice] = {'requests': len(subset),
                'tool_calls': sum(bool(r.get('response', {}).get('choices', [{}])[0].get('message', {}).get('tool_calls')) for r in subset),
                'errors': sum('error' in r for r in subset)}
        (out / 'summary.json').write_text(json.dumps(counts, indent=2) + '\n')
        print(json.dumps(counts), flush=True)
        self.assertEqual(len(results), 8)
        self.assertTrue(all(r['parameters']['temperature'] == 0.7 and r['parameters']['top_p'] == 0.9 for r in results))
