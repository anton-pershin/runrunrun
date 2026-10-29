"""Public operational plans over a sampled model API. No simulator imports."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path


TOOL_CONTRACT = (
    'You operate a Blocks World using a tool that attempts one operation. '
    'Facts (on X Y) mean X is directly on Y; (on-table X) means X is on the table; '
    '(clear X) means nothing is on X; (holding X) means the arm holds X; '
    '(arm-empty) means the arm is free. '
    'Operations: (pickup X) lifts a clear block from the table with the empty arm; '
    '(unstack X Y) lifts clear X off Y with the empty arm; '
    '(putdown X) puts the held X on the table; '
    '(stack X Y) places the held X onto clear Y. '
    'The arm holds at most one block. The observed tool result is authoritative. '
    'The goal is a conjunction: every listed goal fact must hold at the same time. '
)
PLAN_INSTRUCTIONS = (
    TOOL_CONTRACT +
    'Return a compact operational plan to complete the task from the CURRENT observation. '
    'Respect dependencies between operations. Do not treat an intended operation as completed. '
    'Use previous remaining operations if they are still useful; revise them if observations '
    'or recorded failures contradict them. Do not repeat a rejected operation unless its '
    'observed blocking condition has changed. Transient failures are not semantic rejections. '
    'Output only a JSON object {"actions": ["(operation arguments)", ...]}. '
    'List primitive operations in execution order, at most 50; no commentary, explanations, '
    'reasoning trace, placeholders, conditionals, or invented tool observations.'
)


def plan_messages(observation: dict, failures: list, remaining_plan: list[str]) -> list[dict]:
    return [{'role': 'system', 'content': PLAN_INSTRUCTIONS},
            {'role': 'user', 'content': json.dumps({'observation': observation,
             'failures': failures, 'remaining_plan': remaining_plan}, sort_keys=True)}]


def visible_content(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    if '<think>' in text:
        text = text.split('<think>', 1)[0]
    return text.strip()


def parse_plan(text: str) -> list[str]:
    text = visible_content(text)
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.splitlines()[1:-1]).strip()
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError('response is not a JSON operational plan') from exc
    actions = payload.get('actions') if isinstance(payload, dict) else None
    if not isinstance(actions, list) or not 1 <= len(actions) <= 50:
        raise ValueError('plan must contain 1 to 50 operations')
    if not all(isinstance(a, str) and a.strip() for a in actions):
        raise ValueError('all operations must be nonempty strings')
    return actions


def public_response(value):
    if isinstance(value, dict):
        return {k: visible_content(v) if k == 'content' and isinstance(v, str) else public_response(v)
                for k, v in value.items() if k not in {'reasoning', 'reasoning_content', 'thinking', 'reasoning_details'}}
    if isinstance(value, list):
        return [public_response(v) for v in value]
    return value


def append_event(path: Path, event: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(event, sort_keys=True) + '\n')
        stream.flush()


class PlanProposer:
    def __init__(self, client, path: Path, *, model='qwen/qwen3.5-9b', thinking='default',
                 max_tokens=8192, max_requests=100, output_format='json'):
        self.output_format = output_format
        self.client, self.path, self.model = client, path, model
        self.thinking, self.max_tokens, self.max_requests = thinking, max_tokens, max_requests
        self.requests = 0

    async def __call__(self, *, observation, failures, remaining_plan):
        if self.requests >= self.max_requests:
            raise RuntimeError('planning request budget exhausted')
        self.requests += 1
        parameters = {'model': self.model, 'messages': plan_messages(observation, failures, remaining_plan),
                      'temperature': 0.7, 'top_p': 0.9, 'max_tokens': self.max_tokens}
        if self.output_format in {'tool', 'review'}:
            parameters['tools'] = [{'type':'function','function':{'name':'submit_plan',
                'description':'Submit an ordered list of primitive tool operations to execute toward the goal.',
                'parameters':{'type':'object','properties':{'actions':{'type':'array','items':{'type':'string'}}},
                              'required':['actions'],'additionalProperties':False}}}]
            parameters['parallel_tool_calls'] = False
            parameters['tool_choice'] = 'auto'
            parameters['messages'] = [{'role':'system','content':TOOL_CONTRACT +
                'Use submit_plan to provide an ordered operational plan to achieve all goal facts. '
                'Use the observed state as authoritative. Account for what each operation changes. '
                'If a previous plan failed, revise it using the reported outcome. Submit at most 50 operations.'},
                parameters['messages'][-1]]
        if self.output_format == 'review':
            parameters['messages'][0]['content'] += (
                ' Audit the candidate in remaining_plan before any operation is executed. '
                'It is an untrusted draft, not a history of successful actions. '
                'Check each operation against the current observation and the effects of earlier '
                'operations in the draft. Check resource occupancy, dependencies, and all goal facts. '
                'Fix omissions and contradictions. Use submit_plan to return the complete corrected '
                'primitive operation list, or the unchanged list if it is already correct. '
                'Do not report imagined outcomes as observations. No explanation is required.')
        if self.thinking == 'off-native':
            parameters['extra_body'] = {'enable_thinking': False}
        elif self.thinking != 'default':
            parameters['extra_body'] = {'chat_template_kwargs': {'enable_thinking': self.thinking == 'on'}}
        append_event(self.path, {'kind': 'request', 'index': self.requests, 'parameters': parameters})
        start = time.monotonic()
        try:
            result = await self.client.chat.completions.create(**parameters)
            data = public_response(result.model_dump(mode='json', exclude_none=True))
            append_event(self.path, {'kind': 'response', 'index': self.requests,
                                    'latency_s': time.monotonic() - start, 'data': data})
            if data['choices'][0].get('finish_reason') == 'length':
                raise RuntimeError('completion token budget exhausted')
            message = data['choices'][0]['message']
            if self.output_format in {'tool', 'review'}:
                calls = message.get('tool_calls') or []
                if len(calls) != 1 or calls[0]['function']['name'] != 'submit_plan':
                    raise ValueError('expected one submit_plan tool call')
                return parse_plan(calls[0]['function']['arguments'])
            return parse_plan(message.get('content') or '')
        except Exception as exc:
            append_event(self.path, {'kind': 'error', 'index': self.requests,
                                    'latency_s': time.monotonic() - start,
                                    'error': f'{type(exc).__name__}: {exc}'})
            raise
