"""Bounded Caila/Qwen interface diagnosis, never an episode controller."""
from __future__ import annotations

import json
import re
from pathlib import Path

REASONING_KEYS = {'reasoning_content', 'reasoning', 'thinking', 'reasoning_details'}


def public_data(value, secrets=()):
    """Copy JSON data, removing private reasoning and credentials before persistence."""
    if isinstance(value, dict):
        return {k: public_data(v, secrets) for k, v in value.items() if k not in REASONING_KEYS}
    if isinstance(value, list):
        return [public_data(v, secrets) for v in value]
    if isinstance(value, str):
        value = re.sub(r'<think(?:ing)?>.*?(?:</think(?:ing)?>|$)', '', value, flags=re.S | re.I)
        for secret in secrets:
            if secret:
                value = value.replace(secret, '[REDACTED]')
    return value


def reasoning_metadata(raw: dict) -> dict:
    """Only counts/presence, never excerpts or generated private reasoning."""
    choices = raw.get('choices') or []
    message = choices[0].get('message', {}) if choices else {}
    fields = {k: len(v) if isinstance(v, str) else len(json.dumps(v))
              for k, v in message.items() if k in REASONING_KEYS and v is not None}
    return {'native_reasoning_characters': sum(fields.values()), 'native_reasoning_fields': fields,
            'content_has_think_tags': bool(re.search(r'<think(?:ing)?>', message.get('content') or '', re.I))}

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'qwen/qwen3.5-9b'
BASE_URL = 'https://caila.io/api/openai/v1'
ACTION_TOOL = {'type': 'function', 'function': {
    'name': 'execute_action',
    'description': "Attempt one grounded PDDL action, such as '(pickup b1)'.",
    'parameters': {'type': 'object', 'properties': {'action': {'type': 'string'}},
                   'required': ['action'], 'additionalProperties': False}}}
DOCS = (
    'You control a Blocks World. The observation gives the complete current true facts '
    'and the goal facts, all of which must become true. Use exact grounded action spellings. '
    'The fixed vocabulary below includes infeasible actions; it is NOT a list of legal moves. '
    'Ordinary action documentation: pickup x requires clear x, on-table x, arm-empty; '
    'it adds holding x and removes clear x, on-table x, arm-empty. '
    'putdown x requires holding x; it adds clear x, arm-empty, on-table x and removes holding x. '
    'unstack x y requires on x y, clear x, arm-empty; it adds holding x, clear y and removes '
    'on x y, clear x, arm-empty. stack x y requires holding x, clear y; it adds on x y, '
    'clear x, arm-empty and removes clear y, holding x. Facts not changed by an action persist. '
    'Return only the requested operational output, with no explanation or hidden reasoning. '
)


def build_request(context: dict, interface: str, thinking: str) -> dict:
    formats = {
        'tool': 'Choose exactly one next action toward the goal by calling execute_action once.',
        'json_action': 'Choose exactly one next action toward the goal. Return only a JSON object '
                       'with a single string field "action" containing the grounded action.',
        'json_plan': 'Return a complete sequential action plan from the current state to all goal facts. '
                     'Return only a JSON object with an "actions" array of grounded action strings. '
                     'The array must contain at most 50 actions.',
    }
    if thinking not in {'default', 'disabled'}:
        raise ValueError('unknown thinking mode')
    parameters = {
        'model': MODEL, 'temperature': 0.7, 'top_p': 0.9, 'max_tokens': 8192,
        'messages': [
            {'role': 'system', 'content': DOCS + '\nFull fixed action vocabulary: ' + json.dumps(context['catalog'])},
            {'role': 'user', 'content': json.dumps({'task_id': context['task_id'], 'observation': context['observation']}, sort_keys=True)},
            {'role': 'user', 'content': formats[interface]},
        ],
    }
    if interface == 'tool':
        parameters.update(tools=[ACTION_TOOL], parallel_tool_calls=False, tool_choice='auto')
    if thinking == 'disabled':
        parameters['extra_body'] = {'chat_template_kwargs': {'enable_thinking': False}}
    return parameters


def load_contexts():
    import sys
    sys.path.insert(0, str(ROOT / 'lib'))
    from pddl_env import BlocksWorldEnv
    task = next(t for t in json.loads((ROOT / 'results/tasks.json').read_text()) if t['task_id'] == 'task-005')
    env = BlocksWorldEnv(ROOT / 'pddl/domain.pddl', ROOT / task['problem_file'])
    initial = {'name': 'initial', 'task_id': task['task_id'], 'observation': env.snapshot(),
               'catalog': env.list_available_actions(), 'prefix': []}
    event = env.step('(unstack b4 b5)')
    checkpoint = {'name': 'checkpoint', 'task_id': task['task_id'], 'observation': event,
                  'catalog': env.list_available_actions(), 'prefix': ['(unstack b4 b5)']}
    return [initial, checkpoint]


def score_response(context, interface, raw):
    import sys
    sys.path.insert(0, str(ROOT / 'lib'))
    from pddl_env import BlocksWorldEnv
    result = {'usable': False, 'all_actions_feasible': False, 'first_action_feasible': False,
              'goal_reached': False, 'first_failure_index': None}
    try:
        choice = raw['choices'][0]
        if choice.get('finish_reason') == 'length':
            return result
        message = public_data(choice['message'])
        if interface == 'tool':
            calls = message.get('tool_calls', [])
            if len(calls) != 1 or calls[0]['function']['name'] != 'execute_action':
                return result
            actions = [json.loads(calls[0]['function']['arguments'])['action']]
        else:
            text = (message.get('content') or '').strip()
            if text.startswith('```') and text.endswith('```'):
                text = '\n'.join(text.splitlines()[1:-1])
            payload = json.loads(text)
            actions = payload['actions'] if interface == 'json_plan' else [payload['action']]
        if not isinstance(actions, list) or not 1 <= len(actions) <= 50 or not all(isinstance(a, str) and a.strip() for a in actions):
            return result
        result['usable'] = True
        task = next(t for t in json.loads((ROOT / 'results/tasks.json').read_text()) if t['task_id'] == context['task_id'])
        env = BlocksWorldEnv(ROOT / 'pddl/domain.pddl', ROOT / task['problem_file'])
        for action in context['prefix']:
            env.step(action)
        statuses = []
        for index, action in enumerate(actions):
            event = env.step(action)
            statuses.append(event['status'] == 'success')
            if not statuses[-1] and result['first_failure_index'] is None:
                result['first_failure_index'] = index
        result.update(first_action_feasible=statuses[0], all_actions_feasible=all(statuses), goal_reached=env.solved)
    except (KeyError, IndexError, ValueError, TypeError):
        pass
    return result
