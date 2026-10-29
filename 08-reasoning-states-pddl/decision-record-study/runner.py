"""Instrumented tool agent. No planning or action filtering in the policy."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
from pddl_env import BlocksWorldEnv, _ACTION_RE


def disturbance_eligible(env: BlocksWorldEnv, action: str, condition: str, injected: bool) -> bool:
    """Executor-only scheduling; reuse the executor's action grammar."""
    if condition != 'one_transient_failure' or injected or env.attempts < 2:
        return False
    parsed = _ACTION_RE.fullmatch(action.strip())
    if parsed is None:
        return False
    normalized = '(' + ' '.join(parsed.group(1).split()) + ')'
    op = env._operators.get(normalized)
    return op is not None and op.applicable(env.state)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Fact(StrictModel):
    fact: str
    observation_id: str


class LastOperation(StrictModel):
    action: str
    status: str
    observation_id: str


class Strategy(StrictModel):
    commitment: str
    revise_if: str


class DecisionRecord(StrictModel):
    objective: list[str]
    confirmed_facts: list[Fact]
    uncertainties: list[str]
    current_subgoal: str
    last_operation: LastOperation
    strategy: Strategy


ACTION_TOOL = {'type': 'function', 'function': {
    'name': 'execute_action', 'description': "Attempt one grounded PDDL action, such as '(pickup b1)'.",
    'parameters': {'type': 'object', 'properties': {'action': {'type': 'string'}},
                   'required': ['action'], 'additionalProperties': False}}}
RECORD_TOOL = {'type': 'function', 'function': {
    'name': 'update_decision_record', 'description': 'Save the current observation-grounded working task record.',
    'parameters': DecisionRecord.model_json_schema()}}
RECORD_INSTRUCTIONS = (
    'Update the active decision record using ONLY the previous record and latest observation. '
    'This is external working memory, not hidden reasoning. Return one update_decision_record call. '
    'Keep it compact. objective must preserve the user goal exactly. confirmed_facts are CURRENT '
    'true facts, copied exactly from observed state strings, with the observation_id supporting each. '
    'Remove stale facts when contradicted by the latest observation. Do not claim an attempted '
    'action succeeded unless the observation says so. List unresolved questions in uncertainties. '
    'current_subgoal is a brief operational target, not a detailed reasoning trace. '
    'last_operation must report the most recent action and status, or empty action and initial '
    'status at obs-0. strategy gives a short commitment and an observation-based condition for '
    'revising it. Preserve a useful commitment unless evidence warrants a change. '
    'Do not invent observations, action preconditions, or completed work.'
)


def record_messages(previous: dict | None, observation: dict) -> list[dict]:
    return [{'role': 'system', 'content': RECORD_INSTRUCTIONS},
            {'role': 'user', 'content': packed({'goal': observation['goal'],
                'previous_record': previous, 'latest_observation': observation})}]


def tool_arguments(message: dict, name: str) -> tuple[str, dict]:
    calls = message.get('tool_calls') or []
    if len(calls) != 1 or calls[0]['function']['name'] != name:
        raise ValueError(f'expected exactly one {name} call; got {len(calls)} calls')
    return calls[0]['id'], json.loads(calls[0]['function']['arguments'])


async def run_episode(task: dict, arm: str, condition: str, repeat: int,
                      request, max_actions: int = 50) -> dict:
    env = BlocksWorldEnv(ROOT / 'pddl/domain.pddl', ROOT / task['problem_file'])
    initial = env.snapshot()
    observation = {'id': 'obs-0', **initial}
    history = [{'role': 'user', 'content': packed({'task_id': task['task_id'], 'initial_observation': observation})}]
    record = None
    trace, records = [], []
    injected = False
    error = None
    termination = 'action_limit'
    stage = 'action'
    final_response = ''
    try:
        for index in range(max_actions):
            if env.solved:
                termination = 'observed_goal'
                break
            if arm == 'record':
                stage = 'record'
                response = await request(record_messages(record, observation), [RECORD_TOOL], 'record', index)
                if not response.get('tool_calls'):
                    final_response = response.get('content') or ''
                    termination = 'model_stopped' if final_response else 'empty_response'
                    break
                _, payload = tool_arguments(response, 'update_decision_record')
                new_record = DecisionRecord.model_validate(payload).model_dump()
                records.append({'before_action': index + 1, 'observation_id': observation['id'],
                                'previous_record': record, 'record': new_record})
                record = new_record
            stage = 'action'
            messages = action_messages(arm, observation, record, history, env.list_available_actions())
            response = await request(messages, [ACTION_TOOL], 'action', index)
            if not response.get('tool_calls'):
                final_response = response.get('content') or ''
                termination = 'model_stopped' if final_response else 'empty_response'
                break
            call_id, payload = tool_arguments(response, 'execute_action')
            action = payload['action']
            if not isinstance(action, str):
                raise ValueError('action must be a string')
            # Executor only: preserve the original one-transient-failure rule.
            inject = disturbance_eligible(env, action, condition, injected)
            transition = env.step(action, transient_failure=inject)
            injected |= inject
            trace.append(transition)
            observation = {'id': f'obs-{len(trace)}', **transition}
            history.extend([response, {'role': 'tool', 'tool_call_id': call_id, 'content': packed(observation)}])
            if transition['goal_reached']:
                termination = 'observed_goal'
                break
    except Exception as exc:
        termination = 'error'
        error = f'{type(exc).__name__}: {exc}'
    return {'task_id': task['task_id'], 'problem_file': task['problem_file'], 'arm': arm,
            'condition': condition, 'repeat': repeat, 'max_actions': max_actions,
            'initial': initial, 'trace': trace, 'records': records, 'final': env.snapshot(),
            'goal_reached': env.solved, 'disturbance_injected': injected,
            'termination': termination, 'termination_stage': stage,
            'final_response': final_response, 'error': error}


def verify_episode(episode: dict) -> dict:
    errors = []
    env = BlocksWorldEnv(ROOT / 'pddl/domain.pddl', ROOT / episode['problem_file'])
    if env.snapshot() != episode['initial']:
        errors.append('initial mismatch')
    injected = False
    for i, event in enumerate(episode['trace']):
        inject = disturbance_eligible(env, event['action'], episode['condition'], injected)
        actual = env.step(event['action'], transient_failure=inject)
        injected |= inject
        if actual != event:
            errors.append(f'event {i + 1} mismatch')
        if actual['goal_reached'] and i + 1 != len(episode['trace']):
            errors.append('post-goal actions')
    if env.snapshot() != episode['final'] or env.solved != episode['goal_reached']:
        errors.append('final mismatch')
    failures = [e for e in episode['trace'] if e['status'] == 'transient_failure']
    if bool(failures) != episode['disturbance_injected'] or len(failures) > 1:
        errors.append('disturbance flag/count mismatch')
    if failures and (episode['condition'] != 'one_transient_failure' or failures[0]['attempt'] < 3):
        errors.append('invalid disturbance condition/timing')
    if len(episode['trace']) > episode['max_actions']:
        errors.append('action budget exceeded')
    term = episode['termination']
    if (term == 'observed_goal') != episode['goal_reached']:
        errors.append('termination/goal mismatch')
    if term == 'action_limit' and len(episode['trace']) != episode['max_actions']:
        errors.append('action-limit termination before budget')
    if bool(episode['error']) != (term == 'error'):
        errors.append('termination/error mismatch')
    if injected != episode['disturbance_injected']:
        errors.append('disturbance schedule mismatch')
    if episode['arm'] == 'record' and len(episode['records']) < len(episode['trace']):
        errors.append('missing records')
    return {'valid': not errors, 'errors': errors}


ACTION_INSTRUCTIONS = (
    "You control a Blocks World through execute_action. The complete grounded action "
    "vocabulary is provided; it includes infeasible actions and does not filter moves. "
    "Choose exactly one action at a time, read its result, then continue toward the goal. "
    "Use exact action spellings. If an action fails, use the observed result; do not "
    "repeat an unchanged failed action indefinitely. A tool result is authoritative "
    "about what actually happened. Do not explain hidden reasoning. If you cannot "
    "continue, return a brief statement instead of claiming success."
)


def packed(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def action_messages(arm: str, observation: dict, record: dict | None,
                    history: list[dict], catalog: list[str]) -> list[dict]:
    system = {'role': 'system', 'content': ACTION_INSTRUCTIONS + '\nFull fixed action vocabulary: ' + packed(catalog)}
    decision_request = {'role': 'user', 'content':
                        'The latest observation is above. Choose your next execute_action call.'}
    if arm == 'history':
        return [system, *history, decision_request]
    if arm == 'reset':
        context = {'state': sorted(observation['state']), 'goal': sorted(observation['goal'])}
    elif arm == 'record':
        context = {'goal': observation['goal'], 'active_record': record, 'latest_observation': observation}
        system = dict(system, content=system['content'] +
                      '\nUse the active decision record as your working task state. '
                      'Use the latest observation to resolve conflicts. Keep the current '
                      'subgoal unless observed evidence warrants revision. The record can be wrong.')
    else:
        raise ValueError(arm)
    return [system, {'role': 'user', 'content': packed(context)}, decision_request]


def public_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: public_data(v) for k, v in value.items()
                if k not in {'reasoning_content', 'reasoning', 'thinking', 'reasoning_details'}}
    if isinstance(value, list):
        return [public_data(v) for v in value]
    return value


def append_json(path: Path, data: dict) -> None:
    with path.open('a') as stream:
        stream.write(packed(data) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


class APIRecorder:
    def __init__(self, client, path: Path, model: str, temperature: float, top_p: float):
        self.client, self.path, self.model = client, path, model
        self.temperature, self.top_p = temperature, top_p

    async def __call__(self, messages: list[dict], tools: list[dict], stage: str, tag: int) -> dict:
        parameters = {'model': self.model, 'messages': messages, 'tools': tools,
                      'temperature': self.temperature, 'top_p': self.top_p,
                      'parallel_tool_calls': False, 'tool_choice': 'auto', 'max_tokens': 8192}
        stamp = {'stage': stage, 'index': tag, 'utc': datetime.now(timezone.utc).isoformat()}
        append_json(self.path, {'kind': 'request', **stamp, 'parameters': public_data(parameters)})
        started = time.monotonic()
        try:
            result = await self.client.chat.completions.create(**parameters)
            raw = result.model_dump(mode='json', exclude_none=True)
            data = public_data(raw)
            append_json(self.path, {'kind': 'response', **stamp, 'latency_s': time.monotonic() - started, 'data': data})
            if data['choices'][0].get('finish_reason') == 'length':
                raise RuntimeError('completion token limit reached before a usable decision')
            message = raw['choices'][0]['message']
            # Preserve native provider continuation fields in memory only. The log redacts them.
            return {k: v for k, v in message.items() if k in {
                'role', 'content', 'tool_calls', 'reasoning_content', 'reasoning'}}
        except Exception as exc:
            append_json(self.path, {'kind': 'error', **stamp, 'latency_s': time.monotonic() - started,
                                    'error': f'{type(exc).__name__}: {exc}'})
            raise


def configuration(args) -> dict:
    tasks = json.loads((ROOT / 'results/tasks.json').read_text())
    selected = [t for t in tasks if t['task_id'] in args.tasks]
    if set(args.tasks) != {t['task_id'] for t in selected}:
        raise ValueError('unknown task id')
    jobs = [{'task_id': t['task_id'], 'arm': arm, 'condition': cond, 'repeat': rep}
            for t, arm, cond, rep in itertools.product(selected, args.arms, args.conditions, range(args.repeats))]
    random.Random(20261028).shuffle(jobs)
    source_paths = [Path(__file__), ROOT / 'lib/pddl_env.py', ROOT / 'pddl/domain.pddl',
                    ROOT / 'results/tasks.json', Path(__file__).with_name('PROTOCOL.md')]
    source_paths += [ROOT / t['problem_file'] for t in selected]
    return {'model': args.model, 'base_url': args.base_url, 'temperature': args.temperature,
            'top_p': args.top_p, 'seed': None, 'max_tokens': 8192, 'max_actions': args.max_actions,
            'concurrency': args.concurrency, 'jobs': jobs, 'python': sys.version,
            'platform': platform.platform(), 'created_utc': datetime.now(timezone.utc).isoformat(),
            'dependencies': {p: importlib.metadata.version(p) for p in ['openai', 'pydantic', 'pydantic-ai-slim']},
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'git_status': subprocess.check_output(['git', 'status', '--short', '--branch'], cwd=ROOT, text=True),
            'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}


async def main(args) -> None:
    from openai import AsyncOpenAI
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = configuration(args)
    (out / 'manifest.json').write_text(json.dumps(config, indent=2) + '\n')
    # Preserve exact executable and prompt sources with each experiment.
    (out / 'runner.snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out / 'protocol.snapshot.md').write_bytes(Path(__file__).with_name('PROTOCOL.md').read_bytes())
    if args.dry_run:
        print(f'dry-run jobs={len(config["jobs"])} output={out}', flush=True)
        return
    key = os.environ.get('HERMES_CUSTOM_CAILA_IO_API_KEY')
    if not key:
        raise RuntimeError('HERMES_CUSTOM_CAILA_IO_API_KEY is not set')
    client = AsyncOpenAI(api_key=key, base_url=args.base_url, timeout=120, max_retries=0)
    tasks = {t['task_id']: t for t in json.loads((ROOT / 'results/tasks.json').read_text())}
    semaphore = asyncio.Semaphore(args.concurrency)

    async def execute(job):
        async with semaphore:
            name = f'{job["task_id"]}__{job["condition"]}__{job["arm"]}__r{job["repeat"]:02d}'
            request = APIRecorder(client, out / f'{name}.requests.jsonl', args.model, args.temperature, args.top_p)
            started = time.monotonic()
            episode = await run_episode(tasks[job['task_id']], job['arm'], job['condition'],
                                        job['repeat'], request, args.max_actions)
            episode.update(model=args.model, elapsed_s=time.monotonic() - started,
                           request_log=f'{name}.requests.jsonl')
            episode['verification'] = verify_episode(episode)
            (out / f'{name}.episode.json').write_text(json.dumps(episode, indent=2) + '\n')
            print(f'{name} goal={episode["goal_reached"]} actions={len(episode["trace"])} '
                  f'end={episode["termination"]} verified={episode["verification"]["valid"]}', flush=True)
            return episode
    try:
        episodes = await asyncio.gather(*(execute(job) for job in config['jobs']))
        (out / 'episodes.json').write_text(json.dumps(episodes, indent=2) + '\n')
        print(f'completed={len(episodes)} goals={sum(e["goal_reached"] for e in episodes)} output={out}', flush=True)
    finally:
        await client.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='qwen/qwen3.5-9b')
    parser.add_argument('--base-url', default='https://caila.io/api/openai/v1')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--tasks', nargs='+', default=['task-005', 'task-007', 'task-011', 'task-012'])
    parser.add_argument('--arms', nargs='+', choices=['history', 'reset', 'record'], default=['history', 'reset', 'record'])
    parser.add_argument('--conditions', nargs='+', choices=['baseline', 'one_transient_failure'], default=['baseline', 'one_transient_failure'])
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--max-actions', type=int, default=50)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or min(args.repeats, args.max_actions, args.concurrency) < 1:
        parser.error('require positive temperature/counts and 0 < top-p <= 1')
    return args


if __name__ == '__main__':
    asyncio.run(main(parse_args()))
