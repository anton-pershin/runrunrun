"""Domain-independent, budgeted execution of a sampled operational commitment."""
from __future__ import annotations
from typing import Any, Protocol


class Proposer(Protocol):
    async def __call__(self, *, observation: dict, failures: list[dict],
                       remaining_plan: list[str]) -> list[str]: ...


class Executor(Protocol):
    async def __call__(self, action: str) -> dict: ...


async def run_controller(initial_observation: dict, propose: Proposer,
                         execute: Executor, *, mode: str = 'commit',
                         max_actions: int = 50, max_replans: int = 12,
                         transient_retries: int = 1) -> dict[str, Any]:
    if mode not in {'commit', 'recede'} or min(max_actions, max_replans) < 1 or transient_retries < 0:
        raise ValueError('invalid mode or budgets')
    observation = initial_observation
    failures, remaining = [], []
    reason = 'initial'
    result = {'trace': [], 'executed_actions': [], 'plans': [], 'revisions': [], 'error': None}
    def finish(termination):
        result.update(termination=termination, goal_reached=observation.get('goal_reached') is True)
        return result
    try:
        while not observation.get('goal_reached'):
            if len(result['trace']) >= max_actions:
                return finish('action_limit')
            if len(result['plans']) >= max_replans:
                return finish('planning_limit')
            plan = await propose(observation=observation, failures=list(failures), remaining_plan=list(remaining))
            if not isinstance(plan, list) or not 1 <= len(plan) <= 50 or not all(isinstance(a, str) and a.strip() for a in plan):
                raise ValueError('plan must contain 1 to 50 nonempty operations')
            record = {'actions': list(plan), 'cursor': 0}
            result['plans'].append(record)
            result['revisions'].append({'reason': reason, 'plan_index': len(result['plans']) - 1})
            for index, action in enumerate(plan):
                retries = 0
                while True:
                    if len(result['trace']) >= max_actions:
                        return finish('action_limit')
                    observation = await execute(action)
                    result['trace'].append(observation)
                    result['executed_actions'].append(action)
                    if observation.get('goal_reached') is True:
                        if observation.get('status') == 'success':
                            record['cursor'] += 1
                        return finish('goal_reached')
                    status = observation.get('status')
                    if status == 'transient_failure' and retries < transient_retries:
                        retries += 1
                        continue
                    break
                if status == 'success':
                    record['cursor'] += 1
                    remaining = plan[index + 1:]
                    reason = 'receding' if mode == 'recede' else 'exhausted'
                    if mode == 'recede':
                        break
                else:
                    failures.append({'action': action, 'observation': observation})
                    remaining = plan[index:]
                    reason = 'failure'
                    break
        return finish('goal_reached')
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        return finish('error')
