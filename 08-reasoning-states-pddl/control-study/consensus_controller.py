"""Sample-and-agree operational commitment; no domain model or planner."""
from collections import Counter
from controller import run_controller


async def run_consensus_controller(initial_observation, propose, execute, *,
                                   samples=3, min_support=None, max_votes=12,
                                   max_actions=50, transient_retries=1, execution_mode='commit'):
    if min_support is None:
        min_support = samples // 2 + 1
    if samples < 1 or not 1 <= min_support <= samples or max_votes < 1:
        raise ValueError('invalid voting budgets')
    votes, selected = [], []

    async def agree(**context):
        while len(votes) < max_votes:
            record = {'reason': 'initial' if not votes else
                      ('failure' if context['failures'] else 'exhausted'),
                      'failures': list(context['failures']), 'candidates': [], 'errors': []}
            votes.append(record)
            for _ in range(samples):
                try:
                    plan = await propose(**context)
                    if not isinstance(plan, list) or not 1 <= len(plan) <= 50 or not all(
                            isinstance(a, str) and a.strip() for a in plan):
                        raise ValueError('invalid sampled plan')
                    record['candidates'].append(list(plan))
                except Exception as exc:
                    record['errors'].append(f'{type(exc).__name__}: {exc}')
            counts = Counter(p[0] for p in record['candidates'])
            if not counts or max(counts.values()) < min_support:
                record['reason'] = 'no_quorum'
                continue
            first = min(counts, key=lambda a: (-counts[a], a))
            plans = Counter(tuple(p) for p in record['candidates'] if p[0] == first)
            winner = min(plans, key=lambda p: (-plans[p], p))
            record.update(first_action=first, support=counts[first], selected=list(winner))
            selected.append({'actions': list(winner), 'support': counts[first]})
            return list(winner)
        raise RuntimeError('vote budget exhausted')

    result = await run_controller(initial_observation, agree, execute, mode=execution_mode,
                                  max_actions=max_actions, max_replans=max_votes,
                                  transient_retries=transient_retries)
    result['selected_plans'] = selected
    result['revisions'] = votes
    return result
