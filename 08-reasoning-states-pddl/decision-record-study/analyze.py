"""Offline metrics. These checks never feed corrections to the agent."""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from runner import verify_episode


def episode_metrics(episode: dict) -> dict:
    trace = episode['trace']
    statuses = Counter(x['status'] for x in trace)
    visited = {tuple(episode['initial']['state'])}
    failed = set()
    repeated_failures = revisits = 0
    for event in trace:
        key = (tuple(event['state_before']), event['action'])
        if event['status'] not in {'success', 'transient_failure'}:
            repeated_failures += key in failed
            failed.add(key)
        if event['status'] == 'success':
            revisits += tuple(event['state']) in visited
            visited.add(tuple(event['state']))
    first_goal = next((i + 1 for i, e in enumerate(trace) if e['goal_reached']), None)
    observations = {'obs-0': {'id': 'obs-0', **episode['initial']}}
    observations.update({f'obs-{i + 1}': {'id': f'obs-{i + 1}', **e} for i, e in enumerate(trace)})
    qualities = []
    for version in episode['records']:
        obs = observations[version['observation_id']]
        available = {k: v for k, v in observations.items() if int(k.split('-')[1]) <= int(obs['id'].split('-')[1])}
        quality = record_quality(version['record'], obs, available)
        quality['before_action'] = version['before_action']
        quality['action_status'] = trace[version['before_action'] - 1]['status'] if version['before_action'] <= len(trace) else None
        qualities.append(quality)
    return {'actions': len(trace), 'statuses': dict(statuses), 'first_goal_action': first_goal,
            'post_goal_actions': len(trace) - first_goal if first_goal is not None else 0,
            'repeated_failed_state_action': repeated_failures, 'successful_state_revisits': revisits,
            'record_quality': qualities}


def render_record_timeline(episode: dict) -> str:
    lines = [f"{episode['task_id']} | {episode['arm']} | {episode['condition']} | repeat {episode['repeat']}",
             'Declared working records, not hidden neural states or chain-of-thought.',
             'Goal: ' + json.dumps(episode['initial']['goal']),
             f"Final goal: {episode['goal_reached']} | termination: {episode['termination']}", '']
    for version in episode['records']:
        index = version['before_action']
        lines.extend([f"Decision {index} | based on {version['observation_id']}",
                      json.dumps(version['record'], indent=2, ensure_ascii=False)])
        if index <= len(episode['trace']):
            event = episode['trace'][index - 1]
            lines.extend(['Selected action: ' + event['action'], 'Observed status: ' + event['status'],
                          'Observed details: ' + json.dumps(event['details'])])
        else:
            lines.append('No action was executed after this record.')
        lines.append('')
    return '\n'.join(lines) + '\n'


def first_action_agreement(episodes: list[dict]) -> dict:
    first = [e['trace'][0]['action'] for e in episodes if e['trace']]
    pairs = list(itertools.combinations(first, 2))
    return {'no_first_action': len(episodes) - len(first),
            'agreement': sum(a == b for a, b in pairs) / len(pairs) if pairs else None}


def summarize(directory: Path) -> dict:
    manifest = json.loads((directory / 'manifest.json').read_text())
    paths = sorted(directory.glob('*.episode.json'))
    episodes = [json.loads(p.read_text()) for p in paths]
    grouped = defaultdict(list)
    rows = []
    models = Counter()
    request_settings = Counter()
    for path, episode in zip(paths, episodes):
        metrics = episode_metrics(episode)
        verification = verify_episode(episode)
        row = {k: episode[k] for k in ['task_id', 'arm', 'condition', 'repeat', 'goal_reached', 'termination', 'error']}
        row.update(metrics, verification=verification, file=path.name,
                   requests=0, responses=0, prompt_tokens=0, completion_tokens=0, latency_s=0)
        log = directory / episode['request_log']
        for line in log.read_text().splitlines():
            item = json.loads(line)
            if item['kind'] == 'request':
                row['requests'] += 1
                p = item['parameters']
                request_settings[(p['model'], p['temperature'], p['top_p'], p['max_tokens'], p['parallel_tool_calls'])] += 1
            elif item['kind'] == 'response':
                row['responses'] += 1
                models[item['data'].get('model', 'missing')] += 1
                usage = item['data'].get('usage', {})
                row['prompt_tokens'] += usage.get('prompt_tokens', 0)
                row['completion_tokens'] += usage.get('completion_tokens', 0)
                row['latency_s'] += item['latency_s']
        rows.append(row)
        grouped[episode['arm']].append((episode, row))
    arms = {}
    for arm, pairs in grouped.items():
        eps = [e for e, _ in pairs]
        ms = [r for _, r in pairs]
        cells = defaultdict(list)
        for e in eps:
            cells[(e['task_id'], e['condition'])].append(e)
        agreements = []
        mixed = 0
        for cell in cells.values():
            mixed += len({e['goal_reached'] for e in cell}) > 1
            metric = first_action_agreement(cell)
            if metric['agreement'] is not None:
                agreements.append(metric['agreement'])
        statuses = Counter()
        for r in ms:
            statuses.update(r['statuses'])
        quality = [q for r in ms for q in r['record_quality']]
        arms[arm] = {'episodes': len(eps), 'goals': sum(e['goal_reached'] for e in eps),
                     'by_condition': {c: {'episodes': sum(e['condition'] == c for e in eps),
                                         'goals': sum(e['condition'] == c and e['goal_reached'] for e in eps)}
                                      for c in sorted({e['condition'] for e in eps})},
                     'terminations': dict(Counter(e['termination'] for e in eps)),
                     'mean_actions': statistics.mean(r['actions'] for r in ms),
                     'median_actions': statistics.median(r['actions'] for r in ms),
                     'total_actions': sum(r['actions'] for r in ms), 'statuses': dict(statuses),
                     'repeated_failed_state_action': sum(r['repeated_failed_state_action'] for r in ms),
                     'successful_state_revisits': sum(r['successful_state_revisits'] for r in ms),
                     'mixed_outcome_cells': mixed, 'observed_cells': len(cells),
                     'no_first_action': sum(not e['trace'] for e in eps),
                     'mean_pairwise_first_action_agreement': statistics.mean(agreements) if agreements else None,
                     'injections': sum(e['disturbance_injected'] for e in eps),
                     'requests': sum(r['requests'] for r in ms),
                     'prompt_tokens': sum(r['prompt_tokens'] for r in ms),
                     'completion_tokens': sum(r['completion_tokens'] for r in ms),
                     'records': len(quality),
                     'record_totals': {k: sum(q[k] for q in quality) for k in
                         ['claims', 'unsupported_claims', 'false_current_claims', 'goal_drift', 'last_operation_mismatch']},
                     'mean_record_fact_coverage': statistics.mean(q['current_fact_coverage'] for q in quality) if quality else None,
                     'precondition_failures_after_factually_clean_record': sum(q['action_status'] == 'precondition_failed' and
                         q['unsupported_claims'] == q['false_current_claims'] == q['goal_drift'] == q['last_operation_mismatch'] == 0
                         for q in quality)}
    expected = {(j['task_id'], j['arm'], j['condition'], j['repeat']) for j in manifest['jobs']}
    actual = [(e['task_id'], e['arm'], e['condition'], e['repeat']) for e in episodes]
    return {'planned_episodes': len(manifest['jobs']), 'completed_episodes': len(episodes),
            'missing_jobs': [list(k) for k in sorted(expected - set(actual))],
            'duplicate_jobs': len(actual) - len(set(actual)), 'unexpected_jobs': [list(k) for k in set(actual) - expected],
            'all_replays_valid': bool(rows) and all(r['verification']['valid'] for r in rows),
            'response_models': dict(models),
            'request_settings': [{'model': k[0], 'temperature': k[1], 'top_p': k[2], 'max_tokens': k[3],
                                  'parallel_tool_calls': k[4], 'count': n} for k, n in request_settings.items()],
            'arms': arms, 'episodes': rows}


def record_quality(record: dict, observation: dict, observations: dict[str, dict]) -> dict:
    claims = record['confirmed_facts']
    supported = sum(c['observation_id'] in observations and c['fact'] in observations[c['observation_id']]['state']
                    for c in claims)
    current = set(observation['state'])
    last = record['last_operation']
    expected_last = {'action': observation.get('action', ''), 'status': observation.get('status', 'initial'),
                     'observation_id': observation['id']}
    return {'claims': len(claims), 'unsupported_claims': len(claims) - supported,
            'false_current_claims': sum(c['fact'] not in current for c in claims),
            'current_fact_coverage': len({c['fact'] for c in claims} & current) / max(1, len(current)),
            'goal_drift': int(sorted(record['objective']) != sorted(observation['goal'])),
            'last_operation_mismatch': int(last != expected_last)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timelines', action='store_true', help='Write readable declared-record histories beside the summary')
    args = parser.parse_args()
    result = summarize(args.directory)
    if args.timelines:
        timelines = args.output.parent / 'record-timelines'
        timelines.mkdir(parents=True, exist_ok=True)
        for path in sorted(args.directory.glob('*.episode.json')):
            episode = json.loads(path.read_text())
            if episode['records']:
                (timelines / path.name.replace('.episode.json', '.txt')).write_text(render_record_timeline(episode))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'episodes'}, indent=2))
    print(f'saved={args.output.resolve()}')
