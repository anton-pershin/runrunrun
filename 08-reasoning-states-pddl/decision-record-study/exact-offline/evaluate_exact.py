#!/usr/bin/env python3
"""Offline-only exact Blocks World evaluation; no policy/model imports or calls.
Run with the repository's existing .venv/bin/python and PYTHONDONTWRITEBYTECODE=1.
All writes are confined to --output (default: this script's scratch directory).
"""
import argparse
from collections import Counter, deque, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
DEFAULT_REPO = '/home/tony/reps/github/anton-pershin/runrunrun/08-reasoning-states-pddl'

def dump(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')

def op_signature(ops):
    return [(o.name, sorted(o.preconditions), sorted(o.add_effects), sorted(o.del_effects)) for o in sorted(ops, key=lambda o: o.name)]

def build_graph(env, cap):
    ops = sorted(env.task.operators, key=lambda o: o.name)
    initial = frozenset(env.task.initial_state)
    states = [initial]
    index = {initial: 0}
    edges = []
    for state in states:
        outgoing = []
        for oi, op in enumerate(ops):
            if not op.applicable(state):
                continue
            nxt = frozenset(op.apply(state))
            if nxt not in index:
                if len(states) >= cap:
                    raise RuntimeError(f'State cap {cap} exceeded: stop rather than expand unexpectedly huge graph')
                index[nxt] = len(states)
                states.append(nxt)
            outgoing.append((oi, index[nxt]))
        edges.append(outgoing)
    reverse = [[] for _ in states]
    successors = [{j for _, j in row} for row in edges]
    for i, row in enumerate(edges):
        for _, j in row:
            reverse[j].append(i)
    nonreversible = [(i, j) for i, row in enumerate(edges) for _, j in row if i not in successors[j]]
    assert not nonreversible, nonreversible[:10]
    return {'states': states, 'index': index, 'edges': edges, 'reverse': reverse, 'ops': ops,
            'summary': {'reachable_states': len(states), 'arm_empty_states': sum('(arm-empty)' in s for s in states),
                        'holding_states': sum(any(f.startswith('(holding ') for f in s) for s in states),
                        'directed_legal_transitions': sum(map(len, edges)), 'all_edges_have_one_step_inverse': True,
                        'grounded_operators': len(ops), 'state_cap': cap}}

def distances(graph, goal):
    seeds = [i for i, s in enumerate(graph['states']) if goal <= s]
    dist = [None] * len(graph['states'])
    queue = deque(seeds)
    for i in seeds:
        dist[i] = 0
    while queue:
        j = queue.popleft()
        for i in graph['reverse'][j]:
            if dist[i] is None:
                dist[i] = dist[j] + 1
                queue.append(i)
    assert all(d is not None for d in dist)
    for i, row in enumerate(graph['edges']):
        if dist[i]:
            assert dist[i] == 1 + min(dist[j] for _, j in row)
        for _, j in row:
            assert abs(dist[i] - dist[j]) <= 1
    return dist, seeds

def forward_distance(env):
    """Independent direction of search sanity check of initial cost-to-go."""
    init = frozenset(env.task.initial_state)
    q = deque([(init, 0)])
    seen = {init}
    while q:
        s, d = q.popleft()
        if env.task.goal_reached(s):
            return d
        for op in env.task.operators:
            if op.applicable(s):
                nxt = frozenset(op.apply(s))
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, d + 1))
    raise AssertionError('No goal reachable')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(DEFAULT_REPO))
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--state-cap', type=int, default=100000)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.output.resolve()
    scratch = Path('/home/tony/.hermes/cache/scratch').resolve()
    if not out.is_relative_to(scratch):
        raise ValueError('Outputs must remain under scratch')
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo))
    from lib.pddl_env import BlocksWorldEnv
    start = time.monotonic()
    tasks_file = repo / 'results/tasks.json'
    tasks = json.loads(tasks_file.read_text())
    assert len(tasks) == 12
    sources = [repo / 'results/models/qwen-qwen3.5-9b' / b / 'llm_trajectories_action_catalog.json'
               for b in ['', 'batch-02', 'batch-03', 'batch-04']]
    inputs = [tasks_file, repo / 'pddl/domain.pddl', repo / 'lib/pddl_env.py',
              repo / 'vendor/pyperplan/pyperplan/task.py', repo / 'vendor/pyperplan/pyperplan/grounding.py',
              *[repo / t['problem_file'] for t in tasks], *sources]
    manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    graphs, envs, task_dist, task_summaries = {}, {}, {}, []
    for task in tasks:
        tid, n = task['task_id'], task['blocks']
        env = BlocksWorldEnv(repo / 'pddl/domain.pddl', repo / task['problem_file'])
        envs[tid] = env
        assert frozenset(task['initial_state']) == frozenset(env.task.initial_state)
        assert frozenset(task['goal']) == frozenset(env.task.goals)
        if n not in graphs:
            graphs[n] = build_graph(env, args.state_cap)
            g = graphs[n]
            dump(out / f'graph-{n}-blocks.json', {'blocks': n, **g['summary'],
                 'operators': op_signature(g['ops']), 'states': [sorted(s) for s in g['states']],
                 'edges_operator_index_and_target_state_id': g['edges']})
            print(f'Built {n}-block graph: {g["summary"]}', flush=True)
        g = graphs[n]
        assert op_signature(env.task.operators) == op_signature(g['ops'])
        dist, goals = distances(g, frozenset(env.task.goals))
        task_dist[tid] = dist
        initial_d = dist[g['index'][frozenset(env.task.initial_state)]]
        assert forward_distance(env) == initial_d
        ref = env.reset()
        for action in task['reference_plan']:
            result = env.step(action)
            assert result['status'] == 'success'
        assert env.solved
        assert len(task['reference_plan']) == task['reference_plan_length'] >= initial_d
        info = {'task_id': tid, 'blocks': n, 'initial_exact_distance': initial_d,
                'reference_plan_length': task['reference_plan_length'],
                'reference_plan_is_shortest': task['reference_plan_length'] == initial_d,
                'reachable_states': len(dist), 'goal_satisfying_states': len(goals),
                'goal_reachable_states': sum(d is not None for d in dist),
                'environmental_dead_end_states': sum(d is None for d in dist),
                'maximum_distance_to_goal': max(dist), 'distance_histogram': dict(Counter(dist)),
                'forward_bfs_initial_distance_verified': True, 'bellman_and_edge_lipschitz_verified': True}
        task_summaries.append(info)
        dump(out / f'distances-{tid}.json', {**info, 'graph_file': f'graph-{n}-blocks.json',
                                         'distance_by_state_id': dist, 'goal_state_ids': goals})
    annotated, trajectory_summaries = [], []
    checked_occurrences = 0
    expected_keys = {(t['task_id'], c) for t in tasks for c in ['baseline', 'one_transient_failure']}
    task_map = {t['task_id']: t for t in tasks}
    for source in sources:
        batch = source.parent.name if source.parent.name.startswith('batch-') else 'root'
        records = json.loads(source.read_text())
        assert len(records) == 24
        assert Counter((r['task_id'], r['condition']) for r in records) == Counter(expected_keys)
        batch_annotated = []
        for source_index, record in enumerate(records):
            tid, condition = record['task_id'], record['condition']
            task, env = task_map[tid], envs[tid]
            g, dist = graphs[task['blocks']], task_dist[tid]
            initial = env.reset()
            assert initial == record['initial']
            initial_id = g['index'][frozenset(initial['state'])]
            checked_occurrences += 1
            first_goal = 0 if initial['goal_reached'] else None
            seen = {initial_id}
            prefix_counts, all_counts = Counter(), Counter()
            prefix_status = Counter()
            successful_prefix_actions = []
            prefix_regret = 0
            revisits = 0
            transitions = []
            for ti, step in enumerate(record['trace'], 1):
                assert step['attempt'] == ti
                result = env.step(step['action'], transient_failure=step['status'] == 'transient_failure')
                assert result == step, (batch, tid, condition, ti)
                before, after = frozenset(step['state_before']), frozenset(step['state'])
                bi, ai = g['index'][before], g['index'][after]
                checked_occurrences += 2
                db, da = dist[bi], dist[ai]
                delta = da - db
                prefix = first_goal is None
                legal_success = step['status'] == 'success'
                direction = 'decreasing' if delta < 0 else 'increasing' if delta > 0 else 'unchanged'
                hypothetical = None
                if step['status'] == 'transient_failure':
                    op = env._operators[step['action']]
                    hypothetical = dist[g['index'][frozenset(op.apply(before))]]
                if legal_success:
                    assert any(g['ops'][oi].name == step['action'] and j == ai for oi, j in g['edges'][bi])
                    all_counts[direction] += 1
                else:
                    assert before == after and delta == 0
                revisit = legal_success and ai in seen
                if prefix:
                    prefix_status[step['status']] += 1
                    if legal_success:
                        prefix_counts[direction] += 1
                        successful_prefix_actions.append(step['action'])
                        prefix_regret += 1 + delta
                        revisits += int(revisit)
                    seen.add(ai)
                transitions.append({**step, 'offline_exact': {
                    'state_before_id': bi, 'state_after_id': ai, 'distance_before': db,
                    'distance_after': da, 'distance_delta': delta, 'distance_direction': direction,
                    'included_in_pre_first_goal_prefix': prefix, 'legal_applied_action': legal_success,
                    'successful_action_regret': 1 + delta if legal_success else None,
                    'returns_to_previously_seen_prefix_state': revisit if prefix else None,
                    'transient_failure_hypothetical_distance_after': hypothetical}})
                if first_goal is None and step['goal_reached']:
                    first_goal = ti
            assert env.snapshot() == record['final']
            assert record['goal_reached'] == env.solved
            final_id = g['index'][frozenset(record['final']['state'])]
            checked_occurrences += 1
            final_d = dist[final_id]
            prefix_end_d = 0 if first_goal is not None else final_d
            count = sum(prefix_counts.values())
            assert prefix_regret == count + prefix_end_d - dist[initial_id]
            s = {'batch': batch, 'source_index': source_index, 'task_id': tid, 'condition': condition,
                 'initial_exact_distance': dist[initial_id], 'final_exact_distance': final_d,
                 'final_goal_reached': record['goal_reached'], 'ever_reached_goal': first_goal is not None,
                 'first_goal_attempt': first_goal, 'recorded_attempts': len(transitions),
                 'max_tool_calls': record['max_tool_calls'], 'error': record.get('error'),
                 'pre_first_goal_status_counts': dict(prefix_status),
                 'pre_first_goal_legal_action_counts': {k: prefix_counts[k] for k in ['decreasing', 'increasing', 'unchanged']},
                 'all_legal_action_counts': dict(all_counts),
                 'pre_first_goal_legal_actions': count, 'pre_first_goal_successful_state_revisits': revisits,
                 'pre_first_goal_total_action_regret': prefix_regret,
                 'first_goal_legal_path_is_shortest': count == dist[initial_id] if first_goal is not None else None,
                 'first_goal_legal_sequence_matches_reference': successful_prefix_actions == task['reference_plan'] if first_goal is not None else None,
                 'final_state_id': final_id, 'minimum_distance_observed': min([dist[initial_id]] + [x['offline_exact']['distance_after'] for x in transitions])}
            trajectory_summaries.append(s)
            a = {**record, 'trace': transitions, 'offline_exact': s,
                 'initial': {**record['initial'], 'offline_exact_distance': dist[initial_id], 'offline_state_id': initial_id},
                 'final': {**record['final'], 'offline_exact_distance': final_d, 'offline_state_id': final_id}}
            batch_annotated.append(a)
            annotated.append(a)
        dump(out / f'annotated-{batch}.json', {'source_file': str(source), 'trajectories': batch_annotated})
    assert len(annotated) == 96
    unsolved = [s for s in trajectory_summaries if not s['final_goal_reached']]
    assert len(unsolved) == 8
    aggregates = []
    for condition in ['all', 'baseline', 'one_transient_failure']:
        for outcome in ['all', 'final_solved', 'final_unsolved']:
            rows = [s for s in trajectory_summaries if (condition == 'all' or s['condition'] == condition)
                    and (outcome == 'all' or s['final_goal_reached'] == (outcome == 'final_solved'))]
            counts, statuses = Counter(), Counter()
            for s in rows:
                counts.update(s['pre_first_goal_legal_action_counts'])
                statuses.update(s['pre_first_goal_status_counts'])
            legal = sum(counts.values())
            aggregates.append({'condition': condition, 'outcome': outcome, 'trajectories': len(rows),
                               'legal_action_counts': dict(counts), 'legal_actions': legal,
                               'legal_action_percentages': {k: 100 * v / legal if legal else None for k, v in counts.items()},
                               'status_counts': dict(statuses),
                               'successful_state_revisits': sum(s['pre_first_goal_successful_state_revisits'] for s in rows),
                               'total_successful_action_regret': sum(s['pre_first_goal_total_action_regret'] for s in rows)})
    solved_prefix = [s for s in trajectory_summaries if s['ever_reached_goal']]
    shortest = [s for s in solved_prefix if s['first_goal_legal_path_is_shortest']]
    summary = {'method': 'Full forward reachable graph per block count using env.task.operators, then multi-source reverse unit-cost BFS from every goal-satisfying state; replay every historical transition exactly.',
               'scope': 'Offline measurement only; original four files; no model APIs, installations, repository writes, or policy changes.',
               'prefix_definition': 'All transitions before first goal attainment, INCLUDING the goal-attaining transition. If never attained, the whole trace. Post-goal actions excluded from prefix metrics.',
               'legal_action_definition': 'Successfully applied legal PDDL actions (status=success). Transient execution failures are applicable but unapplied, counted separately; all other failed attempts are no-ops.',
               'regret_definition': 'For each successfully applied action: 1 + d(after) - d(before). Zero iff that action preserves a shortest continuation; solved prefix total equals legal path length minus exact initial distance.',
               'graphs': {str(n): g['summary'] for n, g in graphs.items()}, 'tasks': task_summaries,
               'trajectory_count': len(trajectory_summaries), 'final_solved': len(trajectory_summaries) - len(unsolved),
               'final_unsolved': len(unsolved), 'ever_reached_goal': len(solved_prefix),
               'solved_shortest_legal_paths': len(shortest),
               'solved_shortest_legal_paths_different_from_reference': sum(not s['first_goal_legal_sequence_matches_reference'] for s in shortest),
               'solved_nonshortest_legal_paths': len(solved_prefix) - len(shortest),
               'recorded_state_occurrences_verified_in_graph': checked_occurrences,
               'all_recorded_transitions_replayed_exactly': True, 'aggregates_pre_first_goal': aggregates,
               'unsolved_final_states': unsolved, 'trajectories': trajectory_summaries,
               'input_sha256': manifest, 'elapsed_seconds': time.monotonic() - start}
    dump(out / 'exact-evaluation.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ['trajectories', 'input_sha256']}, indent=2), flush=True)

if __name__ == '__main__':
    main()
