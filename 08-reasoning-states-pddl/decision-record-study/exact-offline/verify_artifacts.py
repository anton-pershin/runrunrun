#!/usr/bin/env python3
"""Verify persisted graph/distance/annotation artifacts independently of evaluator."""
import sys
sys.dont_write_bytecode = True
from collections import Counter
import json
from math import comb, factorial
from pathlib import Path

out = Path(__file__).resolve().parent
summary = json.loads((out / 'exact-evaluation.json').read_text())
graphs = {}
for n in [5, 6]:
    g = json.loads((out / f'graph-{n}-blocks.json').read_text())
    states = [frozenset(s) for s in g['states']]
    assert len(states) == len(set(states)) == g['reachable_states']
    # Independent combinatorial count: partitions into unordered nonempty stacks
    # of internally ordered blocks (unsigned Lah numbers), plus holding one block.
    def empty_hand_count(m):
        return sum(factorial(m) * comb(m-1, k-1) // factorial(k) for k in range(1, m+1))
    assert g['arm_empty_states'] == empty_hand_count(n)
    assert g['holding_states'] == n * empty_hand_count(n-1)
    edges = g['edges_operator_index_and_target_state_id']
    succ = [{j for _, j in row} for row in edges]
    for i, row in enumerate(edges):
        for oi, j in row:
            name, pre, add, delete = g['operators'][oi]
            assert set(pre) <= states[i]
            assert (states[i] - set(delete)) | set(add) == states[j]
            assert i in succ[j]
        expected = {oi for oi, (_, pre, _, _) in enumerate(g['operators']) if set(pre) <= states[i]}
        assert {oi for oi, _ in row} == expected
    graphs[n] = (g, states)

dists = {}
for task in summary['tasks']:
    tid, n = task['task_id'], task['blocks']
    a = json.loads((out / f'distances-{tid}.json').read_text())
    g, states = graphs[n]
    d = a['distance_by_state_id']
    assert len(d) == len(states) and all(isinstance(x, int) and x >= 0 for x in d)
    assert {i for i, x in enumerate(d) if x == 0} == set(a['goal_state_ids'])
    for i, row in enumerate(g['edges_operator_index_and_target_state_id']):
        if d[i] > 0:
            assert d[i] == 1 + min(d[j] for _, j in row)
    dists[tid] = d

rows = []
step_total = 0
prefix_counts = Counter()
for batch in ['root', 'batch-02', 'batch-03', 'batch-04']:
    a = json.loads((out / f'annotated-{batch}.json').read_text())
    original = json.loads(Path(a['source_file']).read_text())
    assert len(a['trajectories']) == len(original) == 24
    for record, raw in zip(a['trajectories'], original, strict=True):
        s = record['offline_exact']
        rows.append(s)
        assert s in summary['trajectories']
        n = next(t['blocks'] for t in summary['tasks'] if t['task_id'] == record['task_id'])
        g, states = graphs[n]
        d = dists[record['task_id']]
        for key in raw:
            if key not in ['initial', 'final', 'trace']:
                assert record[key] == raw[key]
        for key in ['initial', 'final']:
            for k, v in raw[key].items():
                assert record[key][k] == v
            state_id = record[key]['offline_state_id']
            assert frozenset(record[key]['state']) == states[state_id]
            assert record[key]['offline_exact_distance'] == d[state_id]
        assert len(record['trace']) == len(raw['trace'])
        for step, raw_step in zip(record['trace'], raw['trace'], strict=True):
            step_total += 1
            for key in raw_step:
                assert step[key] == raw_step[key]
            x = step['offline_exact']
            assert states[x['state_before_id']] == frozenset(step['state_before'])
            assert states[x['state_after_id']] == frozenset(step['state'])
            assert d[x['state_before_id']] == x['distance_before']
            assert d[x['state_after_id']] == x['distance_after']
            if x['included_in_pre_first_goal_prefix'] and x['legal_applied_action']:
                prefix_counts[x['distance_direction']] += 1
assert len(rows) == summary['trajectory_count'] == 96
assert sum(not s['final_goal_reached'] for s in rows) == summary['final_unsolved'] == 8
aggregate = next(a for a in summary['aggregates_pre_first_goal'] if a['condition'] == a['outcome'] == 'all')
assert prefix_counts == Counter(aggregate['legal_action_counts'])
result = {'verified': True, 'persisted_trajectories_checked': len(rows), 'persisted_transitions_checked': step_total,
          'combinatorial_state_count_verified': {str(n): len(states) for n, (_, states) in graphs.items()},
          'pre_first_goal_legal_counts_verified': dict(prefix_counts),
          'checks': ['complete legal edge sets and STRIPS effects', 'one-step edge reversibility',
                     'distance nonnegativity, goal zeros and Bellman equation',
                     'all source record fields preserved', 'all recorded state IDs and exact distances',
                     'trajectory, final-failure and aggregate action counts']}
(out / 'artifact-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
