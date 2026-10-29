"""Matched sampled-plan experiment. PDDL is used only by execution and replay."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
from pddl_env import BlocksWorldEnv, _ACTION_RE
from transport import PlanProposer


def eligible(env, action, condition, injected):
    if condition != 'one_transient_failure' or injected or env.attempts < 2:
        return False
    match = _ACTION_RE.fullmatch(action.strip())
    normalized = '(' + ' '.join(match.group(1).split()) + ')' if match else ''
    op = env._operators.get(normalized)
    return op is not None and op.applicable(env.state)


def make_executor(task, condition):
    env = BlocksWorldEnv(ROOT / 'pddl/domain.pddl', ROOT / task['problem_file'])
    injected = False
    async def execute(action):
        nonlocal injected
        inject = eligible(env, action, condition, injected)
        event = env.step(action, transient_failure=inject)
        injected |= inject
        return event
    return env, execute


def verify_episode(episode):
    task = next(t for t in json.loads((ROOT / 'results/tasks.json').read_text()) if t['task_id'] == episode['task_id'])
    env = BlocksWorldEnv(ROOT / 'pddl/domain.pddl', ROOT / task['problem_file'])
    errors = []
    if env.snapshot() != episode['initial']:
        errors.append('initial mismatch')
    injected = False
    for index, event in enumerate(episode['trace']):
        inject = eligible(env, event['action'], episode['condition'], injected)
        actual = env.step(event['action'], transient_failure=inject)
        injected |= inject
        if actual != event:
            errors.append(f'transition mismatch at {index + 1}')
        if actual['goal_reached'] and index + 1 != len(episode['trace']):
            errors.append('post-goal action')
    if env.snapshot() != episode['final'] or env.solved != episode['goal_reached']:
        errors.append('final mismatch')
    if len(episode['trace']) > episode['max_actions']:
        errors.append('action budget exceeded')
    return {'valid': not errors, 'errors': errors}


def make_jobs(tasks, conditions, repeats, modes=('recede', 'commit')):
    jobs = [dict(task_id=t, condition=c, repeat=r, mode=m)
            for t, c, r, m in itertools.product(tasks, conditions, range(repeats), modes)]
    random.Random(20261028).shuffle(jobs)
    return jobs


def metrics(episode):
    visited = {tuple(episode['initial']['state'])}
    rejected = set()
    revisits = repeats = 0
    for event in episode['trace']:
        if event['status'] == 'success':
            state = tuple(event['state'])
            revisits += state in visited
            visited.add(state)
        elif event['status'] != 'transient_failure':
            key = (tuple(event['state_before']), event['action'], event['status'])
            repeats += key in rejected
            rejected.add(key)
    return {'actions': len(episode['trace']), 'statuses': dict(Counter(e['status'] for e in episode['trace'])),
            'state_revisits': revisits, 'repeated_rejections': repeats}


async def run_policy(initial, propose, execute, *, mode, max_actions, review=None, vote_steps=5):
    if mode == 'prefix_vote':
        from prefix_vote import run_prefix_vote
        return await run_prefix_vote(initial,propose,execute,vote_steps=vote_steps,max_actions=max_actions)
    if mode == 'review':
        from reviewed_controller import run_reviewed_controller
        if review is None:
            raise ValueError('review proposer required')
        return await run_reviewed_controller(initial, propose, review, execute,
            max_actions=max_actions, max_replans=(max_actions + 1) // 2)
    if mode == 'step_vote':
        from consensus_controller import run_consensus_controller
        return await run_consensus_controller(initial, propose, execute,
            samples=3,min_support=2,max_votes=max_actions+1,
            max_actions=max_actions,execution_mode='recede')
    if mode == 'consensus':
        from consensus_controller import run_consensus_controller
        return await run_consensus_controller(initial, propose, execute,
            samples=3, min_support=2, max_votes=(max_actions + 1) // 3,
            max_actions=max_actions, transient_retries=1)
    from controller import run_controller
    return await run_controller(initial, propose, execute, mode=mode,
        max_actions=max_actions, max_replans=max_actions + 1, transient_retries=1)


async def main(args):
    from openai import AsyncOpenAI
    from controller import run_controller
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    tasks = {t['task_id']: t for t in json.loads((ROOT / 'results/tasks.json').read_text())}
    if set(args.tasks) - set(tasks):
        raise ValueError('unknown task')
    jobs = make_jobs(args.tasks, args.conditions, args.repeats, args.modes)
    sources = list(Path(__file__).parent.glob('*.py')) + list(Path(__file__).parent.glob('*PROTOCOL.md')) + [
               ROOT / 'lib/pddl_env.py', ROOT / 'pddl/domain.pddl', ROOT / 'results/tasks.json']
    config = {'created_utc': datetime.now(timezone.utc).isoformat(), 'arguments': vars(args),
              'jobs': jobs, 'temperature': 0.7, 'top_p': 0.9, 'seed': None,
              'python': sys.version, 'dependencies': {p: importlib.metadata.version(p) for p in ['openai', 'pydantic']},
              'git_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (out/'manifest.json').write_text(json.dumps(config,indent=2)+'\n')
    snapshots = out/'sources'
    for p in sources:
        dst = snapshots/p.relative_to(ROOT)
        dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_bytes(p.read_bytes())
    client = AsyncOpenAI(api_key=os.environ['HERMES_CUSTOM_CAILA_IO_API_KEY'],
                         base_url='https://caila.io/api/openai/v1',timeout=args.timeout,max_retries=0)
    sem = asyncio.Semaphore(args.concurrency)
    async def trial(job):
        async with sem:
            name = f'{job["task_id"]}__{job["condition"]}__{job["mode"]}__r{job["repeat"]:02d}'
            env, execute = make_executor(tasks[job['task_id']],job['condition'])
            initial = env.snapshot()
            proposer = PlanProposer(client,out/f'{name}.requests.jsonl',thinking=args.thinking,
                                     max_tokens=args.max_tokens,max_requests=((args.max_actions + 1)*(3 if job['mode']=='step_vote' else 1)
                                                   + (2*min(args.vote_steps,args.max_actions) if job['mode']=='prefix_vote' else 0)),
                                     output_format=args.output_format)
            reviewer = PlanProposer(client,out/f'{name}.review.requests.jsonl',thinking=args.thinking,
                                    max_tokens=args.max_tokens,max_requests=(args.max_actions + 1)//2,
                                    output_format='review') if job['mode']=='review' else None
            start = time.monotonic()
            result = await run_policy(initial,proposer,execute,mode=job['mode'],
                                      max_actions=args.max_actions,review=reviewer,vote_steps=args.vote_steps)
            count = proposer.requests + (reviewer.requests if reviewer else 0)
            result.update(job,initial=initial,final=env.snapshot(),max_actions=args.max_actions,
                          model='qwen/qwen3.5-9b',elapsed_s=time.monotonic()-start,request_count=count)
            result['verification'] = verify_episode(result)
            result['metrics'] = metrics(result)
            (out/f'{name}.episode.json').write_text(json.dumps(result,indent=2)+'\n')
            print(f'{name} goal={result["goal_reached"]} actions={len(result["trace"])} '
                  f'calls={count} termination={result["termination"]} '
                  f'verified={result["verification"]["valid"]}',flush=True)
            return result
    try:
        episodes = await asyncio.gather(*(trial(j) for j in jobs))
        (out/'episodes.json').write_text(json.dumps(episodes,indent=2)+'\n')
        summary = {'planned':len(jobs),'completed':len(episodes),
                   'all_replays_valid':all(e['verification']['valid'] for e in episodes),
                   'arms':{m:{'episodes':sum(e['mode']==m for e in episodes),
                              'goals':sum(e['mode']==m and e['goal_reached'] for e in episodes),
                              'terminations':dict(Counter(e['termination'] for e in episodes if e['mode']==m))}
                           for m in args.modes}}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2),flush=True)
    finally:
        await client.close()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--tasks',nargs='+',default=['task-005','task-011'])
    parser.add_argument('--conditions',nargs='+',choices=['baseline','one_transient_failure'],default=['baseline'])
    parser.add_argument('--modes',nargs='+',choices=['recede','commit','consensus','review','step_vote','prefix_vote'],default=['recede','commit'])
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--concurrency',type=int,default=4)
    parser.add_argument('--vote-steps',type=int,default=5)
    parser.add_argument('--max-actions',type=int,default=50)
    parser.add_argument('--max-tokens',type=int,default=8192)
    parser.add_argument('--thinking',choices=['default','on','off','off-native'],default='default')
    parser.add_argument('--output-format',choices=['json','tool'],default='json')
    parser.add_argument('--timeout',type=int,default=180)
    args = parser.parse_args()
    if args.vote_steps < 0:
        parser.error('vote-steps must be nonnegative')
    if min(args.repeats,args.concurrency,args.max_actions,args.max_tokens,args.timeout)<1:
        parser.error('budgets must be positive')
    return args


if __name__ == '__main__':
    asyncio.run(main(arguments()))
