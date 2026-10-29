"""Bounded prefix-only voting over public proposals, followed by single sampling."""
from collections import Counter
from controller import run_controller


async def run_prefix_vote(initial, propose, execute, *, vote_steps=5, max_actions=50):
    if not isinstance(vote_steps,int) or vote_steps < 0:
        raise ValueError('vote_steps must be a nonnegative integer')
    decisions=0
    votes=[]
    async def choose(**context):
        nonlocal decisions
        decisions += 1
        if decisions > vote_steps:
            return await propose(**context)
        record={'decision':decisions,'candidates':[],'errors':[]}
        votes.append(record)
        for _ in range(3):
            try:
                plan=await propose(**context)
                if not isinstance(plan,list) or not 1<=len(plan)<=50 or not all(
                        isinstance(a,str) and a.strip() for a in plan):
                    raise ValueError('invalid sampled plan')
                record['candidates'].append(list(plan))
            except Exception as exc:
                record['errors'].append(f'{type(exc).__name__}: {exc}')
        if not record['candidates']:
            raise RuntimeError('no usable prefix proposal')
        counts=Counter(p[0] for p in record['candidates'])
        # Stable sample-order tie break: all-distinct votes use the first usable sample.
        winner=max(counts,key=lambda action: counts[action])
        plan=next(p for p in record['candidates'] if p[0]==winner)
        record.update(selected=list(plan),support=counts[winner],majority=counts[winner]>=2)
        return list(plan)
    result=await run_controller(initial,choose,execute,mode='recede',max_actions=max_actions,
                                max_replans=max_actions+1,transient_retries=1)
    result.update(votes=votes,vote_steps=vote_steps,decision_count=decisions)
    return result
