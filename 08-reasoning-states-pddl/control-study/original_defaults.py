"""Run the unchanged historical agent with original provider defaults and public telemetry."""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'decision-record-study'))
from runner import public_data, append_json

async def main(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('original_agent', ROOT/'scripts/run_llm_agent.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    os.environ['REASONING_STATES_MODEL'] = 'qwen/qwen3.5-9b'
    agent = module.make_agent()
    create = agent.model.client.chat.completions.create
    count = 0
    async def logged(**kwargs):
        nonlocal count
        count += 1
        index = count
        logged_kwargs = {k:v for k,v in kwargs.items() if type(v).__name__ not in {'Omit','NotGiven'}}
        append_json(out/'requests.jsonl', {'kind':'request','index':index,'parameters':public_data(logged_kwargs)})
        started = time.monotonic()
        try:
            response = await create(**kwargs)
            raw = response.model_dump(mode='json',exclude_none=True)
            append_json(out/'requests.jsonl',{'kind':'response','index':index,'latency_s':time.monotonic()-started,'data':public_data(raw)})
            print('response',index,'finish',raw['choices'][0].get('finish_reason'),'usage',raw.get('usage'),flush=True)
            return response
        except Exception as exc:
            append_json(out/'requests.jsonl',{'kind':'error','index':index,'error':f'{type(exc).__name__}: {exc}'})
            raise
    agent.model.client.chat.completions.create = logged
    tasks = json.loads((ROOT/'results/tasks.json').read_text())
    (out/'manifest.json').write_text(json.dumps({'tasks':args.tasks,'model':'qwen/qwen3.5-9b','model_settings':agent.model_settings,'protocol':'unchanged original runner; no explicit sampling, thinking or completion settings'},indent=2)+'\n')
    for tid in args.tasks:
        task = next(t for t in tasks if t['task_id']==tid)
        try:
            episode = await asyncio.wait_for(module.run_episode(agent,task,'baseline'),timeout=args.episode_timeout)
            (out/f'{tid}.episode.json').write_text(json.dumps(episode,indent=2)+'\n')
            print(tid,'goal',episode['goal_reached'],'actions',len(episode['trace']),'error',episode['error'],flush=True)
        except asyncio.TimeoutError:
            (out/f'{tid}.interrupted.json').write_text(json.dumps({'reason':'external episode wall-time limit','seconds':args.episode_timeout})+'\n')
            print(tid,'external wall-time limit',flush=True)
    await agent.model.client.close()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--tasks',nargs='+',default=['task-005'])
    p.add_argument('--episode-timeout',type=int,default=300)
    asyncio.run(main(p.parse_args()))
