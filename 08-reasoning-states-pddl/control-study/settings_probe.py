"""Replay a saved public request with one controlled settings difference."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'control-study'))
from transport import public_response, append_event
from openai import AsyncOpenAI

async def main(args):
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=False)
    events=[json.loads(x) for x in Path(args.source).read_text().splitlines()]
    contexts=[e['parameters'] for e in events if e['kind']=='request']
    original=contexts[args.context]
    client=AsyncOpenAI(api_key=os.environ['HERMES_CUSTOM_CAILA_IO_API_KEY'],base_url='https://caila.io/api/openai/v1',timeout=120,max_retries=0)
    variants=[('original',{}),('sampling',{'temperature':0.7,'top_p':0.9}),('cap',{'max_tokens':8192}),('sampling_cap',{'temperature':0.7,'top_p':0.9,'max_tokens':8192})]
    try:
        for name,overrides in variants:
            params={**original,**overrides}
            append_event(out/'events.jsonl',{'kind':'request','name':name,'parameters':params})
            start=time.monotonic()
            try:
                raw=await client.chat.completions.create(**params)
                data=public_response(raw.model_dump(mode='json',exclude_none=True))
                append_event(out/'events.jsonl',{'kind':'response','name':name,'latency_s':time.monotonic()-start,'data':data})
                choice=data['choices'][0]
                print(name,'finish',choice.get('finish_reason'),'tools',len(choice['message'].get('tool_calls') or []),'content_chars',len(choice['message'].get('content') or ''),'usage',data.get('usage'),flush=True)
            except Exception as exc:
                append_event(out/'events.jsonl',{'kind':'error','name':name,'error':f'{type(exc).__name__}: {exc}'})
                print(name,type(exc).__name__,str(exc)[:200],flush=True)
    finally: await client.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--context',type=int,default=1)
    asyncio.run(main(p.parse_args()))
