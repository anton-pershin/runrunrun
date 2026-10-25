#!/usr/bin/env python3
"""MMLU-Redux evaluation for Qwen3.5-9B BF16, AWQ, or SINQ."""
from __future__ import annotations
import argparse, json, re, time, urllib.request
from pathlib import Path
from datasets import get_dataset_config_names, load_dataset

DATASET = "edinburgh-dawg/mmlu-redux"
REVISION = "3720db6aeb3d019de48bf37916c1a54074ff4997"
CONFIGS = get_dataset_config_names(DATASET)
PROMPT = "Answer the multiple-choice question. Output only the letter of the correct option (A, B, C, or D). Do not explain."
MAX_NEW_TOKENS = 16
THINKING = False


def format_question(row):
    return "\n".join([row["question"]] + [f"{chr(65+i)}. {x}" for i, x in enumerate(row["choices"])])


def gold_answer(row):
    # Redux supplies corrected_answer for manually corrected records. For clean
    # records, answer is the original zero-based option index.
    value = row.get("correct_answer")
    if value is None:
        value = row["answer"]
    if isinstance(value, int):
        return chr(65 + value) if 0 <= value < 4 else None
    text = str(value).strip()
    if re.fullmatch(r"[A-Da-d]", text):
        return text.upper()
    if re.fullmatch(r"[0-3]", text):
        return chr(65 + int(text))
    return None


def prediction(text):
    m = re.fullmatch(r"\s*([A-Da-d])\s*[.!]?\s*", text)
    return m.group(1).upper() if m else None


def call_vllm(base_url, model, prompt):
    payload = {"model": model, "messages": [{"role":"system","content":PROMPT},{"role":"user","content":prompt}], "temperature": 0, "top_p": 1, "max_tokens": MAX_NEW_TOKENS, "seed": 42, "chat_template_kwargs": {"enable_thinking": THINKING}}
    req = urllib.request.Request(base_url.rstrip("/")+"/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r: result = json.load(r)
    c=result["choices"][0]
    return c["message"].get("content") or "", c.get("finish_reason", ""), result.get("usage",{}).get("completion_tokens")


def load_sinq(model_id):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, SinqConfig
    from huggingface_hub import hf_hub_download
    tokenizer=AutoTokenizer.from_pretrained(model_id, revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    p=hf_hub_download("cyankiwi/Qwen3.5-9B-AWQ-4bit","config.json",revision="e4d2d39cede8c5d1716d824cd843e2889c648083")
    q=json.loads(Path(p).read_text())["quantization_config"]
    group=q["config_groups"]["group_0"]
    if group["weights"]["num_bits"] != 4 or group["weights"]["group_size"] != 32: raise ValueError("AWQ recipe changed")
    model=AutoModelForImageTextToText.from_pretrained(model_id, revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a", quantization_config=SinqConfig(nbits=4,group_size=32,tiling_mode="1D",modules_to_not_convert=q["ignore"]), dtype=torch.bfloat16, device_map="cuda:0")
    model.eval(); return tokenizer, model


def call_sinq(tokenizer, model, prompt):
    import torch
    messages=[{"role":"system","content":PROMPT},{"role":"user","content":prompt}]
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=THINKING)
    inp=tokenizer(text,return_tensors="pt").to("cuda:0"); n=inp["input_ids"].shape[1]
    with torch.inference_mode(): out=model.generate(**inp,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,use_cache=True,pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id)
    toks=out[0,n:]; return tokenizer.decode(toks,skip_special_tokens=True), ("length" if len(toks)>=MAX_NEW_TOKENS else "stop"), int(len(toks))


def main():
    global MAX_NEW_TOKENS, THINKING
    ap=argparse.ArgumentParser(); ap.add_argument("--backend",choices=["vllm","sinq"],required=True); ap.add_argument("--label",required=True); ap.add_argument("--model",required=True); ap.add_argument("--base-url",default="http://127.0.0.1:8000"); ap.add_argument("--output-dir",default="results"); ap.add_argument("--limit",type=int); ap.add_argument("--per-subject",type=int); ap.add_argument("--max-new-tokens",type=int,default=MAX_NEW_TOKENS); ap.add_argument("--thinking",choices=["on","off"],default="off")
    a=ap.parse_args(); MAX_NEW_TOKENS=a.max_new_tokens; THINKING=a.thinking=="on"; outdir=Path(a.output_dir); outdir.mkdir(exist_ok=True,parents=True); path=outdir/(a.label+".jsonl")
    done=set()
    if path.exists():
        for l in path.read_text().splitlines():
            r=json.loads(l); done.add((r["subject"],str(r["id"])))
    tok=model=None
    if a.backend=="sinq": print("Loading and SINQ-quantizing",flush=True); tok,model=load_sinq(a.model); print("Model loaded",flush=True)
    rows=[]
    for subject in CONFIGS:
        d=load_dataset(DATASET,subject,split="test",revision=REVISION)
        selected=0
        for i, r in enumerate(d):
            if gold_answer(r) is None:
                continue
            rows.append((subject,i,r))
            selected += 1
            if a.per_subject is not None and selected >= a.per_subject:
                break
    if a.limit is not None: rows=rows[:a.limit]
    print(f"MMLU-Redux {len(rows)} rows across {len(CONFIGS)} subjects at {REVISION}",flush=True)
    with path.open("a") as f:
        for ix,(subject,i,row) in enumerate(rows,1):
            key=(subject,str(i))
            if key in done: continue
            gold=gold_answer(row); start=time.perf_counter(); prompt=format_question(row)
            text,finish,toks=call_vllm(a.base_url,a.model,prompt) if a.backend=="vllm" else call_sinq(tok,model,prompt)
            pred=prediction(text); elapsed=time.perf_counter()-start
            rec={"label":a.label,"backend":a.backend,"model":a.model,"dataset":DATASET,"revision":REVISION,"subject":subject,"id":str(i),"gold":gold,"prediction":pred,"valid_gold":gold is not None,"correct":gold is not None and pred==gold,"error_type":row["error_type"],"finish_reason":finish,"completion_tokens":toks,"elapsed_s":round(elapsed,3),"response":text}
            f.write(json.dumps(rec,ensure_ascii=False)+"\n"); f.flush(); print(f"{a.label} {ix}/{len(rows)} {subject}/{i} pred={pred} gold={gold} correct={rec['correct']} finish={finish} tokens={toks} elapsed={elapsed:.2f}s",flush=True)
    print("Wrote",path,flush=True)
if __name__=="__main__": main()
