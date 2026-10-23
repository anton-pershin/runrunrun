#!/usr/bin/env python3
"""Run Ouro at a chosen depth while reusing prior GSM8K questions."""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from evaluate import install_ouro_cache_compat, final_number

MODEL = "ByteDance/Ouro-2.6B"
REVISION = "1ed04250da1a9936042725d302e81c8fa2ab5abd"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/ouro-gsm8k-depth4.json"))
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("results/ouro-gsm8k-depth8.json"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    source = json.loads(args.input.read_text())
    cases = [{k: row[k] for k in ("index", "question", "gold")} for row in source["results"]]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    config.total_ut_steps = args.depth
    install_ouro_cache_compat()
    start_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, config=config, torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda").eval()
    torch.cuda.synchronize()
    load_time = time.perf_counter() - start_load

    def encode(question):
        text = (
            "Solve this math word problem. Think step by step. End your response with exactly "
            "one line in this format: #### <integer>.\n\nQuestion: " + question
        )
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        return tokenizer(chat, return_tensors="pt", add_special_tokens=False)

    warmup = encode(cases[0]["question"]).to("cuda")
    with torch.inference_mode():
        model.generate(**warmup, max_new_tokens=2, do_sample=False, use_cache=True,
                       pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    records = []
    for case in cases:
        encoded = encode(case["question"]).to("cuda")
        input_ids = encoded.input_ids
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                use_cache=True, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        text = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
        pred = final_number(text)
        records.append({
            **case, "prediction": pred, "correct": pred == case["gold"],
            "raw_output": text, "input_tokens": int(input_ids.shape[1]),
            "output_tokens": int(generated.shape[1] - input_ids.shape[1]), "e2e_s": elapsed,
        })
    total_t = sum(x["e2e_s"] for x in records)
    total_tokens = sum(x["output_tokens"] for x in records)
    result = {
        "model": MODEL, "revision": REVISION, "task": "gsm8k",
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__, "dtype": "bfloat16", "depth": args.depth,
        "dataset_revision": source["dataset_revision"], "n": len(records),
        "seed": source["seed"], "max_new_tokens": args.max_new_tokens,
        "model_load_s": load_time,
        "accuracy": sum(x["correct"] for x in records) / len(records),
        "mean_e2e_s_per_example": total_t / len(records),
        "examples_per_s": len(records) / total_t,
        "output_tokens_per_s": total_tokens / total_t,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "results": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
