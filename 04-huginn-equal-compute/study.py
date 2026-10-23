#!/usr/bin/env python3
"""Minimal, reproducible equal-compute Huginn vs Qwen3-8B pilot."""
import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "huginn": {
        "id": "tomg-group-umd/huginn-0125",
        "revision": "bb6621b65e90b6a4b9b29ef88dc83866d450470c",
        "num_steps": 4,
    },
    "qwen3": {
        "id": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "num_steps": None,
    },
}


def tasks(n: int, seed: int):
    rng = random.Random(seed)
    pairs = [(rng.randint(100, 999), rng.randint(100, 999)) for _ in range(n)]
    return [{"a": a, "b": b, "gold": a + b} for a, b in pairs]


def prompt_for(tokenizer, model_name: str, a: int, b: int):
    text = f"Compute the sum. Reply with only the integer, no other text.\n{a} + {b} ="
    if model_name == "qwen3":
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        return encoded.input_ids
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(chat, return_tensors="pt", add_special_tokens=False).input_ids


def predicted_integer(text: str):
    values = re.findall(r"-?\d+", text)
    return int(values[-1]) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Override the recurrent depth for Huginn; ignored for Qwen3")
    args = parser.parse_args()
    spec = MODELS[args.model]
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this study")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        spec["id"], revision=spec["revision"], trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec["id"], revision=spec["revision"], torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats(device)

    output = []
    for i, case in enumerate(tasks(args.n, args.seed)):
        input_ids = prompt_for(tokenizer, args.model, case["a"], case["b"]).to(device)
        kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False,
                  "use_cache": True, "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id}
        steps = args.num_steps if args.num_steps is not None else spec["num_steps"]
        if steps is not None:
            kwargs["num_steps"] = steps
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(input_ids, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        text = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
        pred = predicted_integer(text)
        output.append({
            "index": i, "a": case["a"], "b": case["b"], "gold": case["gold"],
            "prediction": pred, "raw_output": text, "correct": pred == case["gold"],
            "input_tokens": int(input_ids.shape[1]),
            "output_tokens": int(generated.shape[1] - input_ids.shape[1]),
            "e2e_s": elapsed,
        })

    result = {
        "model": spec["id"], "revision": spec["revision"],
        "model_type": args.model, "dtype": "bfloat16", "num_steps": (args.num_steps if args.num_steps is not None else spec["num_steps"]),
        "n": args.n, "seed": args.seed, "max_new_tokens": args.max_new_tokens,
        "load_time_s": load_s,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "accuracy": sum(x["correct"] for x in output) / len(output),
        "mean_e2e_s": sum(x["e2e_s"] for x in output) / len(output),
        "results": output,
    }
    dest = args.out / f"{args.model}.json"
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
    print(f"Results: {dest}")


if __name__ == "__main__":
    main()
