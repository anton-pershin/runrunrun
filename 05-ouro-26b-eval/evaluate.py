#!/usr/bin/env python3
"""Evaluate a pinned model on a deterministic GSM8K subset or addition control."""
import argparse
import json
import random
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_IDS = {
    "ouro": ("ByteDance/Ouro-2.6B", "1ed04250da1a9936042725d302e81c8fa2ab5abd"),
    "qwen3": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
}
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
DATASET_TOTAL = 1319


def final_number(text: str):
    marked = re.findall(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    values = marked or re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not values:
        return None
    try:
        return float(values[-1].replace(",", ""))
    except ValueError:
        return None


def get_gsm8k_subset(size: int, seed: int):
    indices = sorted(random.Random(seed).sample(range(DATASET_TOTAL), size))
    wanted = set(indices)
    rows = {}
    for offset in range(0, DATASET_TOTAL, 100):
        params = urllib.parse.urlencode({
            "dataset": "openai/gsm8k", "config": "main", "split": "test",
            "revision": DATASET_REVISION, "offset": offset, "length": min(100, DATASET_TOTAL - offset),
        })
        with urllib.request.urlopen("https://datasets-server.huggingface.co/rows?" + params, timeout=60) as response:
            page = json.load(response)
        for item in page["rows"]:
            row = item["row"]
            idx = item["row_idx"]
            if idx in wanted:
                answer = row["answer"].split("####")[-1].strip().replace(",", "")
                rows[idx] = {"index": idx, "question": row["question"], "gold": float(answer)}
    if len(rows) != size:
        raise RuntimeError(f"Fetched {len(rows)} of {size} requested GSM8K rows")
    return [rows[i] for i in indices]


def get_addition_subset(size: int, seed: int):
    rng = random.Random(seed)
    items = []
    for i in range(size):
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        items.append({"index": i, "a": a, "b": b, "question": f"{a} + {b}", "gold": float(a + b)})
    return items


def install_ouro_cache_compat():
    """Restore list-style cache access required by this pinned Ouro code."""
    from transformers.cache_utils import Cache

    for name in ("key_cache", "value_cache"):
        descriptor = Cache.__dict__.get(name)
        if isinstance(descriptor, property) and descriptor.fset is None:
            storage_name = "_ouro_" + name
            setattr(
                Cache,
                name,
                property(
                    lambda self, key=storage_name: self.__dict__.setdefault(key, []),
                    lambda self, value, key=storage_name: self.__dict__.__setitem__(key, value),
                ),
            )
    original_get_mask_sizes = Cache.get_mask_sizes

    def get_mask_sizes_compat(self, cache_position, layer_idx):
        if type(self).__name__ == "UniversalTransformerCache":
            return self.get_seq_length(layer_idx) + cache_position.numel(), 0
        return original_get_mask_sizes(self, cache_position, layer_idx)

    Cache.get_mask_sizes = get_mask_sizes_compat


def format_prompt(tokenizer, task: str, row: dict):
    if task == "addition":
        content = f"Compute the sum. Reply with only the integer, no other text.\n{row['a']} + {row['b']} ="
    else:
        content = (
            "Solve this math word problem. Think step by step. End your response with exactly "
            "one line in this format: #### <integer>.\n\nQuestion: " + row["question"]
        )
    messages = [{"role": "user", "content": content}]
    templated = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(templated, return_tensors="pt", add_special_tokens=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODEL_IDS)
    parser.add_argument("--task", choices=("gsm8k", "addition"), default="gsm8k")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--dataset-json", type=Path, default=None,
                        help="Reuse questions/gold answers from an earlier result JSON")
    args = parser.parse_args()
    if args.task == "gsm8k" and not 1 <= args.n <= DATASET_TOTAL:
        raise ValueError(f"GSM8K --n must be from 1 to {DATASET_TOTAL}")
    if args.task == "addition" and args.n > 20:
        raise ValueError("Addition control has 20 fixed cases")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    name, revision = MODEL_IDS[args.model]
    args.out.mkdir(parents=True, exist_ok=True)
    if args.dataset_json:
        cached = json.loads(args.dataset_json.read_text())
        rows = [{k: r[k] for k in ("index", "question", "gold")} for r in cached["results"]]
        if args.n != len(rows):
            raise ValueError(f"--n={args.n} does not match cached dataset size {len(rows)}")
    else:
        rows = get_gsm8k_subset(args.n, args.seed) if args.task == "gsm8k" else get_addition_subset(args.n, args.seed)

    start_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=True)
    config = AutoConfig.from_pretrained(name, revision=revision, trust_remote_code=True)
    if args.model == "ouro":
        config.total_ut_steps = args.depth
        install_ouro_cache_compat()
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, config=config, torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda").eval()
    torch.cuda.synchronize()
    load_time = time.perf_counter() - start_load

    warmup = format_prompt(tokenizer, args.task, rows[0]).to("cuda")
    with torch.inference_mode():
        model.generate(**warmup, max_new_tokens=2, do_sample=False, use_cache=True,
                       pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    outputs = []
    for row in rows:
        encoded = format_prompt(tokenizer, args.task, row).to("cuda")
        input_ids = encoded.input_ids
        kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False, "use_cache": True,
                  "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id}
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**encoded, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        text = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
        pred = final_number(text)
        outputs.append({
            **row, "prediction": pred, "correct": pred == row["gold"],
            "raw_output": text, "input_tokens": int(input_ids.shape[1]),
            "output_tokens": int(generated.shape[1] - input_ids.shape[1]), "e2e_s": elapsed,
        })
    generated_tokens = sum(x["output_tokens"] for x in outputs)
    generation_time = sum(x["e2e_s"] for x in outputs)
    result = {
        "model": name, "revision": revision, "task": args.task,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__, "dtype": "bfloat16",
        "depth": args.depth if args.model == "ouro" else None,
        "dataset_revision": DATASET_REVISION if args.task == "gsm8k" else None,
        "n": len(outputs), "seed": args.seed, "max_new_tokens": args.max_new_tokens,
        "model_load_s": load_time,
        "accuracy": sum(x["correct"] for x in outputs) / len(outputs),
        "mean_e2e_s_per_example": generation_time / len(outputs),
        "examples_per_s": len(outputs) / generation_time,
        "output_tokens_per_s": generated_tokens / generation_time,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "results": outputs,
    }
    output_path = args.out / f"{args.model}-{args.task}-depth{args.depth if args.model == 'ouro' else 'na'}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
