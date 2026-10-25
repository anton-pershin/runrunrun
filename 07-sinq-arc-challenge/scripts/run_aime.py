#!/usr/bin/env python3
"""Reproducible greedy AIME24+AIME25 evaluation for vLLM or HF SINQ."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

DATASETS = {
    "aime24": ("Math-AI/aime24", "83a7f387baaa524a8bda0022eac0541582297103"),
    "aime25": ("Math-AI/aime25", "563bb8404243c5f09de6ec262f2db674fe5bce9b"),
}
PROMPT_PREFIX = (
    "Solve this AIME problem carefully, but keep your reasoning concise. "
    "Give the final integer answer (0 to 999) on the last line as: ANSWER: <integer>."
)
MAX_NEW_TOKENS = 16000
THINKING_ENABLED = True


def get_answer(row: dict, dataset: str) -> str:
    if dataset == "aime25":
        return str(row["answer"]).strip()
    matches = re.findall(r"\\boxed\s*\{([^{}]+)\}", row["solution"])
    if not matches:
        raise ValueError(f"Cannot extract AIME24 gold answer for id={row['id']}")
    return matches[-1].strip()


def get_prediction(text: str) -> str | None:
    plain = re.fullmatch(r"\s*(\d{1,3})\s*", text)
    if plain:
        return plain.group(1)
    sums = re.findall(
        r"(?i)\bsum\s*=\s*\$?(\d{1,3})\s*\+\s*\$?(\d{1,3})\s*=\s*\$?(\d{1,3})",
        text,
    )
    if sums:
        return sums[-1][-1]
    summaries = re.findall(
        r"(?i)\b(?:the\s+)?sum\s+(?:is|equals)\s*(?:\$?\d{1,3}\s*\+\s*\$?\d{1,3}\s*=\s*\$?)?(\d{1,3})\b",
        text,
    )
    if summaries:
        return summaries[-1]
    answers = re.findall(r"(?i)\bANSWER\s*[:=]\s*(\d{1,3})\b", text)
    if answers:
        return answers[-1]
    boxes = re.findall(r"\\boxed\s*\{\s*(\d{1,3})\s*\}", text)
    if boxes:
        return boxes[-1]
    for pattern in (
        r"(?im)^\s*sum\s*=\s*\$?\s*(\d{1,3})\s*$",
        r"(?im)^\s*(?:therefore,?\s*)?(?:the\s+)?answer\s+is\s*\$?\s*(\d{1,3})\s*[.!]?\s*$",
        r"(?i)\b(?:final\s+)?answer\s*(?:is|[:=])\s*\$?\s*(\d{1,3})\b",
    ):
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
    return None


def call_vllm(base_url: str, model: str, problem: str) -> tuple[str, str, int | None]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT_PREFIX},
            {"role": "user", "content": problem},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": MAX_NEW_TOKENS,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": THINKING_ENABLED},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as response:
        result = json.load(response)
    choice = result["choices"][0]
    return choice["message"].get("content") or "", choice.get("finish_reason", ""), result.get("usage", {}).get("completion_tokens")


def load_sinq_model(model_id: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, SinqConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a", trust_remote_code=False)
    from huggingface_hub import hf_hub_download

    awq_config_path = hf_hub_download(
        "cyankiwi/Qwen3.5-9B-AWQ-4bit",
        "config.json",
        revision="e4d2d39cede8c5d1716d824cd843e2889c648083",
    )
    quantization = json.loads(Path(awq_config_path).read_text())["quantization_config"]
    quant_group = quantization["config_groups"]["group_0"]
    skip_modules = quantization["ignore"]
    if quant_group["weights"]["num_bits"] != 4 or quant_group["weights"]["group_size"] != 32:
        raise ValueError("Approved AWQ checkpoint changed quantization scheme")
    torch.manual_seed(42)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        quantization_config=SinqConfig(
            nbits=4, group_size=32, tiling_mode="1D",
            modules_to_not_convert=skip_modules,
        ),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=False,
    )
    model.eval()
    return tokenizer, model


def call_sinq(tokenizer, model, problem: str) -> tuple[str, str, int]:
    import torch

    messages = [
        {"role": "system", "content": PROMPT_PREFIX},
        {"role": "user", "content": problem},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=THINKING_ENABLED,
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, input_len:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    finish = "length" if len(new_tokens) >= MAX_NEW_TOKENS else "stop"
    return completion, finish, int(len(new_tokens))


def main() -> None:
    global MAX_NEW_TOKENS, THINKING_ENABLED
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["vllm", "sinq"], required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--limit", type=int, default=None, help="limit rows per dataset for pilot testing")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--thinking", choices=["on", "off"], default="on")
    args = parser.parse_args()
    MAX_NEW_TOKENS = args.max_new_tokens
    THINKING_ENABLED = args.thinking == "on"

    from datasets import load_dataset

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"{args.label}.jsonl"
    completed: set[tuple[str, str]] = set()
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            record = json.loads(line)
            completed.add((record["dataset"], str(record["id"])))

    tokenizer = model = None
    if args.backend == "sinq":
        print(f"Loading and SINQ-quantizing {args.model}", flush=True)
        tokenizer, model = load_sinq_model(args.model)
        print("Model loaded", flush=True)

    with records_path.open("a", encoding="utf-8") as out:
        for dataset_key in args.datasets:
            repo, revision = DATASETS[dataset_key]
            data = load_dataset(repo, split="test", revision=revision)
            if args.limit is not None:
                data = data.select(range(min(args.limit, len(data))))
            print(f"{dataset_key}: {len(data)} items at {revision}", flush=True)
            for index, row in enumerate(data):
                item_id = str(row["id"])
                if (dataset_key, item_id) in completed:
                    continue
                gold = get_answer(row, dataset_key)
                start = time.perf_counter()
                if args.backend == "vllm":
                    text, finish, generated = call_vllm(args.base_url, args.model, row["problem"])
                else:
                    text, finish, generated = call_sinq(tokenizer, model, row["problem"])
                elapsed = time.perf_counter() - start
                prediction = get_prediction(text)
                record = {
                    "label": args.label,
                    "backend": args.backend,
                    "model": args.model,
                    "dataset": dataset_key,
                    "dataset_revision": revision,
                    "id": item_id,
                    "gold": gold,
                    "prediction": prediction,
                    "correct": prediction == gold,
                    "finish_reason": finish,
                    "completion_tokens": generated,
                    "elapsed_s": round(elapsed, 3),
                    "response": text,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                completed.add((dataset_key, item_id))
                status = "correct" if record["correct"] else "wrong"
                print(
                    f"{args.label} {dataset_key} {index + 1}/{len(data)} id={item_id} "
                    f"pred={prediction} gold={gold} {status} finish={finish} "
                    f"tokens={generated} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    print(f"Wrote {records_path}", flush=True)


if __name__ == "__main__":
    main()
