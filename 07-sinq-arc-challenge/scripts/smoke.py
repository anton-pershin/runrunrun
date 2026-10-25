#!/usr/bin/env python3
import json
import os
import sys
import time
import urllib.error
import urllib.request

base = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
model = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B")
prompt = r"Solve the problem. Return your final answer as a single integer inside \\boxed{}. Problem: Find the sum of all integer bases b>9 for which 17_b is a divisor of 97_b."
url = base.rstrip("/") + "/v1/chat/completions"
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 512,
    "seed": 42,
    "chat_template_kwargs": {"enable_thinking": True},
}
req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
for i in range(90):
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/health", timeout=5) as response:
            if response.status == 200:
                break
    except Exception:
        time.sleep(5)
else:
    raise SystemExit("vLLM health check timed out")
try:
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.load(response)
except urllib.error.HTTPError as error:
    print(error.read().decode(errors="replace"), file=sys.stderr)
    raise
print(json.dumps(result, indent=2))
