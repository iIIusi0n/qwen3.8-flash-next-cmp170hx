"""Measure mostly uncached PLE prefill using deterministic varied token IDs."""

import argparse
import json
import random
import time
from pathlib import Path

import requests

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, default=7391)
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--lengths", type=int, nargs="+", default=[512, 2048, 8192])
args = parser.parse_args()
results = []
for length in args.lengths:
    for repeat in range(args.repeats):
        rng = random.Random(args.seed + length * 7 + repeat)
        tokens = [rng.randrange(1000, 30000) for _ in range(length)]
        start = time.perf_counter()
        response = requests.post("http://127.0.0.1:8000/v1/completions", json={
            "model": "Qwen3.8-Flash-Next", "prompt": tokens,
            "max_tokens": 1, "temperature": 0, "ignore_eos": True,
        }, timeout=1200)
        response.raise_for_status()
        elapsed = time.perf_counter() - start
        body = response.json()
        assert body["usage"]["prompt_tokens"] == length
        row = dict(label=args.label, prompt_tokens=length, repeat=repeat,
                   latency_s=elapsed, prompt_tokens_s=length / elapsed,
                   usage=body["usage"])
        results.append(row)
        print(json.dumps(row), flush=True)
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
