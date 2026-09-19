"""Reproducible streaming latency/throughput sweep for the local vLLM API."""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--model", default="Qwen3.8-Flash-Next")
parser.add_argument("--label", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8])
parser.add_argument("--tokens", type=int, default=128)
parser.add_argument("--rounds", type=int, default=2)
args = parser.parse_args()

PROMPTS = [
    "Explain how a database index speeds up lookups. Give a concrete example.",
    "Write a Python function that merges two sorted lists and explain its complexity.",
    "Explain why the sky looks blue in language a teenager can understand.",
    "Describe the tradeoffs between asynchronous and synchronous disk I/O.",
    "Write a short story about a researcher who discovers a new island.",
    "Explain how a hash table handles collisions, with two different strategies.",
    "Describe how to design a reliable background job queue.",
    "Compare TCP and UDP for a real-time multiplayer game.",
]


async def request(session, prompt):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": args.tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    start = time.perf_counter()
    first = None
    pieces = []
    usage = None
    async with session.post(args.url + "/v1/chat/completions", json=payload) as response:
        if response.status != 200:
            raise RuntimeError(await response.text())
        async for line in response.content:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content") or delta.get("reasoning") or ""
                if content:
                    first = first or time.perf_counter()
                    pieces.append(content)
    end = time.perf_counter()
    if not usage or first is None:
        raise RuntimeError("Missing streamed output or token usage")
    count = usage["completion_tokens"]
    return {
        "ttft_s": first - start,
        "latency_s": end - start,
        "output_tokens": count,
        "decode_tokens_s": (count - 1) / max(end - first, 1e-6),
        "output": "".join(pieces),
    }


async def main():
    results = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as session:
        await request(session, "Briefly explain what a compiler does.")
        for concurrency in args.concurrency:
            for repeat in range(args.rounds):
                start = time.perf_counter()
                responses = await asyncio.gather(*[
                    request(session, PROMPTS[(i + repeat) % len(PROMPTS)])
                    for i in range(concurrency)
                ])
                elapsed = time.perf_counter() - start
                item = {
                    "label": args.label,
                    "concurrency": concurrency,
                    "repeat": repeat,
                    "elapsed_s": elapsed,
                    "aggregate_tokens_s": sum(r["output_tokens"] for r in responses) / elapsed,
                    "median_decode_tokens_s": statistics.median(r["decode_tokens_s"] for r in responses),
                    "median_ttft_s": statistics.median(r["ttft_s"] for r in responses),
                    "responses": responses,
                }
                results.append(item)
                print(json.dumps({k: v for k, v in item.items() if k != "responses"}), flush=True)
                Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


asyncio.run(main())
