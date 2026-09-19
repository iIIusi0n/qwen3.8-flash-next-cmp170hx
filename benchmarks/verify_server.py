"""Small serving smoke evaluation; this is not a general model-quality benchmark."""

import argparse
import json
import time
from pathlib import Path

import requests

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--output", required=True)
parser.add_argument("--long", action="store_true")
parser.add_argument("--tools", action="store_true")
args = parser.parse_args()
cases = [
    ("capital", "What is the capital of France? Answer briefly.", "paris"),
    ("arithmetic", "Compute 17 times 23. Answer with just the integer.", "391"),
    ("python", "What does sum(x for x in range(11) if x % 2 == 0) return? Answer with the integer.", "30"),
    ("korean", "한국어로 '안녕하세요'라고만 답하세요.", "안녕하세요"),
]
if args.long:
    cases.append((
        "long_context",
        "Remember the access code ORCHID-7294.\n"
        + "This is background material about a library and its collection of books.\n" * 1200
        + "\nWhat access code was given at the beginning? Answer with only the code.",
        "orchid-7294",
    ))
results = []
for name, prompt, expected in cases:
    start = time.perf_counter()
    response = requests.post(args.url + "/v1/chat/completions", json={
        "model": "Qwen3.8-Flash-Next",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=1200)
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"].get("content") or ""
    result = dict(name=name, passed=expected in content.lower(), content=content,
                  latency_s=time.perf_counter() - start, usage=body.get("usage"), response=body)
    results.append(result)
    print(json.dumps({k: v for k, v in result.items() if k != "response"}, ensure_ascii=False), flush=True)
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
if args.tools:
    response = requests.post(args.url + "/v1/chat/completions", json={
        "model": "Qwen3.8-Flash-Next",
        "messages": [{"role": "user", "content": "Use get_weather to check the weather in Seoul."}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "Get current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                           "required": ["city"]},
        }}],
        "tool_choice": "auto", "temperature": 0, "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=1200)
    response.raise_for_status()
    body = response.json()
    calls = body["choices"][0]["message"].get("tool_calls") or []
    passed = any(c["function"]["name"] == "get_weather"
                 and "seoul" in c["function"]["arguments"].lower() for c in calls)
    result = dict(name="tool_call", passed=passed, response=body)
    results.append(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
if not all(r["passed"] for r in results):
    raise SystemExit("One or more serving smoke checks failed; inspect the saved responses")
