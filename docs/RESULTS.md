# Measurements and validation

Measurements use the pinned model and vLLM revisions in the [README](../README.md).
The prior baseline already used threaded PLE SSD offload, three-token MTP,
piecewise graphs, 1K prefill chunks, a 512 MiB cache, and 94% GPU memory use.
The final default uses native AIO, read-ahead, full decode graphs, one-token MTP,
2K chunks, 96% GPU memory use, and larger prefill graph captures.

## Final measurements

The fresh production process measured the following three-round medians:

| Concurrent requests | Aggregate output tokens/s | Per-request decode tokens/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 103.30 | 110.99 | 0.105 s |
| 4 | 277.22 | 77.95 | 0.195 s |
| 8 | 428.97 | 60.63 | 0.183 s |
| 16 | 667.13 | 47.32 | 0.262 s |

Compared with the previous server, aggregate throughput improves by 37%, 35%,
23%, and 72% at concurrency 1, 4, 8, and 16. Single-request decode improves
from 82.50 to 110.99 tokens/s (+35%). Per-request decode at concurrency 8 and
16 is approximately unchanged; aggregate gains include reduced prefill and
scheduling time. The separate 512-output-token sweep measured 116.88 single-user
decode tokens/s and 435.28 aggregate tokens/s at concurrency eight.

| Prompt tokens | Median prefill latency | Prompt tokens/s |
| ---: | ---: | ---: |
| 512 | 0.317 s | 1,616 |
| 2,048 | 1.069 s | 1,915 |
| 8,192 | 3.108 s | 2,636 |
| 32,768 | 15.372 s | 2,132 |

The 8K prefill improves from 5.761 to 3.108 seconds: 1,422 to 2,636 prompt
tokens/s (+85%). Read-ahead stops at the configured bounded prefix, which is
one reason the longer 32K workload should not be extrapolated from the 8K rate.

All six smoke checks passed in this final run, including retrieval from a
16,841-token prompt in 4.53 seconds and a parsed tool call. The previously
observed arithmetic inconsistency remains a limitation, as detailed below.
A mixed workload with one fresh 16K prefill plus eight chat generations completed
all nine requests in 10.63 seconds.

The final audit at 2026-09-19 14:08:29 UTC recorded a healthy detached screen
service using the workdir's `.venv`, without profiling instrumentation.
Metrics showed 134 completed requests, zero API errors, zero aborts, and zero
preemptions. There were no error entries, tracebacks, or CUDA OOMs in the server
log. Total GPU usage including a separate process was 62,049 MiB of 65,536 MiB.
The API advertised 262,144-token context.

Raw final results are in [`../results/pass2`](../results/pass2), with the
original generated responses retained. See the [artifact index](../results/README.md).

The serving sweep uses eight ordinary chat prompts, temperature zero,
`enable_thinking=false`, and 128 output tokens per request with EOS ignored.
Aggregate throughput includes prefill and HTTP overhead; per-request decode
throughput excludes time to first token. Raw generated text is retained.
The final sweep uses three rounds after warmup; previous comparisons used two.
These are measured workloads, not guaranteed production rates.

The prefill sweep submits deterministic random ordinary token IDs, requests one
output token, and changes seeds between experiments. It avoids prefix-cache
reuse and measures predominantly new PLE n-grams. System caches are not forcibly
flushed; native PLE reads bypass the filesystem page cache. Application row
caching remains enabled. A separate 32K sweep and longer 512-output-token chat
sweep extend the final validation.

See the [README validation commands](../README.md#validate-and-reproduce-timings)
for reproducing the API benchmarks.

## Correctness and remaining limits

Ten SSD tests pass, covering exact BF16 rows, page/shard boundaries, duplicates,
cache behavior, asynchronous CUDA progress, graph-produced and overwritten IDs,
changed IDs across replay, partial I/O failures, prompt context, and cancellation.
All 21 graph-primitive tests pass. The unchanged INC/MTP selection (14 tests)
and loader/registry selection (7, including one overlapping SSD test) passed in
the first pass. Repository-pinned Ruff 0.14.0 checks and formatting pass across
all 12 modified Python files; C compilation with warnings as errors passes.
The saved patch passes reverse-application checks against deployed source.
These are selected regressions, not the entire upstream suite.

A Python arithmetic prompt is inconsistent in both graph modes: in a controlled
16-run comparison, full graphs returned the correct `30` 12 times and incorrect
`20` four times; piecewise returned `30` 11 times and `20` five times. Leading
logits sometimes tie. This model/runtime consistency issue remains unresolved;
it is not evidence of a new full-graph regression. Exact-row and CPU/GPU hash
comparisons pass. Individual successful smoke runs do not establish general
model quality. Other checks cover factual answers, Korean, retrieval from a
16,841-token prompt, and parsed tool calls.

The broader upstream PLE tests include an unchanged FP8 pinned-memory kernel
that does not compile on this SM80 GPU. This deployment uses the BF16 SSD path.
The API advertises the native 262,144-token context, but a full-length request
has not been evaluated. The largest varied-token timing sweep is 32,768 tokens.
CUDA-event spans and host telemetry provide the stage profile; CUPTI cannot
profile kernels on this GPU. Details and raw-result references are in
[PERFORMANCE.md](PERFORMANCE.md) and [raw results](../results/pass2).

## Regression commands

After provisioning the matching upstream test dependencies and patched runtime:

```bash
cd "$VLLM_WORKDIR/src"
VLLM_PLE_SSD_NATIVE_LIBRARY="$VLLM_WORKDIR/optimization/ple_ssd_io.so" \
VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
"$VLLM_WORKDIR/.venv/bin/python" -m pytest tests/models/qwen4_exp/test_ple.py -k ssd -q
VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
"$VLLM_WORKDIR/.venv/bin/python" -m pytest tests/v1/cudagraph/test_breakable_cudagraph.py -q
```

The first selection passed 10 tests and the second 21. Run CUDA tests with
sufficient free GPU memory, separate from a serving instance. Saved
[test logs](../results/tests) record the measured deployment's checks. The public
wrappers also pass shell syntax checks; their generated default serving command
was compared with the measured process, and the complete patch was independently
applied to a pristine baseline before publication.
