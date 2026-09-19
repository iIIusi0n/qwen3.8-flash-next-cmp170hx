# PP and TG profiling, September 19, 2026

The measurements below come from the target SM80 GPU and enterprise SATA
storage, using the actual AutoRound checkpoint. Raw data are in [`../results/pass2`](../results/pass2).

## What was limiting performance

Baseline: piecewise graphs, three MTP tokens, 1,024-token chunks, 16 Python
I/O workers, 512 MiB row cache, and native maximum context.

| Workload | Wall time | GPU-visible PLE wait | Target execution, including PLE | MTP total, including later steps | Later MTP steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two fresh 8K prefills | 11.34 s | 5.82 s (51%) | 10.63 s | 0.30 s | 0.14 s |
| Two single-request, 512-token generations | 18.14 s | 0.44 s (2.4%) | 8.72 s | 6.20 s | 5.18 s |
| Eight simultaneous 256-token generations | 5.97 s | 0.98 s (16%) | 4.88 s | 0.71 s | 0.50 s |

PP was primarily stalled on PLE delivery. Single-request TG was dominated by
target-model dispatch/execution and the later MTP steps, which were still eager.
At eight requests, target-model work dominated and PLE I/O became more visible.
GPU utilization averaged 39%, 48%, and 72% respectively.

These are CUDA-event spans and CPU timers, with disk/CPU/GPU telemetry.
An event span includes GPU idle time caused by CPU dispatch; it is not a sum of
kernel execution times. Nested columns must not be added together. CUPTI reports
`CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED` on this GPU, so a CUDA kernel trace was not
available. Unprofiled serving runs determine the throughput results.

## Changes supported by measurements

- Full decode graphs retain only the two CPU breaks needed to submit and consume
  SSD work. Attention stays inside the graphs, and MTP uses its existing full
  graph path. Larger piecewise graphs cover prefill.
- A small standalone C helper issues Linux native AIO reads through `O_DIRECT`.
  A bounded queue reads aligned pages and returns the original BF16 row bytes.
  The blocking completion wait runs in the offload thread and releases the GIL.
  `io_destroy` drains outstanding requests on submission failures before buffers
  are released. No GPU kernels or model arithmetic are changed by this helper.
- Bounded prompt read-ahead computes the existing n-gram hashes on the CPU and
  warms exact rows while previous chunks run on CUDA. Demand reads take priority;
  cancellation stops further batches. The normal on-demand path remains active.

With three MTP tokens and 1,024-token chunks, the measured 8K prefill progression
was 5.76 s before this pass, 4.89 s with native asynchronous reads alone, 3.70 s
with native AIO plus read-ahead, and 3.38 s after extending prefill graph sizes.
That last result is approximately 2,421 prompt tokens/s, 70% above the baseline.
Each value is a two-round median. Fresh prompt seeds avoid prefix-cache reuse;
system page caches were not forcibly flushed. Native AIO bypasses the filesystem
page cache for PLE reads, while preserving the application row cache.

Before extending prefill graphs, full decode graphs plus native AIO/read-ahead
measured 135, 275, 400, and 428 aggregate output tokens/s at concurrency
1, 4, 8, and 16. The previous configuration measured 76, 205, 348, and 387.
These short 128-output-token chat sweeps include prefill and HTTP overhead.

## Profile after the main changes

This profile uses full decode graphs, three MTP tokens, native AIO, and prompt
read-ahead. Prefill still used eager execution beyond 64 tokens in this trace;
the larger prefill graph experiment was measured separately afterward.

| Workload | Wall time | GPU-visible PLE wait | Target execution, including PLE | MTP total | Later MTP steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two fresh 8K prefills | 7.33 s | 1.33 s (18%) | 6.89 s | 0.25 s | 0.05 s |
| Two single-request, 512-token generations | 8.65 s | 0.36 s (4.1%) | 6.52 s | 1.55 s | 1.00 s |
| Eight simultaneous 256-token generations | 5.10 s | 0.49 s (9.5%) | 4.38 s | 0.52 s | 0.32 s |

GPU utilization increased to 62%, 92%, and 82%. The remaining TG bottleneck is
mainly target-model execution. PP still has SSD work to hide, but its critical
I/O stall is much smaller. The profiled prefill MoE GEMM spans total 1.72 s;
the remaining target span includes other kernels and dispatch gaps.

## Selected configuration and remaining bottlenecks

The final candidate uses one-token MTP, 2,048-token prefill chunks, prefill
graph sizes through 2,048, and the integrated native AIO/read-ahead backend.
It measured 3.132 s for an 8K prefill (2,616 prompt tokens/s), versus 5.761 s
before this pass. At concurrency 1/4/8/16, the two-round candidate sweep measured
105.8/271.2/427.1/675.5 aggregate output tokens/s. Per-request single-user decode
was 114.7 tokens/s. Three-token MTP retained the best measured single-user decode
rate of 149.7 tokens/s, but its concurrency-16 aggregate result was 428.4.
The final default selects one-token MTP for aggregate throughput under load.

A separate profile of the selected configuration gives:

| Workload | Wall time | GPU-visible PLE wait | Target execution including PLE | MTP total | Mean GPU utilization |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two fresh 8K prefills | 6.223 s | 2.253 s (36.2%) | 6.006 s | 0.150 s | 67.2% |
| Two single-request, 512-token generations | 9.463 s | 0.419 s (4.4%) | 7.912 s | 0.729 s | 94.3% |
| Eight simultaneous 256-token generations | 4.639 s | 0.353 s (7.6%) | 4.147 s | 0.231 s | 86.4% |

The larger prefill chunks reduce overall execution time even though the I/O
wait fraction is higher than the earlier 1K-chunk optimized trace. PP remains
a combination of SSD delivery and target execution; native asynchronous reads
and overlap have reduced, not eliminated, the storage constraint. Single-user
TG is now mainly target execution (84% of the profiled wall time), with little
exposed PLE wait. Mean single-user GPU power was 176.5 W against the unchanged
180 W limit; the profile does not establish how much a higher power limit would
help. Batched TG is also dominated by target execution (89%).

These traces use fresh PP seeds; TG text and draft acceptance can vary. The
stage timers have overhead, and concurrent read-ahead overlaps demand reads, so
summed CPU durations and instrumented global-cache-counter deltas must not be
interpreted as serial wall time or independent cache hits. Raw files are
`../results/pass2/profiles/selected-*-{stages,requests}.json` and GPU telemetry.
Final unprofiled production results are recorded in [RESULTS.md](RESULTS.md).
The fresh process measured 2,636 prompt tokens/s at 8K (+85%), 111 single-user
decode tokens/s (+35%), and 667 aggregate output tokens/s at concurrency 16
(+72%). The 512-output-token sweep measured 117 single-user decode tokens/s;
it is a separate workload from the 128-token comparison. Final validation
completed 134 requests without API errors, aborts, preemptions, or server errors.
The mixed 16K-prefill/eight-generation check completed all nine requests.

## Correctness and limits

Ten SSD regressions pass, covering exact BF16 data, duplicates, shard/page
boundaries, graph-produced IDs, asynchronous buffer reuse, partial I/O failure,
prompt context, and cancellation. All 21 graph-primitive tests pass. Python lint,
formatting, and native compilation with warnings treated as errors pass.

The arithmetic smoke prompt asking for the sum of even numbers in `range(11)`
is inconsistent in both graph modes. A controlled 16-run comparison returned
`30` twelve times and `20` four times with full graphs; piecewise returned `30`
eleven times and `20` five times. Leading token logits were sometimes tied.
This remains a model/runtime consistency limitation; individual successful smoke
runs are not a comprehensive quality evaluation. Exact-row tests and CPU/GPU
n-gram comparisons pass. Other checks cover factual answers, Korean, a code in a
16,841-token prompt, and parsed tool calls.

Hot-switching an engine initialized for piecewise graphs into full mode is not
valid: attention metadata initialization depends on the startup mode. That
experiment was discarded. Reported full-graph serving results use a fresh start
with full graphs configured. Python-only prompt read-ahead and extra Python
I/O threads were also slower and are not selected.

Final deployment measurements and settings are recorded in [RESULTS.md](RESULTS.md).
