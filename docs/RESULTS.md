# Final results

Measured September 19, 2026 on a 64 GiB SM80 GPU at 180 W, with 15 GiB RAM,
31 GiB swap, and enterprise SATA SSD storage. Configuration: MTP=1, 2K prefill
chunks, 16 sequences, full/piecewise graphs, AIO depth 256, 512 MiB row cache,
and 16K prompt read-ahead. Pinned inputs are in [manifest.json](../results/manifest.json).

## Generation

| Concurrent requests | Aggregate output tok/s | Per-request decode tok/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 103.30 | 110.99 | 0.105 s |
| 4 | 277.22 | 77.95 | 0.195 s |
| 8 | 428.97 | 60.63 | 0.183 s |
| 16 | 667.13 | 47.32 | 0.262 s |

Three-round medians with 128 output tokens, temperature zero, thinking disabled,
and EOS ignored. Aggregate throughput includes prefill and HTTP overhead;
decode excludes first-token latency. A separate two-round, 512-output-token
sweep measured 116.88 single-user decode tok/s and 435.28 aggregate tok/s at
concurrency eight.

## Prefill

| Prompt tokens | Median latency | Prompt tok/s |
| ---: | ---: | ---: |
| 512 | 0.317 s | 1,616 |
| 2,048 | 1.069 s | 1,915 |
| 8,192 | 3.108 s | 2,636 |
| 32,768 | 15.372 s | 2,132 |

Three rounds of fresh deterministic token-ID prompts and one output token.
Prefix-cache reuse was avoided. Native PLE reads bypass the filesystem page
cache; the application row cache remains enabled. These are measured workloads,
not guaranteed production rates. [Raw data](../results/README.md)

## Validation and limits

- 10 SSD and 21 graph tests passed; [test logs](../results/tests) include INC/MTP
  and loader checks. This is not the full upstream suite.
- 134 requests completed with zero API errors, aborts, or preemptions. Six smoke
  checks passed, including Korean, tool calls, and retrieval from 16,841 tokens.
- One fresh 16K prefill plus eight concurrent generations completed in 10.63 s.
- An arithmetic prompt can return an incorrect answer; this model/runtime
  consistency issue remains unresolved. Smoke tests are not a quality benchmark.
- The API advertises 262,144-token context; a full-length request was not tested.
- The upstream FP8 pinned-memory PLE kernel does not compile on this SM80 target;
  this deployment uses the BF16 SSD path.

See [benchmark commands](GUIDE.md#validate-and-reproduce-timings) and
[the final stage profile](PERFORMANCE.md).
