# Final stage profile

Selected configuration: one-token MTP, 2K prefill chunks, full/piecewise graphs,
native Linux AIO and bounded read-ahead.

| Workload | Wall time | Exposed PLE wait | Target execution, including PLE | MTP | GPU utilization |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two fresh 8K prefills | 6.223 s | 2.253 s (36%) | 6.006 s | 0.150 s | 67% |
| Two single-request, 512-token generations | 9.463 s | 0.419 s (4%) | 7.912 s | 0.729 s | 94% |
| Eight concurrent 256-token generations | 4.639 s | 0.353 s (8%) | 4.147 s | 0.231 s | 86% |

PP is limited by SSD delivery and target execution. TG is mainly target-model
GPU execution. Mean single-request GPU power was 176.5 W against the 180 W limit.

CUDA-event spans include dispatch-induced GPU idle time; nested spans overlap
and must not be added. CUPTI kernel profiling is unavailable on this GPU.
These instrumented profiles are separate from the [unprofiled serving results](RESULTS.md).
[Raw profile data](../results/pass2/profiles)
