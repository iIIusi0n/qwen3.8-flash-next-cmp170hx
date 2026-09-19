# Measurement artifacts

These are measurements from the tested deployment, not output from installing
the portable scripts on another machine. Generated responses are retained.
Machine addresses, process IDs, credentials, and full service logs are omitted.

| Artifact | Meaning |
| --- | --- |
| `manifest.json` | Pinned inputs, hardware, final settings, and validation counts. |
| `bench-final-fixed3.json`, `prefill-final-fixed3.json` | Previous tuned piecewise-graph/three-token-MTP baseline. |
| `pass2/final-native-aio-bench.json` | Final 128-output-token chat sweep, three rounds, concurrency 1/4/8/16. |
| `pass2/final-native-aio-long-bench.json` | Final 512-output-token sweep, two rounds, concurrency 1/8. |
| `pass2/final-native-aio-prefill.json` | Final fresh 512/2K/8K/32K prompt sweeps, three rounds. |
| `pass2/final-native-aio-smoke.json` | Six final smoke checks; this single run passed all six. |
| `pass2/final-mixed.json` | One fresh 16K prefill plus eight chat generations. |
| `pass2/final-comparison.csv`, `final-summary.json` | Baseline/final medians and ratios. |
| `pass2/full-aio-lookahead-*.json` | Three-token MTP, native AIO and read-ahead, before larger prefill graphs. |
| `pass2/aio-pp-graphs-prefill.json` | Three-token MTP with added 1K prefill graph sizes. |
| `pass2/native-k1-b2048-*.json` | Integrated one-token-MTP/2K candidate before final clean restart. |
| `pass2/full-cold-*.json` | Cold full-graph startup with the earlier threaded I/O backend. |
| `pass2/full-uring-*.json` | Intermediate io_uring prototype; native Linux AIO is the final backend. |
| `pass2/repeated-{full,piecewise}.json` | Controlled arithmetic consistency comparison, including failures. |
| `pass2/profiles/{baseline,optimized,selected}-*` | CUDA-event/host stage timings, requests, and GPU telemetry. |
| `pass2/stage-summary.csv` | Compact wall time, exposed PLE wait, and MTP spans. |
| `tests/` | SSD, graph, INC/MTP, loader, and lint check outputs. |

Profile labels: `baseline` is the previous tuned server; `optimized` uses
three-token MTP, native AIO/read-ahead, and eager prefill above 64 tokens;
`selected` uses one-token MTP, 2K prefill chunks, and larger prefill graphs.
Nested stage spans overlap and must not be summed. Profiled timings are distinct
from the unprofiled serving benchmarks. See [interpretation and limitations](../docs/PERFORMANCE.md).
