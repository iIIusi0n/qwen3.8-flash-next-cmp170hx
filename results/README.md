# Final measurement artifacts

- `manifest.json`: hardware, pinned inputs, settings, and validation counts.
- `pass2/final-native-aio-bench.json`: three-round 128-token chat sweep.
- `pass2/final-native-aio-long-bench.json`: two-round 512-token chat sweep.
- `pass2/final-native-aio-prefill.json`: fresh 512/2K/8K/32K prefill sweeps.
- `pass2/final-native-aio-smoke.json`: six serving smoke checks.
- `pass2/final-mixed.json`: one 16K prefill plus eight generations.
- `pass2/profiles/selected-*`: final stage timings, requests, and GPU telemetry.
- `tests/`: SSD, graph, INC/MTP, loader, and lint logs.
- `publication-checks.json`: patch application, compilation, and launcher checks.

[Results and limitations](../docs/RESULTS.md) · [Stage profile](../docs/PERFORMANCE.md)
