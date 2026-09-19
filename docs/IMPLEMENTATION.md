# Required vLLM changes

Apply the complete [patch](../patches/qwen38-ple-ssd.patch) to upstream commit
`a5a30471ff2bb7f0824f2da10e358af98d304472`. This is a downstream patch against a
pinned source tree, not a claim that current upstream releases contain this path.
The [model checkpoint](https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP)
remains unchanged.

| File in vLLM | Purpose |
| --- | --- |
| `vllm/models/qwen4_exp/nvidia/ple_ssd.py` (new) | Header-only PLE iterator, exact-row LRU, native/threaded reads, prompt read-ahead, pinned buffers, and CUDA-stream overlap. |
| `vllm/models/qwen4_exp/nvidia/ple_ssd_io.c` (new) | Bounded native Linux AIO queue with aligned `O_DIRECT` reads and failure cleanup. |
| `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` | Select the BF16 SSD backend through `additional_config`. |
| `vllm/model_executor/model_loader/default_loader.py` | Route dedicated PLE shards through the metadata-only loader. |
| `vllm/models/qwen4_exp/nvidia/model_state.py` | Start/cancel bounded read-ahead when requests enter/leave the runner. |
| `vllm/compilation/breakable_cudagraph.py` | Allow explicit host breaks even during full graph capture. |
| `vllm/v1/worker/gpu/cudagraph_utils.py` | Capture/replay full graphs around the required SSD host breaks. |
| `vllm/model_executor/layers/quantization/inc/config_parser.py` | Preserve mixed-bit per-expert overrides for `RoutedExperts`. |
| `vllm/models/qwen4_exp/nvidia/mtp.py` | Remap checkpoint MTP layer 0 quantization entries to runtime layer 48. |

Four existing test files carry the regressions: `tests/models/qwen4_exp/test_ple.py`,
`tests/models/qwen4_exp/test_config.py`, `tests/quantization/test_auto_round.py`,
and `tests/v1/cudagraph/test_breakable_cudagraph.py`.

## Asynchronous data path

The dedicated 95.37 GiB PLE shard is opened as immutable BF16 row storage.
Metadata tensors avoid mapping the entire table into host virtual memory.
The preceding decoder layer overlaps with the selected-row read and transfer:

1. Preserve graph-produced IDs in a persistent device buffer and asynchronously
   transfer them to pinned host memory.
2. A coordinator submits native AIO reads for missing exact rows, using a
   bounded queue. The C call releases the Python GIL while waiting.
3. Read aligned pages into bounded scratch buffers and extract the original
   320-byte BF16 rows. Validate completion sizes; drain pending operations on
   errors before releasing or reusing buffers.
4. Transfer pinned output on a separate CUDA stream. CUDA events protect buffer
   reuse and order the consuming graph segment after the transfer.

The approximate 512 MiB LRU preserves exact bytes and collapses duplicate IDs.
Read-ahead reuses the model's CPU n-gram calculation, including prompt history
and EOS handling. It processes 256-token batches, gives demand reads priority,
and bounds work to the first 16,384 remaining prompt tokens by default.
Only one prompt read-ahead task runs at a time; other requests retain demand reads.

Full decode graphs retain attention and MTP capture, with two required host
breaks for SSD submission/consumption. During initial capture, ID-producing
kernels have not executed; disk reads are skipped and placeholder output is
replaced during actual replay. Full mode must be configured at startup because
attention metadata initialization depends on it.

## Configuration

The launcher supplies the following vLLM `additional_config` keys:

```json
{
  "ple_ssd_offload": true,
  "ple_ssd_native_library": "/absolute/path/to/optimization/ple_ssd_io.so",
  "ple_ssd_io_depth": 256,
  "ple_ssd_cache_mb": 512,
  "ple_ssd_workers": 16,
  "ple_ssd_prefetch_tokens": 16384
}
```

`ple_ssd_workers` controls the threaded fallback when no native library is
configured; the selected native-AIO path uses `ple_ssd_io_depth` instead.
Use `VLLM_USE_BREAKABLE_CUDAGRAPH=1` with full/piecewise graph mode. No new GPU
kernel or embedding/activation requantization is introduced by this backend.

The C helper is built separately by `scripts/build-ple-io.sh`, uses Linux ABI
headers/system calls, and has no libaio/liburing dependency. Its target-tested
binary hash was `0d5f5ff6cc02ff340c2eee9ccf7805f1b4c909d5837680e5fcc26a7844e9d8a7`;
compiler differences can change the binary hash. The portable source patch hash
is recorded in the main README.
