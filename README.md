# Qwen3.8 Flash Next on CMP 170HX

[Hugging Face model](https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP)
· [Full vLLM patch](patches/qwen38-ple-ssd.patch)
· [Setup guide](docs/GUIDE.md)

Runs the AutoRound 3 bpw + MTP checkpoint on one **64 GiB SM80 GPU**, with the
**95.37 GiB BF16 PLE table on SSD** using native asynchronous Linux AIO.
Tested with 15 GiB RAM, 31 GiB swap, enterprise SATA storage, and a 180 W GPU limit.

## Performance

| Workload | Throughput |
| --- | ---: |
| Fresh 8K prefill | **2,636 prompt tok/s** |
| Single-request decode | **111 tok/s** |
| Aggregate output, 16 requests | **667 tok/s** |

Three-round medians. Chat: 128 output tokens, temperature zero, thinking disabled,
EOS ignored. Aggregate throughput includes prefill; decode excludes first-token
latency. [Details](docs/RESULTS.md) · [Raw data](results/README.md)

## vLLM requirements

Apply the patch to **`a5a30471ff2bb7f0824f2da10e358af98d304472`** and build
`ple_ssd_io.so`. It adds async PLE reads/read-ahead, metadata-only loading,
CUDA graph host breaks, mixed-bit expert handling, and MTP layer remapping.
[Implementation](docs/IMPLEMENTATION.md)

Follow the [installation and screen guide](docs/GUIDE.md). The environment is
`~/vllm/.venv`; defaults are MTP=1, 2K prefill chunks, 16 sequences, and a
512 MiB row cache. API: `http://localhost:8000/v1`, model `Qwen3.8-Flash-Next`.

Validation: 31 SSD/graph tests passed; 134 requests completed without API errors.
An arithmetic consistency issue remains. Context is advertised as 262K;
varied-token timing was tested through 32K.

Code: [Apache-2.0](LICENSE). Model: [Qwen Community License 1.0](https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP/blob/main/LICENSE).
