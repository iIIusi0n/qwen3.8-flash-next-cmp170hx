# Installation and operation

[Model](https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP) · [Patch](../patches/qwen38-ple-ssd.patch)

## Setup

Use Linux x86-64, a CUDA-compatible driver, a C compiler, Git, GNU screen,
[uv](https://docs.astral.sh/uv/getting-started/installation/), and a CUDA toolkit.
The tested environment used Python 3.12.13, driver 610.43.03, PyTorch 2.13.0
(CUDA 13.0 build), CUDA toolkit 13.3 at `/usr/local/cuda`, Transformers 5.17.0,
and Humming kernels 0.1.12. The older `/usr/bin/nvcc` on the target could not
compile this runtime's sampler. [Core version pins](../requirements/tested-constraints.txt)
are provided alongside a [runtime package snapshot](../requirements/runtime-snapshot.txt).
The snapshot is an audit artifact, not a standalone lockfile.

The full checkpoint occupies **153,012,762,024 bytes (about 142.5 GiB)** before
environment, source, and cache space. Keep all 13 safetensors files, including
the MTP shards. The PLE shard remains in the model directory; no transformed
embedding file is required.

These commands create a fresh checkout. Existing deployments should preserve
their worktree changes before preparing another checkout.

```bash
git clone https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx.git
cd qwen3.8-flash-next-cmp170hx
export VLLM_WORKDIR="$HOME/vllm"
mkdir -p "$VLLM_WORKDIR"
git clone https://github.com/vllm-project/vllm.git "$VLLM_WORKDIR/src"
git -C "$VLLM_WORKDIR/src" checkout a5a30471ff2bb7f0824f2da10e358af98d304472
./scripts/install-runtime.sh
./scripts/download-model.sh
```

The runtime environment is **`$VLLM_WORKDIR/.venv`**, normally `~/vllm/.venv`.
The installer checks and applies the patch, installs the matching precompiled
wheel and build dependencies, installs the source in editable mode, and compiles
`optimization/ple_ssd_io.so`. The C helper uses Linux system calls and does not
need libaio/liburing or a vLLM CUDA-kernel rebuild. The model download is pinned
to weight revision `ce0e0b94083895bd836b916f29bf105c40a8162a`.

For an already provisioned matching environment, the individual operations are:

```bash
./scripts/apply-patch.sh
./scripts/build-ple-io.sh
```

## Run in screen

From the repository root:

```bash
export VLLM_WORKDIR="$HOME/vllm"
export CUDA_HOME=/usr/local/cuda
./scripts/start.sh
# Model loading and graph preparation took about 8–10 minutes on the target.
curl -f http://127.0.0.1:8000/health
screen -r vllm
```

Detach with `Ctrl-A`, then `D`. Logs are in
`$VLLM_WORKDIR/optimization/server.log`. `start.sh` refuses to replace an
existing `vllm` screen. To restart, stop that session with
`screen -S vllm -X quit`, wait for its engine to exit, then run `start.sh` again.
For a foreground process, use `./scripts/serve.sh`.

The OpenAI-compatible endpoint is `http://localhost:8000/v1`, with model name
`Qwen3.8-Flash-Next`. The launcher binds to `0.0.0.0` by default; override with
`QWEN_HOST` and `QWEN_PORT` as needed. Reasoning and tool parsers are configured
as `qwen3` and `qwen3_xml`.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Explain asynchronous disk I/O."}],"max_tokens":128,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

## Tuning

| Setting | Default |
| --- | --- |
| `QWEN_MTP` | `1` draft token; `0` disables MTP |
| `QWEN_BATCH_TOKENS` / `QWEN_SEQS` | `2048` / `16` |
| `QWEN_GRAPH_MODE` | `FULL_AND_PIECEWISE` |
| `QWEN_GPU_MEMORY` / `QWEN_CONTEXT` | `0.96` / `auto` |
| `QWEN_SSD_DEPTH` / `QWEN_SSD_CACHE_MB` | `256` / `512` |
| `QWEN_SSD_PREFETCH` | `16384` remaining prompt tokens; `0` disables read-ahead |
| `QWEN_OMP_THREADS` | `1` |

The launcher enables breakable CUDA graphs, prefix caching, chunked prefill,
and aligned Mamba state caching. Default graph sizes cover small decode batches
and prefill through 2,048 tokens. `QWEN_CAPTURE_SIZES` overrides the JSON array;
keep it consistent with the batch-token setting. `QWEN_PRINT_COMMAND=1
./scripts/serve.sh` prints the command without starting the model.

## Validate and reproduce timings

Run these against the healthy local API, from this repository's root:

```bash
mkdir -p measurements
"$VLLM_WORKDIR/.venv/bin/python" benchmarks/verify_server.py --long --tools --output measurements/smoke.json
"$VLLM_WORKDIR/.venv/bin/python" benchmarks/bench_server.py --label local --output measurements/chat.json --concurrency 1 4 8 16 --rounds 3
"$VLLM_WORKDIR/.venv/bin/python" benchmarks/bench_prefill.py --label local --output measurements/prefill.json --seed 729156 --repeats 3 --lengths 512 2048 8192 32768
```

Use a new prefill seed to avoid prefix-cache reuse. The scripts retain responses
and token usage. [Validation details](RESULTS.md#validation-and-limits)
include regression commands and known limitations.
