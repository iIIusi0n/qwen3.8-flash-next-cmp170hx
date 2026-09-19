# Qwen3.8 Flash Next on CMP 170HX: async PLE SSD offload for vLLM

Serve [klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP](https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP)
with its **95.37 GiB BF16 PLE table on SSD**, on one 64 GiB SM80 NVIDIA GPU.
This repository contains the full tested vLLM patch, setup scripts, and PP/TG
measurements from September 19, 2026. It does not redistribute model weights.

The deployment used a CMP 170HX target reported by the driver as `NVIDIA Graphics
Device`: SM80, 64 GiB VRAM, 180 W power limit, four vCPUs, 15 GiB host RAM,
31 GiB swap, and enterprise SATA storage exposed through a QEMU block device.
This was a measured configuration, not a minimum RAM or swap guarantee.

## Results

| Workload | Previous tuned server | Final default | Improvement |
| --- | ---: | ---: | ---: |
| Fresh 8K prefill | 1,422 prompt tok/s | **2,636 prompt tok/s** | **85%** |
| Single-request decode | 82.5 tok/s | **111 tok/s** | **35%** |
| Aggregate output, 16 concurrent requests | 387 tok/s | **667 tok/s** | **72%** |

The chat comparison uses 128 output tokens, temperature zero, thinking disabled,
and EOS ignored. Final numbers are three-round medians; the previous tuned
baseline used two rounds. Aggregate output throughput includes prefill and HTTP
overhead. Single-request decode excludes time to first token. The baseline
already had threaded SSD offload and piecewise graphs; it was not stock vLLM.

The final server also measured **117 decode tok/s** for 512-token single-user
outputs and **2,132 prompt tok/s** for fresh 32K prefills. Three-token MTP reached
**150 single-request decode tok/s** in a separate configuration; the default
uses one draft token for better aggregate throughput under load.

PP originally spent 51% of its profiled wall time waiting for PLE delivery;
the selected configuration reduced that to 36%. TG is now mainly target-model
GPU execution, with single-request GPU utilization rising from 48% to 94%.
See [full results and limits](docs/RESULTS.md),
[stage profiles](docs/PERFORMANCE.md), and [raw measurements](results/README.md).

## What must be added to vLLM

Apply the **complete [patch](patches/qwen38-ple-ssd.patch)** to vLLM commit
[`a5a30471ff2bb7f0824f2da10e358af98d304472`](https://github.com/vllm-project/vllm/commit/a5a30471ff2bb7f0824f2da10e358af98d304472),
then build the included C I/O helper. The patch adds:

1. Header-only PLE loading and exact BF16 row lookup from the original checkpoint.
2. Native Linux AIO with `O_DIRECT`, a bounded queue, pinned staging, CUDA stream
   overlap, an LRU cache, and bounded prompt read-ahead.
3. Required host breaks inside full decode graphs, keeping attention and MTP
   execution captured; larger piecewise graphs cover prefill.
4. INC mixed-bit expert configuration handling and MTP layer-number remapping
   for this checkpoint.

These are required together for the documented deployment. The original
architecture/AutoRound support alone does not provide this SSD path.
The changes span nine production files, including two new files, plus tests in
four existing test files. The patch does not introduce extra quantization or a
new approximation to model arithmetic. See [file-by-file details](docs/IMPLEMENTATION.md).

Patch SHA-256:
`09c4e13fbd2b77169a7f8cb28d36242cc6e3916fe9c5a3c25d8e90101a38ccb0`.

## Setup

Use Linux x86-64, a CUDA-compatible driver, a C compiler, Git, GNU screen,
[uv](https://docs.astral.sh/uv/getting-started/installation/), and a CUDA toolkit.
The tested environment used Python 3.12.13, driver 610.43.03, PyTorch 2.13.0
(CUDA 13.0 build), CUDA toolkit 13.3 at `/usr/local/cuda`, Transformers 5.17.0,
and Humming kernels 0.1.12. The older `/usr/bin/nvcc` on the target could not
compile this runtime's sampler. [Core version pins](requirements/tested-constraints.txt)
are provided alongside a [runtime package snapshot](requirements/runtime-snapshot.txt).
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

From this repository's root:

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

To select the three-token MTP configuration used for the faster single-user
comparison, start a stopped server with:

```bash
QWEN_MTP=3 QWEN_BATCH_TOKENS=1024 \
QWEN_CAPTURE_SIZES='[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,64]' \
./scripts/start.sh
```

Its measured single-user decode rate was 149.7 tok/s, but concurrency-16
aggregate throughput was 428.4 tok/s, versus 667.1 in the final default.
The earlier dynamic draft-count schedule was slower with piecewise graphs and
is not selected. No configuration was fastest on every measured workload.

## Validate and reproduce timings

Run these against the healthy local API, from this repository's root:

```bash
mkdir -p measurements
"$VLLM_WORKDIR/.venv/bin/python" benchmarks/verify_server.py --long --tools --output measurements/smoke.json
"$VLLM_WORKDIR/.venv/bin/python" benchmarks/bench_server.py --label local --output measurements/chat.json --concurrency 1 4 8 16 --rounds 3
"$VLLM_WORKDIR/.venv/bin/python" benchmarks/bench_prefill.py --label local --output measurements/prefill.json --seed 729156 --repeats 3 --lengths 512 2048 8192 32768
```

Use a new prefill seed to avoid prefix-cache reuse. The scripts retain responses
and token usage. [Validation details](docs/RESULTS.md#correctness-and-remaining-limits)
include regression commands and known limitations.

## Scope and limitations

- One GPU (`TP=DP=1`), BF16 PLE, and a local indexed safetensors checkpoint with
  dedicated PLE shards. The original non-SSD path remains available.
- The API fitted and advertises 262,144-token context. The largest varied-token
  timing test was 32,768; a full 262K request was not evaluated.
- Ten SSD regressions and 21 graph-primitive tests passed. Final validation
  completed 134 requests with zero API errors, aborts, or preemptions.
- An arithmetic prompt was inconsistent in **both** piecewise and full graph
  modes. Exact-row and CPU/GPU n-gram checks passed, but this runtime/model
  consistency limitation remains unresolved. Successful smoke tests are not a
  comprehensive model-quality evaluation.
- The broader upstream PLE suite contains an unchanged FP8 pinned-memory kernel
  that does not compile on this SM80 target. This deployment uses BF16 SSD reads.
- The portable wrappers reproduce the tested settings; a fresh dependency
  installation on other machines was not benchmarked. The patch itself is the
  exact artifact used by the measured deployment.

## License and provenance

The patch derives from [vLLM](https://github.com/vllm-project/vllm) under
Apache-2.0; code and scripts in this repository use the included [Apache-2.0
license](LICENSE). Model weights retain the [Qwen Community License 1.0](https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP/blob/main/LICENSE).
Development and documentation were AI-assisted; measured results and validation
limits are supplied for review.
