#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export VLLM_WORKDIR="${VLLM_WORKDIR:-$HOME/vllm}"
export VLLM_SOURCE="$VLLM_WORKDIR/src"
export VLLM_VENV="$VLLM_WORKDIR/.venv"
export QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$VLLM_WORKDIR/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP}"
VLLM_REVISION=a5a30471ff2bb7f0824f2da10e358af98d304472
MODEL_REVISION=ce0e0b94083895bd836b916f29bf105c40a8162a
VLLM_WHEEL="https://wheels.vllm.ai/$VLLM_REVISION/vllm-0.29.1rc1.dev402%2Bga5a30471f-cp38-abi3-manylinux_2_28_x86_64.whl"
