#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
mkdir -p "$VLLM_WORKDIR/optimization"
cc -O3 -shared -fPIC -Wall -Wextra -Werror \
    "$VLLM_SOURCE/vllm/models/qwen4_exp/nvidia/ple_ssd_io.c" \
    -o "$VLLM_WORKDIR/optimization/ple_ssd_io.so.tmp"
mv "$VLLM_WORKDIR/optimization/ple_ssd_io.so.tmp" \
    "$VLLM_WORKDIR/optimization/ple_ssd_io.so"
