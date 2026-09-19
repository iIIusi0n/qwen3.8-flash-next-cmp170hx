#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
"$REPO_DIR/scripts/apply-patch.sh"
if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
    uv venv --python 3.12 "$VLLM_VENV"
fi
uv pip install --python "$VLLM_VENV/bin/python" --torch-backend=cu130 \
    -c "$REPO_DIR/requirements/tested-constraints.txt" "$VLLM_WHEEL"
uv pip install --python "$VLLM_VENV/bin/python" --torch-backend=cu130 \
    -c "$REPO_DIR/requirements/tested-constraints.txt" \
    -r "$VLLM_SOURCE/requirements/build/cuda.txt" 'huggingface-hub==1.32.0'
export VLLM_USE_PRECOMPILED=1
export VLLM_VERSION_OVERRIDE=0.29.1rc1.dev402+ga5a30471f.ple1
export VLLM_PRECOMPILED_WHEEL_LOCATION="$VLLM_WHEEL"
cd "$VLLM_SOURCE"
uv pip install --python "$VLLM_VENV/bin/python" --reinstall \
    --config-settings editable_mode=compat --no-deps --no-build-isolation -e .
"$REPO_DIR/scripts/build-ple-io.sh"
