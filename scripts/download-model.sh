#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
export MODEL_REVISION
"$VLLM_VENV/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP",
    revision=os.environ["MODEL_REVISION"],
    local_dir=os.environ["QWEN_MODEL_DIR"],
    max_workers=2,
)
PY
