#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
patch="$REPO_DIR/patches/qwen38-ple-ssd.patch"
cd "$VLLM_SOURCE"
if git apply --reverse --check "$patch" 2>/dev/null; then
    echo 'The complete PLE SSD patch is already applied.'
    exit 0
fi
if [[ $(git rev-parse HEAD) != "$VLLM_REVISION" ]]; then
    echo "Expected vLLM revision $VLLM_REVISION; check out that revision first." >&2
    exit 1
fi
git apply --check "$patch"
git apply "$patch"
git diff --check
