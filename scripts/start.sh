#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if screen -ls | grep -q '[.]vllm[[:space:]]'; then
    echo 'The vllm screen already exists. Attach with: screen -r vllm' >&2
    exit 1
fi
mkdir -p "$VLLM_WORKDIR/optimization"
log="$VLLM_WORKDIR/optimization/server.log"
if [[ -f "$log" ]]; then
    mv "$log" "$VLLM_WORKDIR/optimization/server-$(date +%s).log"
fi
screen -L -Logfile "$log" -dmS vllm "$REPO_DIR/scripts/serve.sh"
echo "Started screen vllm; log: $log"
echo 'Wait for readiness: curl -f http://127.0.0.1:8000/health'
