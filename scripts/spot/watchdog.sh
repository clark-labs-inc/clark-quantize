#!/usr/bin/env bash
set -euo pipefail

REPO="${QUANT_REPO:-/workspace/clark-quantize}"
STATUS_PY="${REPO}/scripts/spot/spot_status.py"
STATE_ROOT="${QUANT_STATE_ROOT:-/workspace/quant-state}"
mkdir -p "${STATE_ROOT}"
exec 8>"${STATE_ROOT}/job.lock"
while true; do
  # The foreground job owns this same exclusive lock. Only publish idle-node
  # health when doing so cannot erase a running job's authoritative status.
  if flock -n 8; then
    python3 "${STATUS_PY}" --event heartbeat
    flock -u 8
  fi
  sleep 30
done
