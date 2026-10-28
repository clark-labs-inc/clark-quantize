#!/usr/bin/env bash
# Shared systemd wait primitive for receipt-gated Qwen3.8-27B experiments.

qwen38_unit_has_job() {
  local unit=${1:?unit is required}
  "${SYSTEMCTL_BIN:-systemctl}" list-jobs --no-legend --plain 2>/dev/null |
    awk -v target="$unit" '$2 == target { found = 1 } END { exit(found ? 0 : 1) }'
}

qwen38_wait_for_terminal_receipt() {
  local unit=${1:?unit is required}
  local receipt=${2:?receipt is required}
  local poll_seconds=${3:-30}
  if (( poll_seconds <= 0 )); then
    echo "Qwen3.8 receipt poll interval must be positive" >&2
    return 64
  fi
  while "${SYSTEMCTL_BIN:-systemctl}" is-active --quiet "$unit" ||
    qwen38_unit_has_job "$unit"; do
    sleep "$poll_seconds"
  done
  if [[ ! -s "$receipt" ]]; then
    echo "Qwen3.8 parent unit lacks a terminal receipt: $unit -> $receipt" >&2
    return 3
  fi
}
