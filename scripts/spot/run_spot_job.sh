#!/usr/bin/env bash
# Run one resumable job with atomic status receipts on an interruptible node.
set -euo pipefail

if [[ $# -lt 3 || "$2" != "--" ]]; then
  echo "usage: $0 JOB_NAME -- COMMAND [ARGS...]" >&2
  exit 2
fi

JOB_NAME="$1"
shift 2
RUN_ROOT="${QUANT_RUN_ROOT:-/workspace/quant-runs}"
STATE_ROOT="${QUANT_STATE_ROOT:-/workspace/quant-state}"
REPO="${QUANT_REPO:-/workspace/clark-quantize}"
RUN_DIR="${RUN_ROOT}/${JOB_NAME}"
STATUS_PY="${REPO}/scripts/spot/spot_status.py"

mkdir -p "${RUN_DIR}" "${STATE_ROOT}"
exec 9>"${STATE_ROOT}/job.lock"
if ! flock -w 5 9; then
  echo "another quant spot job owns ${STATE_ROOT}/job.lock" >&2
  exit 1
fi

COMMAND_FILE="${RUN_DIR}/command.sh"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\nexec ' "${REPO}"
  printf '%q ' "$@"
  printf '\n'
} >"${COMMAND_FILE}.tmp"
chmod 700 "${COMMAND_FILE}.tmp"
mv "${COMMAND_FILE}.tmp" "${COMMAND_FILE}"

date -u +%FT%TZ >"${RUN_DIR}/started_at.txt"
python3 "${STATUS_PY}" --event started --job "${JOB_NAME}"

child_pid=""
child_group=""
heartbeat_pid=""
finish() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [[ -n "${heartbeat_pid}" ]]; then
    # The heartbeat shell can be blocked in its child sleep.  Kill that child
    # first so the service does not retain an orphan process after the job, and
    # reap the shell before releasing our lock descriptor.
    pkill -TERM -P "${heartbeat_pid}" 2>/dev/null || true
    kill "${heartbeat_pid}" 2>/dev/null || true
    wait "${heartbeat_pid}" 2>/dev/null || true
  fi
  sync
  python3 "${STATUS_PY}" --event exited --job "${JOB_NAME}" \
    ${child_pid:+--pid "${child_pid}"} --exit-code "${rc}" || true
  date -u +%FT%TZ >"${RUN_DIR}/finished_at.txt"
  printf '%s\n' "${rc}" >"${RUN_DIR}/exit_code.txt"
  exit "${rc}"
}
request_stop() {
  local signal="$1"
  date -u +%FT%TZ >"${RUN_DIR}/checkpoint-requested-${signal}.txt"
  python3 "${STATUS_PY}" --event "signal_${signal}" --job "${JOB_NAME}" \
    ${child_pid:+--pid "${child_pid}"} || true
  sync
  if [[ -n "${child_pid}" ]]; then
    if [[ -n "${child_group}" ]]; then
      kill -TERM -- "-${child_group}" 2>/dev/null || true
    else
      kill -TERM "${child_pid}" 2>/dev/null || true
    fi
  fi
}
trap finish EXIT
trap 'request_stop INT' INT
trap 'request_stop TERM' TERM
trap 'request_stop HUP' HUP

set +e
if command -v setsid >/dev/null 2>&1; then
  # Only this harness process owns the inter-job lock.  The command, tee, and
  # heartbeat must not inherit fd 9 or a short-lived helper can block the next
  # receipt-gated stage after the command itself has completed.
  setsid "$@" 9>&- > >(tee -a "${RUN_DIR}/job.log" 9>&-) 2>&1 &
  child_group=$!
else
  "$@" 9>&- > >(tee -a "${RUN_DIR}/job.log" 9>&-) 2>&1 &
fi
child_pid=$!
printf '%s\n' "${child_pid}" >"${RUN_DIR}/pid.txt"
python3 "${STATUS_PY}" --event running --job "${JOB_NAME}" --pid "${child_pid}" || true
(
  exec 9>&-
  while true; do
    python3 "${STATUS_PY}" --event heartbeat --job "${JOB_NAME}" --pid "${child_pid}" || true
    sleep 30
  done
) &
heartbeat_pid=$!
wait "${child_pid}"
rc=$?
set -e
exit "${rc}"
