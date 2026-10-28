#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/clark-quantize}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venv/bin/python}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the sealed six-arm token directory}"
OUT_ROOT="${OUT_ROOT:?set OUT_ROOT to an absolute durable run directory}"
STOP_AFTER_TOKENS="${STOP_AFTER_TOKENS:-100000000}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
CAPABILITY_GATE_RECEIPT="${CAPABILITY_GATE_RECEIPT:-}"
MODEL_ID="Qwen/Qwen3.5-4B"
MODEL_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MIN_FREE_GPU_MIB=$((65 * 1024))
MIN_FREE_DISK_KIB=$((81481656832 / 1024))

case "${DATA_ROOT}" in
  /*) ;;
  *) echo "DATA_ROOT must be absolute" >&2; exit 2 ;;
esac
case "${OUT_ROOT}" in
  /*) ;;
  *) echo "OUT_ROOT must be absolute" >&2; exit 2 ;;
esac

test -x "${PYTHON_BIN}"
test -f "${DATA_ROOT}/manifest.json"
command -v nvidia-smi >/dev/null

GPU_TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits --id=0 | tr -d ' ')"
GPU_FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=0 | tr -d ' ')"
if (( GPU_TOTAL_MIB < 80000 )); then
  echo "refusing launch: GPU 0 has ${GPU_TOTAL_MIB} MiB total; this recipe requires an 80 GB class GPU" >&2
  exit 2
fi
if (( GPU_FREE_MIB < MIN_FREE_GPU_MIB )); then
  echo "refusing launch: GPU 0 has ${GPU_FREE_MIB} MiB free; require ${MIN_FREE_GPU_MIB} MiB" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}"
DISK_FREE_KIB="$(df -Pk "${OUT_ROOT}" | awk 'NR==2 {print $4}')"
if (( DISK_FREE_KIB < MIN_FREE_DISK_KIB )); then
  echo "refusing launch: output filesystem has ${DISK_FREE_KIB} KiB free; require ${MIN_FREE_DISK_KIB} KiB" >&2
  exit 2
fi

cd "${REPO_ROOT}"
git rev-parse HEAD

TRAIN_ARGS=(
  scripts/train_qwen35_ternary_kd.py
  --model "${MODEL_ID}"
  --revision "${MODEL_REVISION}"
  --data-root "${DATA_ROOT}"
  --out "${OUT_ROOT}"
  --budget-tokens 5000000000
  --stop-after-tokens "${STOP_AFTER_TOKENS}"
  --seed 20260817
  --seq-len 1536
  --micro-batch 1
  --grad-accum 64
  --master-lr 3e-3
  --train-norms
  --norm-lr 1e-4
  --optimizer muon-fast
  --grad-clip 20
  --hidden-layers full
  --hidden-weight 1.0
  --hidden-anneal-floor 0.0
  --anchor-frac 0.2
  --kd-temperature 2.0
  --kd-topk 128
  --probe-every 150
  --ckpt-every 250
  --checkpoint-keep 2
  --checkpoint-reserve-gb 10
  --min-free-gb 65
)

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi
if [[ -n "${CAPABILITY_GATE_RECEIPT}" ]]; then
  TRAIN_ARGS+=(--capability-gate-receipt "${CAPABILITY_GATE_RECEIPT}")
fi

exec "${PYTHON_BIN}" "${TRAIN_ARGS[@]}"
