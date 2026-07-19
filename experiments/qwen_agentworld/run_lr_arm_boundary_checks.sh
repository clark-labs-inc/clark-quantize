#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/models/Qwen3.5-9B}"
CACHE="${CACHE:?set CACHE to the dense teacher-cache .pt file}"
ARM_DIR="${ARM_DIR:?set ARM_DIR to one lr_* arm directory}"
OUT_DIR="${OUT_DIR:-${ARM_DIR}/boundary_checks}"
START_ROW="${START_ROW:-2000}"
MAX_ROWS="${MAX_ROWS:-}"
MAX_ROWS_PER_TASK="${MAX_ROWS_PER_TASK:-32}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
SHUFFLE_ROWS="${SHUFFLE_ROWS:-1}"
ROW_SEED="${ROW_SEED:-17}"
GATE_DATA_DIR="${GATE_DATA_DIR:-/workspace/AgentWorldBench_round8_fixed10_20260629}"
GATE_ROWS="${GATE_ROWS:-3}"
GATE_LENGTHS="${GATE_LENGTHS:-512}"
PATCH_EXCLUDE_HEAD_EMBED_REGEX="${PATCH_EXCLUDE_HEAD_EMBED_REGEX:-(^|\\.)(embed_tokens|lm_head)(\\.|$)}"
TRAIN_VENV="${TRAIN_VENV:-/workspace/venv}"

if [[ -d "${TRAIN_VENV}" ]]; then
  # shellcheck source=/dev/null
  source "${TRAIN_VENV}/bin/activate"
fi

mkdir -p "${OUT_DIR}"

python - "$ARM_DIR" "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

arm = Path(sys.argv[1])
out = Path(sys.argv[2])
report_path = arm / "report.json"
payload = {
    "arm_dir": str(arm),
    "report": str(report_path),
    "can_run_patch_checks": False,
    "reason": None,
}
if not report_path.exists():
    payload["reason"] = "missing_report_json"
elif not json.loads(report_path.read_text()).get("out_patch"):
    payload["reason"] = "report_has_no_out_patch"
else:
    report = json.loads(report_path.read_text())
    out_patch = Path(report["out_patch"])
    if report.get("skip_final_save"):
        payload["reason"] = "arm_was_launched_with_skip_final_save"
    elif not out_patch.exists():
        payload["reason"] = f"out_patch_missing: {out_patch}"
    else:
        payload["can_run_patch_checks"] = True
        payload["out_patch"] = str(out_patch)
(out / "boundary_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload), flush=True)
if not payload["can_run_patch_checks"]:
    sys.exit(3)
PY

PATCH="$(python - "$OUT_DIR/boundary_manifest.json" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["out_patch"])
PY
)"

eval_args=(
  --model-dir "${MODEL_DIR}"
  --cache "${CACHE}"
  --start-row "${START_ROW}"
  --max-tokens "${MAX_TOKENS}"
)
if [[ "${SHUFFLE_ROWS}" == "1" || "${SHUFFLE_ROWS}" == "true" ]]; then
  eval_args+=(--shuffle-rows --row-seed "${ROW_SEED}")
fi
if [[ -n "${MAX_ROWS}" ]]; then
  eval_args+=(--max-rows "${MAX_ROWS}")
fi
if [[ -n "${MAX_ROWS_PER_TASK}" ]]; then
  eval_args+=(--max-rows-per-task "${MAX_ROWS_PER_TASK}")
fi

run_eval() {
  local out_json="$1"
  shift
  python experiments/qwen_agentworld/eval_teacher_behavior_cache.py "${eval_args[@]}" --out "${out_json}" "$@"
}

echo "eval_init_start"
run_eval "${OUT_DIR}/eval_init.json"

echo "eval_arm_start"
run_eval "${OUT_DIR}/eval_arm.json" --weight-patch "${PATCH}"

echo "eval_arm_fp16_head_embed_start"
run_eval "${OUT_DIR}/eval_arm_fp16_head_embed.json" \
  --weight-patch "${PATCH}" \
  --weight-patch-exclude-regex "${PATCH_EXCLUDE_HEAD_EMBED_REGEX}"

echo "generation_gate_start"
python experiments/qwen_agentworld/run_long_generation_gate.py \
  --model-dir "${MODEL_DIR}" \
  --data-dir "${GATE_DATA_DIR}" \
  --weight-patch "${PATCH}" \
  --out-dir "${OUT_DIR}/generation_gate" \
  --lengths "${GATE_LENGTHS}" \
  --max-rows "${GATE_ROWS}" \
  --max-model-len "${MAX_TOKENS}"

echo "boundary_checks_done out_dir=${OUT_DIR}"
