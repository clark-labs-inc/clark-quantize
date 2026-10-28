#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v68}
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
runner="$runtime/run_qwen38_27b_sensitivity_selected_cq3_half_v67.sh"
unit=quant-qwen38-27b-sensitivity-selected-cq3-48-v68
layers='0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,36,37,38,39,40,41,43,44,45,48,49,52,53,56,57,60'
layer_regex='(0|1|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|19|20|21|23|24|25|26|27|28|29|30|31|32|33|34|36|37|38|39|40|41|43|44|45|48|49|52|53|56|57|60)'

if systemctl list-units --type=service --state=running --no-legend \
  | grep -Eiq 'quant-qwen38|qwen35|qwen3[.]5|4b'; then
  echo "another Qwen experiment is active" >&2
  exit 2
fi
test -x "$runner"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"

systemd-run \
  --unit="$unit" \
  --property=Type=exec \
  --property=CPUQuota=400% \
  --property=AllowedCPUs=0,2,4,6 \
  --property=MemoryMax=30000000000 \
  --property=MemorySwapMax=0 \
  --property=Restart=on-failure \
  --property=RestartPreventExitStatus=2 \
  --property=RestartSec=10s \
  --property=WorkingDirectory="$repo" \
  /usr/bin/env \
    REPO="$repo" \
    PREREG_PATH="$runtime/qwen38_27b_sensitivity_selected_cq3_48_preregistration_v68.json" \
    CANDIDATE_MODEL=/workspace/models/candidates/Qwen3.8-27B-Q4_K_M-trained-CQ3-sensitivity48-v68.gguf \
    STATE_DIR=/workspace/quant-state/qwen38-sensitivity-selected-cq3-48-v68 \
    RUN_DIR=/workspace/quant-runs/qwen38-sensitivity-selected-cq3-48-v68 \
    PORT=8108 \
    LAYERS="$layers" \
    LAYER_REGEX="$layer_regex" \
    SELECTED_LAYER_COUNT=48 \
    EXPECTED_BYTES=15163607392 \
    EXPECTED_PREREG_SHA256=eee724ff201de0591a0e8267500387a3fa5c0716958c40f769360b2d30ebc605 \
    PARENT_SUMMARY_PATH="$runtime/receipts/qwen38_27b_sensitivity_selected_cq3_half_v67/summary.json" \
    EXPECTED_PARENT_SUMMARY_SHA256=0f895cea20c3ed5fa4e7be39dd890b92f45113198d73aed8ea34683fe64452cd \
    EXPERIMENT_LABEL=qwen38-v68-sensitivity48 \
    SERVER_UNIT=quant-qwen38-27b-v68-server \
    CANARY_SCHEMA=qwen38_27b_v68_france_canary \
    MATERIALIZATION_SCHEMA=qwen38_27b_sensitivity_selected_cq3_materialization_v68 \
    SUMMARY_SCHEMA=qwen38_27b_sensitivity_selected_cq3_48_summary_v68 \
    MARKER_SCHEMA=qwen38_27b_sensitivity_selected_cq3_48_marker_v68 \
    DECISION_PASS=narrow_interval_to_48_64_test_56 \
    DECISION_FAIL=narrow_interval_to_32_48_test_40 \
    "CLAIM_BOUNDARY=Frozen 48-layer CUTE screening result only; no perplexity, MATH, code, MMLU, throughput, energy, novelty, or breakthrough claim." \
    /usr/bin/bash "$runner"
