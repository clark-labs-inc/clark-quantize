#!/usr/bin/env bash
set -euo pipefail

# Terminal-gated handoff for the Qwen/Qwen3.8-27B affine block experiment.
# Run this watcher as a root-owned transient service.  It performs no training;
# it only stops a terminal parent restart loop and starts the frozen QAT runner
# when the authenticated three-block PTQ gate passed.

readonly model="Qwen/Qwen3.8-27B"
readonly parent_unit="quant-qwen38-27b-affine-butterfly-block-gate-v1.service"
readonly qat_unit="quant-qwen38-27b-affine-butterfly-block-qat-v1.service"
readonly parent_receipt="/workspace/quant-state/qwen38-affine-butterfly-block-gate-v1.json"
readonly qat_receipt="/workspace/quant-state/qwen38-affine-butterfly-block-qat-v1.json"
readonly qat_repo="/workspace/clark-quantize-affine-qat-v1"
readonly qat_repo_commit="1343f9616fc145ec9dbfb6716f6f8c9e21006cca"
readonly parent_runner_commit="1e524a9eb48e223b259a4d3714afc765bb107dcd"
readonly qat_prereg_sha256="28d7ef3ca0eca806fc25e8a0b7de315a82e9e295fb2302210ad9720a1fdd0a26"
readonly python_bin="/workspace/venvs/quant-cpu/bin/python"
readonly qat_runner="experiments/llm_ternary_158bit_dossier/cpu_runtime/run_qwen38_27b_affine_butterfly_block_qat_v1.py"

if [[ -f "$qat_receipt" ]]; then
  jq -e --arg model "$model" '
    .schema == "qwen38_27b_affine_butterfly_block_qat_receipt_v1" and
    .model == $model and .complete == true and .terminal == true
  ' "$qat_receipt" >/dev/null
  exit 0
fi

while [[ ! -f "$parent_receipt" ]]; do
  if ! systemctl is-active --quiet "$parent_unit"; then
    echo "parent PTQ is inactive without a terminal receipt" >&2
    exit 75
  fi
  sleep 30
done

jq -e --arg model "$model" --arg commit "$parent_runner_commit" '
  .schema == "qwen38_27b_affine_butterfly_block_gate_receipt_v1" and
  .model == $model and .repository_commit == $commit and
  .complete == true and .terminal == true
' "$parent_receipt" >/dev/null

# A scientific rejection is terminal.  Stop a Restart=on-failure loop and do
# not launch QAT or reinterpret exit status 2 as infrastructure failure.
systemctl stop "$parent_unit" 2>/dev/null || true
if ! jq -e '.gate.passed == true' "$parent_receipt" >/dev/null; then
  echo "parent PTQ gate rejected; QAT remains unrun"
  exit 0
fi

[[ "$(git -C "$qat_repo" rev-parse HEAD)" == "$qat_repo_commit" ]]
[[ -z "$(git -C "$qat_repo" status --porcelain)" ]]
git -C "$qat_repo" cat-file -e "${parent_runner_commit}^{commit}"
[[ "$(sha256sum "$qat_repo/experiments/llm_ternary_158bit_dossier/cpu_runtime/qwen38_27b_affine_butterfly_block_qat_preregistration_v1.json" | awk '{print $1}')" == "$qat_prereg_sha256" ]]

if systemctl is-active --quiet "$qat_unit"; then
  exit 0
fi
systemctl reset-failed "$qat_unit" 2>/dev/null || true
systemd-run \
  --unit="$qat_unit" \
  --description="Qwen3.8-27B sensitivity-aware affine block QAT v1" \
  --working-directory="$qat_repo" \
  --property=Restart=on-failure \
  --property=RestartPreventExitStatus=2 \
  --property=RestartSec=15s \
  --property=CPUAccounting=yes \
  --property=MemoryAccounting=yes \
  "$python_bin" "$qat_runner" --threads 4

