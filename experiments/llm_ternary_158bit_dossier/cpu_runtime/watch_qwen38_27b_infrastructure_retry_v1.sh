#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
bundle=${BUNDLE:?BUNDLE must name the frozen Qwen3.8-27B retry source bundle}
expected_commit=${EXPECTED_COMMIT:?EXPECTED_COMMIT must pin the deployed retry source}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-infrastructure-retry-v1}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime
amendment="$runtime/qwen38_27b_infrastructure_retry_amendment_v1.json"

units=(
  quant-qwen38-27b-scale-codebook4-watch-v1.service
  quant-qwen38-27b-block203-codec-watch-v2.service
  quant-qwen38-27b-block203-radix243-watch-v1.service
)
while :; do
  active=false
  for unit in "${units[@]}"; do
    if systemctl is-active --quiet "$unit"; then
      active=true
      break
    fi
  done
  if [[ "$active" == false ]]; then
    break
  fi
  sleep 30
done

mkdir -p "$state"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty Qwen3.8-27B retry repository" >&2
  exit 73
fi
git fetch "$bundle" refs/heads/codex/qwen38-base3-research
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed Qwen3.8-27B retry source commit" >&2
  exit 65
fi
sha256sum "$amendment" > "$state/amendment.sha256"
git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"

"$runtime/run_qwen38_27b_cpu_throughput_v1.sh"
printf 'complete\n' > "$state/BASELINE_THROUGHPUT_COMPLETE"

"$runtime/run_qwen38_27b_layer0_mlp_direct_cq3_v1.sh"
printf 'complete\n' > "$state/DIRECT_CANDIDATE_COMPLETE"

"$runtime/run_qwen38_27b_layer0_mlp_direct_cq3_quality_v1.sh"
printf 'complete\n' > "$state/DIRECT_QUALITY_COMPLETE"
if [[ -s /workspace/quant-state/qwen38-27b-layer0-mlp-direct-cq3-quality-v1/PROMOTION_PASS ]]; then
  "$runtime/run_qwen38_27b_layer0_mlp_cq3_paired_throughput_v1.sh"
  printf 'complete\n' > "$state/PAIRED_THROUGHPUT_COMPLETE"
else
  "$python_bin" - "$state" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

state = Path(sys.argv[1])
quality = Path("/workspace/quant-state/qwen38-27b-layer0-mlp-direct-cq3-quality-v1/summary.json")
content = quality.read_bytes()
receipt = {
    "schema": "qwen38_27b_paired_throughput_quality_skip_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "quality_summary": str(quality),
    "quality_summary_sha256": hashlib.sha256(content).hexdigest(),
    "terminal": True,
    "valid_nonpromotion": True,
    "reason": "the frozen direct-CQ3 quality gate rejected before paired throughput",
    "paired_throughput_launched": False,
}
(state / "paired-throughput-skip.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
)
(state / "PAIRED_THROUGHPUT_SKIPPED").write_text("quality rejected\n")
PY
fi

"$runtime/run_qwen38_27b_cq3_x86_pair_v1.sh"
printf 'complete\n' > "$state/X86_PAIR_TERMINAL"

mapped_scale=/workspace/quant-runs/qwen38-scale-codebook4-v1/manifest.json
if ! "$python_bin" - "$mapped_scale" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
raise SystemExit(
    0
    if payload.get("complete") is True and payload.get("gate", {}).get("passed") is True
    else 1
)
PY
then
  "$python_bin" - "$mapped_scale" "$state" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

parent = Path(sys.argv[1])
state = Path(sys.argv[2])
content = parent.read_bytes()
receipt = {
    "schema": "qwen38_27b_complete_scale_and_export_parent_rejection_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "parent_receipt": str(parent),
    "parent_sha256": hashlib.sha256(content).hexdigest(),
    "terminal": True,
    "valid_nonpromotion": True,
    "reason": "the frozen mapped 402-tensor scale-codebook gate rejected",
    "supplemental_scale_fit_launched": False,
    "container_export_launched": False,
}
(state / "complete-scale-export-skip.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
)
(state / "COMPLETE_SCALE_AND_EXPORT_SKIPPED").write_text("mapped scale gate rejected\n")
PY
else
  set +e
  "$python_bin" "$runtime/run_qwen38_27b_full_scale_codebook4_v1.py"
  full_scale_status=$?
  set -e
  full_scale=/workspace/quant-runs/qwen38-full-scale-codebook4-v1/manifest.json
  if [[ ! -s "$full_scale" ]]; then
    echo "complete scale-codebook runner exited $full_scale_status without a receipt" >&2
    exit 70
  fi
  if "$python_bin" - "$full_scale" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
raise SystemExit(
    0
    if payload.get("complete") is True and payload.get("gate", {}).get("passed") is True
    else 1
)
PY
  then
    if (( full_scale_status != 0 )); then
      echo "complete scale-codebook gate passed but runner exited $full_scale_status" >&2
      exit 70
    fi
    printf 'pass\n' > "$state/COMPLETE_SCALE_PASS"
    scripts/cpu/build_ternary_cpu.sh > "$state/build-native.stdout" 2> "$state/build-native.stderr"
    set +e
    "$python_bin" "$runtime/run_qwen38_27b_cq3_export_v1.py"
    export_status=$?
    set -e
    export_receipt=/workspace/quant-state/qwen38-27b-cq3-export-v1.json
    if [[ ! -s "$export_receipt" ]]; then
      echo "CQ3 exporter exited $export_status without a receipt" >&2
      exit 70
    fi
    if (( export_status != 0 && export_status != 2 )); then
      echo "CQ3 exporter had unexpected exit $export_status" >&2
      exit "$export_status"
    fi
    printf 'terminal\n' > "$state/CQ3_EXPORT_TERMINAL"
  else
    if (( full_scale_status != 2 )); then
      echo "complete scale-codebook rejection had unexpected exit $full_scale_status" >&2
      exit 70
    fi
    "$python_bin" - "$full_scale" "$state" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

parent = Path(sys.argv[1])
state = Path(sys.argv[2])
content = parent.read_bytes()
receipt = {
    "schema": "qwen38_27b_cq3_export_full_scale_rejection_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "parent_receipt": str(parent),
    "parent_sha256": hashlib.sha256(content).hexdigest(),
    "terminal": True,
    "valid_nonpromotion": True,
    "reason": "the frozen complete 666-tensor scale-codebook gate rejected",
    "container_export_launched": False,
}
(state / "cq3-export-skip.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
)
(state / "CQ3_EXPORT_SKIPPED").write_text("complete scale gate rejected\n")
PY
  fi
fi

"$python_bin" - "$state" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

state = Path(sys.argv[1])
files = {}
for path in sorted(state.iterdir()):
    if path.is_file() and path.name not in {"receipt.json", "RETRY_TERMINAL"}:
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
receipt = {
    "schema": "qwen38_27b_infrastructure_retry_receipt_v1",
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "terminal": True,
    "stages": {
        "baseline_throughput": (state / "BASELINE_THROUGHPUT_COMPLETE").is_file(),
        "direct_candidate": (state / "DIRECT_CANDIDATE_COMPLETE").is_file(),
        "direct_quality": (state / "DIRECT_QUALITY_COMPLETE").is_file(),
        "paired_throughput_complete": (state / "PAIRED_THROUGHPUT_COMPLETE").is_file(),
        "paired_throughput_skipped": (state / "PAIRED_THROUGHPUT_SKIPPED").is_file(),
        "x86_pair_terminal": (state / "X86_PAIR_TERMINAL").is_file(),
        "complete_scale_pass": (state / "COMPLETE_SCALE_PASS").is_file(),
        "complete_scale_and_export_skipped": (state / "COMPLETE_SCALE_AND_EXPORT_SKIPPED").is_file(),
        "cq3_export_terminal": (state / "CQ3_EXPORT_TERMINAL").is_file(),
        "cq3_export_skipped": (state / "CQ3_EXPORT_SKIPPED").is_file(),
    },
    "artifacts": files,
    "claim_boundary": "orchestration receipt only; scientific claims remain in each frozen stage receipt",
}
(state / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(state / "RETRY_TERMINAL").write_text("terminal\n")
PY
