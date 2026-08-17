#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-cpu-qat}
bundle=${BUNDLE:?BUNDLE must name the frozen Qwen3.8-27B activation-QAT source bundle}
bundle_branch=${BUNDLE_BRANCH:-refs/heads/codex/qwen38-base3-research}
expected_commit=${EXPECTED_COMMIT:-}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-layer0-mlp-activation-qat-watch-v1}
parent_state=${PARENT_STATE:-/workspace/quant-state/qwen38-27b-infrastructure-retry-v1}
parent_unit=${PARENT_UNIT:-quant-qwen38-27b-infrastructure-retry-v1.service}
geometry_unit=${GEOMETRY_UNIT:-quant-qwen38-27b-geometric-ternary-ablation-v2.service}
rate_unit=${RATE_UNIT:-quant-qwen38-27b-ternary-type-class-rate-v1.service}
hash_unit=${HASH_UNIT:-quant-qwen38-27b-hash-residual-screen-v1.service}
geometry_receipt=${GEOMETRY_RECEIPT:-/workspace/quant-state/qwen38-27b-geometric-ternary-ablation-v1.json}
rate_receipt=${RATE_RECEIPT:-/workspace/quant-state/qwen38-27b-ternary-type-class-rate-v1.json}
hash_receipt=${HASH_RECEIPT:-/workspace/quant-state/qwen38-27b-hash-residual-screen-v1.json}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime
candidate=/workspace/models/controls/Qwen3.8-27B-Q4_K_M-layer0-MLP-CQ3_0-activation-QAT-v1.gguf
materialization=/workspace/quant-state/qwen38-27b-layer0-mlp-activation-qat-materialization-v1.json
training=/workspace/quant-state/qwen38-27b-layer0-mlp-activation-qat-v1.json
quality=/workspace/quant-state/qwen38-27b-layer0-mlp-activation-qat-quality-v1

if [[ -z "$expected_commit" ]]; then
  mapfile -t bundle_commits < <(
    git bundle list-heads "$bundle" "$bundle_branch" |
      awk -v branch="$bundle_branch" '$2 == branch {print $1}'
  )
  if (( ${#bundle_commits[@]} != 1 )) ||
    [[ ! "${bundle_commits[0]}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "frozen Qwen3.8-27B bundle does not contain exactly one pinned branch head" >&2
    exit 64
  fi
  expected_commit=${bundle_commits[0]}
fi

for prerequisite_unit in "$parent_unit" "$geometry_unit" "$rate_unit" "$hash_unit"; do
  while systemctl is-active --quiet "$prerequisite_unit"; do
    sleep 30
  done
done
if [[ ! -s "$parent_state/RETRY_TERMINAL" || ! -s "$parent_state/receipt.json" ]]; then
  echo "parent Qwen3.8-27B retry did not produce its terminal receipt" >&2
  exit 3
fi
"$python_bin" - "$geometry_receipt" "$rate_receipt" "$hash_receipt" <<'PY'
import json
from pathlib import Path
import sys

expected = {
    "qwen38_27b_geometric_ternary_ablation_receipt_v1",
    "qwen38_27b_ternary_type_class_rate_receipt_v1",
    "qwen38_27b_hash_residual_screen_receipt_v1",
}
observed = set()
for raw in sys.argv[1:]:
    path = Path(raw)
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Qwen3.8-27B prerequisite receipt is absent: {path}")
    payload = json.loads(path.read_text())
    observed.add(payload.get("schema"))
    gate = payload.get("gate")
    if not isinstance(gate, dict) or not isinstance(gate.get("passed"), bool):
        raise SystemExit(f"Qwen3.8-27B prerequisite receipt is nonterminal: {path}")
if observed != expected:
    raise SystemExit(f"Qwen3.8-27B prerequisite receipt schemas changed: {observed}")
PY
for marker in \
  /workspace/quant-state/qwen38-27b-cpu-q4-baseline-v1/QUALITY_COMPLETE \
  /workspace/quant-state/qwen38-27b-layer0-mlp-direct-cq3-v1/CANDIDATE_READY \
  /workspace/quant-state/qwen38-27b-layer0-mlp-direct-cq3-quality-v1/QUALITY_COMPLETE; do
  if [[ ! -s "$marker" ]]; then
    echo "required Qwen3.8-27B parent marker is absent: $marker" >&2
    exit 3
  fi
done

mkdir -p "$state"
sha256sum "$parent_state/receipt.json" "$geometry_receipt" "$rate_receipt" \
  "$hash_receipt" > "$state/prerequisites.sha256"
cd "$repo"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing dirty Qwen3.8-27B research repository" >&2
  exit 73
fi
git fetch "$bundle" "$bundle_branch"
git merge --ff-only FETCH_HEAD
if [[ "$(git rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "unexpected deployed Qwen3.8-27B activation-QAT commit" >&2
  exit 65
fi
git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"
sha256sum \
  "$runtime/qwen38_27b_layer0_mlp_activation_qat_preregistration_v1.json" \
  "$runtime/collect_qwen38_27b_layer0_mlp_activations_v1.py" \
  "$runtime/train_qwen38_27b_layer0_mlp_activation_qat_v1.py" \
  "$runtime/materialize_qwen38_27b_layer0_mlp_activation_qat_v1.py" \
  "$runtime/run_qwen38_27b_layer0_mlp_activation_qat_quality_v1.sh" \
  > "$state/source.sha256"

llama_conflicts=$(ps -eo pid=,comm=,args= | awk \
  '$2 ~ /^llama-(server|cli|bench|perplexi)/ {print}')
if [[ -n "$llama_conflicts" ]]; then
  echo "another llama.cpp workload is active" >&2
  printf '%s\n' "$llama_conflicts" >&2
  exit 2
fi
training_conflicts=$(ps -eo pid=,comm=,args= | awk \
  '$2 ~ /^(python|python3|torchrun)$/ && $0 ~ /(train_mlp|qwen.*qat|activation.*cache)/ {print}')
if [[ -n "$training_conflicts" ]]; then
  echo "another QAT or activation-cache process is active" >&2
  printf '%s\n' "$training_conflicts" >&2
  exit 2
fi
free -b > "$state/memory-before.txt"
df -B1 /workspace > "$state/disk-before.txt"

"$python_bin" "$runtime/collect_qwen38_27b_layer0_mlp_activations_v1.py" \
  > "$state/activation-cache.stdout" 2> "$state/activation-cache.stderr"
printf 'complete\n' > "$state/ACTIVATION_CACHE_COMPLETE"

set +e
PYTHONPATH="$repo" "$python_bin" \
  "$runtime/train_qwen38_27b_layer0_mlp_activation_qat_v1.py" \
  > "$state/training.stdout" 2> "$state/training.stderr"
training_status=$?
set -e
if [[ ! -s "$training" ]]; then
  echo "Qwen3.8-27B activation QAT exited without a terminal receipt" >&2
  exit 70
fi
if (( training_status != 0 && training_status != 2 )); then
  echo "Qwen3.8-27B activation QAT had unexpected exit $training_status" >&2
  exit "$training_status"
fi
printf 'terminal\n' > "$state/TRAINING_TERMINAL"

if "$python_bin" - "$training" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
raise SystemExit(0 if value.get("complete") is True and value.get("numeric_promotion_pass") is True else 1)
PY
then
  printf 'pass\n' > "$state/NUMERIC_PROMOTION_PASS"
  PYTHONPATH=/workspace/src/llama.cpp/gguf-py:$repo \
    "$python_bin" "$runtime/materialize_qwen38_27b_layer0_mlp_activation_qat_v1.py" \
    --in-place > "$state/materialization.stdout" 2> "$state/materialization.stderr"
  printf 'complete\n' > "$state/MATERIALIZATION_COMPLETE"

  /usr/bin/time -v /workspace/build/llama.cpp-qwen38-selective-v1/bin/llama-cli \
    -m "$candidate" -t 4 -tb 4 -c 2048 -b 512 -ub 512 \
    --no-warmup --single-turn --simple-io \
    -p 'Reply with only OK.' -n 2 --temp 0 \
    > "$state/load-smoke.log" 2>&1
  grep -Eq 'Exit status:[[:space:]]+0$' "$state/load-smoke.log"
  grep -Eq 'Swaps:[[:space:]]+0$' "$state/load-smoke.log"
  printf 'pass\n' > "$state/LOAD_SMOKE_PASS"

  "$runtime/run_qwen38_27b_layer0_mlp_activation_qat_quality_v1.sh" \
    > "$state/quality.stdout" 2> "$state/quality.stderr"
  printf 'complete\n' > "$state/QUALITY_COMPLETE"
  if [[ -s "$quality/PROMOTION_PASS" ]]; then
    printf 'required\n' > "$state/PAIRED_THROUGHPUT_REQUIRED"
  else
    printf 'quality rejected\n' > "$state/PAIRED_THROUGHPUT_SKIPPED"
  fi
else
  if (( training_status != 2 )); then
    echo "numeric rejection receipt had unexpected process status $training_status" >&2
    exit 70
  fi
  "$python_bin" - "$training" "$state" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
training = Path(sys.argv[1])
state = Path(sys.argv[2])
content = training.read_bytes()
receipt = {
    "schema": "qwen38_27b_activation_qat_numeric_rejection_skip_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "training_receipt": str(training),
    "training_receipt_sha256": hashlib.sha256(content).hexdigest(),
    "terminal": True,
    "valid_nonpromotion": True,
    "materialization_launched": False,
    "quality_launched": False,
    "reason": "the frozen activation-QAT numeric promotion gate rejected",
}
(state / "numeric-rejection-skip.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY
  printf 'numeric rejected\n' > "$state/MATERIALIZATION_AND_QUALITY_SKIPPED"
fi

free -b > "$state/memory-after.txt"
df -B1 /workspace > "$state/disk-after.txt"
"$python_bin" - "$state" "$materialization" "$quality" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
state, materialization, quality = map(Path, sys.argv[1:])
files = {}
for path in sorted(state.iterdir()):
    if path.is_file() and path.name not in {"receipt.json", "WATCH_TERMINAL"}:
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
receipt = {
    "schema": "qwen38_27b_layer0_mlp_activation_qat_watch_receipt_v1",
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "terminal": True,
    "stages": {
        "activation_cache_complete": (state / "ACTIVATION_CACHE_COMPLETE").is_file(),
        "training_terminal": (state / "TRAINING_TERMINAL").is_file(),
        "numeric_promotion_pass": (state / "NUMERIC_PROMOTION_PASS").is_file(),
        "materialization_complete": (state / "MATERIALIZATION_COMPLETE").is_file(),
        "load_smoke_pass": (state / "LOAD_SMOKE_PASS").is_file(),
        "quality_complete": (state / "QUALITY_COMPLETE").is_file(),
        "paired_throughput_required": (state / "PAIRED_THROUGHPUT_REQUIRED").is_file(),
        "paired_throughput_skipped": (state / "PAIRED_THROUGHPUT_SKIPPED").is_file(),
        "materialization_and_quality_skipped": (state / "MATERIALIZATION_AND_QUALITY_SKIPPED").is_file(),
    },
    "materialization_receipt": str(materialization) if materialization.is_file() else None,
    "quality_summary": str(quality / "summary.json") if (quality / "summary.json").is_file() else None,
    "artifacts": files,
    "claim_boundary": "orchestration receipt only; scientific claims remain in the training and capability receipts",
}
(state / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(state / "WATCH_TERMINAL").write_text("terminal\n")
PY
