#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/clark-quantize-tq-controls-v1}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
model_dir=${MODEL_DIR:-/workspace/models/Qwen3.8-27B}
controls_dir=${CONTROLS_DIR:-/workspace/models/controls}
llama_source=${LLAMA_SOURCE:-/workspace/src/llama.cpp-qwen38-selective-v1}
llama_build=${LLAMA_BUILD:-/workspace/build/llama.cpp-qwen38-selective-v1}
state=${STATE_DIR:-/workspace/quant-state/qwen38-27b-llamacpp-ternary-controls-v1}
runtime=experiments/llm_ternary_158bit_dossier/cpu_runtime
prereg="$runtime/qwen38_27b_llamacpp_ternary_controls_preregistration_v1.json"

mkdir -p "$state" "$controls_dir"
cd "$repo"
"$python_bin" - "$repo" "$prereg" <<'PY'
import hashlib, json, sys
from pathlib import Path
repo, prereg_path = map(Path, sys.argv[1:])
payload = json.loads(prereg_path.read_text())
assert payload["status"] == "frozen_before_conversion_or_evaluation"
for label, descriptor in payload["frozen_inputs"].items():
    path = repo / descriptor["path"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]:
        raise SystemExit(f"frozen TQ source changed: {label}")
PY
if [[ -n "$(git status --porcelain)" ]]; then
  echo "TQ controls require a clean source checkout" >&2
  exit 73
fi
if [[ $(wc -l < /proc/swaps) -ne 1 ]]; then
  echo "swap is active during TQ controls" >&2
  exit 74
fi
if [[ $(df -Pk /workspace | awk 'NR == 2 {print $4}') -lt $((30 * 1024 * 1024)) ]]; then
  echo "less than 30 GiB disk is free for TQ controls" >&2
  exit 75
fi
if [[ $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) -lt $((12 * 1024 * 1024)) ]]; then
  echo "less than 12 GiB memory is available for TQ controls" >&2
  exit 72
fi
sha256sum "$prereg" "$llama_source/convert_hf_to_gguf.py" \
  "$llama_build/bin/llama-cli" "$llama_build/bin/llama-server" \
  "$llama_build/bin/llama-perplexity" "$llama_build/bin/llama-bench" \
  > "$state/source.sha256"
git rev-parse HEAD > "$state/repository.commit"
date -u +%FT%TZ > "$state/started-at.txt"

for format in TQ1_0 TQ2_0; do
  lower=${format,,}
  final="$controls_dir/Qwen3.8-27B-${format}.gguf"
  partial="$final.partial"
  conversion="$state/${lower}-conversion.json"
  if [[ ! -s "$final" ]]; then
    if [[ -e "$partial" ]]; then
      echo "unmarked partial TQ output requires inspection: $partial" >&2
      exit 76
    fi
    "$python_bin" "$llama_source/convert_hf_to_gguf.py" "$model_dir" \
      --outfile "$partial" --outtype "$lower" --no-nextn \
      > "$state/${lower}-conversion.stdout" \
      2> "$state/${lower}-conversion.stderr"
    "$llama_build/bin/llama-cli" -m "$partial" -t 4 -tb 4 -c 512 \
      -b 512 -ub 512 --no-warmup --single-turn --simple-io \
      -p 'Reply with only OK.' -n 1 --temp 0 \
      > "$state/${lower}-conversion-load-smoke.log" 2>&1
    mv "$partial" "$final"
  fi
  bytes=$(stat -c %s "$final")
  digest=$(sha256sum "$final" | awk '{print $1}')
  "$python_bin" - "$conversion" "$format" "$final" "$bytes" "$digest" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
out = Path(sys.argv[1])
fmt, model, size, digest = sys.argv[2:]
payload = {
    "schema": "qwen38_27b_llamacpp_ternary_conversion_receipt_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "format": fmt,
    "artifact": model,
    "artifact_bytes": int(size),
    "artifact_sha256": digest,
    "effective_file_bits_per_26895998464_parameters": int(size) * 8 / 26895998464,
    "load_smoke_passed": True,
    "complete": True,
    "terminal": True,
}
temporary = out.with_suffix(out.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, out)
PY
  eval_state="/workspace/quant-state/qwen38-27b-${lower}-control-v1"
  eval_run="/workspace/quant-runs/qwen38-27b-${lower}-control-v1"
  port=8110
  if [[ "$format" == "TQ2_0" ]]; then
    port=8111
  fi
  MODEL="$final" FORMAT="$format" MODEL_SHA256="$digest" MODEL_BYTES="$bytes" \
  PORT="$port" \
  RUN_DIR="$eval_run" STATE_DIR="$eval_state" \
  REPO="$repo" PYTHON_BIN="$python_bin" LLAMA_BUILD="$llama_build" \
    "$runtime/eval_qwen38_27b_llamacpp_ternary_control_v1.sh"
done

"$python_bin" - "$state" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
state = Path(sys.argv[1])
controls = {}
for fmt in ("tq1_0", "tq2_0"):
    conversion = state / f"{fmt}-conversion.json"
    quality = Path(f"/workspace/quant-state/qwen38-27b-{fmt}-control-v1/summary.json")
    conversion_content = conversion.read_bytes()
    quality_content = quality.read_bytes()
    controls[fmt.upper()] = {
        "conversion_receipt": str(conversion),
        "conversion_sha256": hashlib.sha256(conversion_content).hexdigest(),
        "quality_receipt": str(quality),
        "quality_sha256": hashlib.sha256(quality_content).hexdigest(),
        "quality": json.loads(quality_content),
    }
payload = {
    "schema": "qwen38_27b_llamacpp_ternary_controls_receipt_v1",
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "Qwen/Qwen3.8-27B",
    "controls": controls,
    "complete": True,
    "terminal": True,
    "claim_boundary": "whole-model upstream ternary controls; descriptive throughput is not an authoritative paired comparison",
}
out = state / "receipt.json"
temporary = state / "receipt.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, out)
(state / "CONTROLS_TERMINAL").write_text("terminal\n")
PY
