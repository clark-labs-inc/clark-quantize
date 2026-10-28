#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v55}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
baseline=${BASELINE_MODEL:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf}
candidate=${CANDIDATE_MODEL:-/workspace/models/candidates/Qwen3.8-27B-Q4_K_M-all-MLP-CQ3_0-template-v55.gguf}
llama_source=${LLAMA_SOURCE:-/workspace/src/llama.cpp-qwen38-selective-v1}
llama_build=${LLAMA_BUILD:-/workspace/build/llama.cpp-qwen38-selective-v1}
run_dir=${RUN_DIR:-/workspace/quant-runs/qwen38-27b-all-mlp-cq3-template-v55}
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-27b-all-mlp-cq3-template-v55}
runtime_dir="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime_dir/qwen38_27b_all_mlp_cq3_template_preregistration_v55.json"
scope_verifier="$runtime_dir/verify_gguf_selective_conversion.py"
quantize="$llama_build/bin/llama-quantize"
test_cq3="$llama_build/bin/test-cq3"
target_regex='^blk\.([0-9]|[1-5][0-9]|6[0-3])\.ffn_(down|gate|up)\.weight$'
partial="${candidate}.partial"

expected_prereg_sha=2a01a2cd5d8365f8ca1e2305c2b1bfd44f3195de4475ccc5d145bb6f619904f4
expected_verifier_sha=3a2a864ad8d93b28154c04bddf2410a2f22c6887c373abea08c84c78cd526cce
expected_baseline_sha=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34
expected_baseline_bytes=18973870432
expected_candidate_bytes=13893519712
expected_llama_revision=4df29be4f4c3673f428170fda944a5b19f743bb8
expected_llama_diff_sha=79294f0fdf173d6909145a5481d866c813a549db40ebe506ed185dd2f78a3364
expected_quantize_sha=30c8e7234e1e3486077f238901df5f83accd49248e928b80586f93e84bb87b7d
expected_test_cq3_sha=2c1dfa7f1e1fca79d7fef2ff16de8315349b16a7346865c39a57a70288e01834

require_sha256() {
  local path=$1
  local expected=$2
  local observed
  observed=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$observed" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, observed $observed" >&2
    exit 2
  fi
}

mkdir -p "$run_dir" "$state_dir" "$(dirname "$candidate")"
if [[ -s "$state_dir/COMPLETE" ]]; then
  (cd / && sha256sum -c "$state_dir/artifacts.sha256")
  echo "authenticated v55 template already complete"
  exit 0
fi

for unit in \
  quant-qwen38-27b-all-layer-mixed-trained-v44.service \
  quant-qwen38-27b-all-layer-sensitivity-residual-v45.service; do
  if systemctl is-active --quiet "$unit"; then
    echo "refusing to overlap v55 template conversion with $unit" >&2
    exit 2
  fi
done
if [[ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]]; then
  echo "v55 requires a clean tracked research worktree" >&2
  exit 2
fi
if [[ -e "$candidate" ]]; then
  if [[ -s "$state_dir/artifacts.sha256" && -s "$state_dir/receipt.json" && -s "$state_dir/gguf-scope.json" ]]; then
    (cd / && sha256sum -c "$state_dir/artifacts.sha256")
    printf 'complete\n' > "$state_dir/COMPLETE"
    echo "recovered authenticated v55 completion marker"
    exit 0
  fi
  echo "existing final template lacks recoverable v55 receipts: $candidate" >&2
  exit 2
fi
if [[ -e "$partial" ]]; then
  test -f "$partial"
  echo "removing only the interrupted v55 partial before restart: $partial"
  rm -f -- "$partial"
fi

test -x "$python_bin"
test -f "$baseline"
test -x "$quantize"
test -x "$test_cq3"
require_sha256 "$prereg" "$expected_prereg_sha"
require_sha256 "$scope_verifier" "$expected_verifier_sha"
require_sha256 "$quantize" "$expected_quantize_sha"
require_sha256 "$test_cq3" "$expected_test_cq3_sha"
if [[ "$(git -C "$llama_source" rev-parse HEAD)" != "$expected_llama_revision" ]]; then
  echo "llama.cpp revision changed" >&2
  exit 2
fi
if [[ "$(git -C "$llama_source" diff --binary | sha256sum | awk '{print $1}')" != "$expected_llama_diff_sha" ]]; then
  echo "selective CQ3 llama.cpp source diff changed" >&2
  exit 2
fi
if [[ "$(stat -c %s "$baseline")" != "$expected_baseline_bytes" ]]; then
  echo "Qwen3.8-27B baseline byte count changed" >&2
  exit 2
fi
require_sha256 "$baseline" "$expected_baseline_sha"

"$python_bin" - "$candidate" <<'PY'
import os
import sys
from pathlib import Path

parent = Path(sys.argv[1]).parent
available = os.statvfs(parent).f_bavail * os.statvfs(parent).f_frsize
required = 18 * 1024**3
if available < required:
    raise SystemExit(f"v55 needs {required} free bytes, observed {available}")
PY

"$test_cq3" > "$run_dir/test-cq3.log" 2>&1
cleanup_partial() {
  status=$?
  if [[ $status -ne 0 && -f "$partial" && ! -e "$candidate" ]]; then
    rm -f -- "$partial"
  fi
  exit "$status"
}
trap cleanup_partial EXIT

git -C "$repo" rev-parse HEAD > "$state_dir/source.commit"
/usr/bin/time -v "$quantize" \
  --allow-requantize \
  --tensor-type "${target_regex}=CQ3_0" \
  "$baseline" "$partial" COPY 4 \
  > "$run_dir/conversion.log" 2>&1
sync "$partial"
if [[ "$(stat -c %s "$partial")" != "$expected_candidate_bytes" ]]; then
  echo "v55 structural template byte count changed" >&2
  exit 2
fi

"$python_bin" - "$run_dir/conversion.log" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
actual = set(re.findall(
    r"(blk\.(?:[0-9]|[1-5][0-9]|6[0-3])\.ffn_(?:down|gate|up)\.weight)\s+- applying selective COPY override",
    text,
))
expected = {
    f"blk.{layer}.ffn_{kind}.weight"
    for layer in range(64)
    for kind in ("down", "gate", "up")
}
if actual != expected or text.count("applying selective COPY override") != 192:
    raise SystemExit(
        f"v55 override denominator changed: observed={len(actual)} expected=192"
    )
PY

PYTHONPATH="$llama_source/gguf-py" "$python_bin" "$scope_verifier" \
  --baseline "$baseline" \
  --candidate "$partial" \
  --target-regex "$target_regex" \
  --expected-target-count 192 \
  --baseline-target-type 12 \
  --candidate-target-type 43 \
  --out "$state_dir/gguf-scope.json" \
  > "$run_dir/gguf-scope.log" 2>&1

"$python_bin" - "$state_dir/gguf-scope.json" "$candidate" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

scope_path = Path(sys.argv[1])
candidate = Path(sys.argv[2]).resolve()
scope = json.loads(scope_path.read_text())
scope.pop("receipt_content_sha256", None)
scope["models"]["candidate"]["path"] = str(candidate)
payload = json.dumps(
    scope, sort_keys=True, separators=(",", ":"), ensure_ascii=False
).encode()
scope["receipt_content_sha256"] = hashlib.sha256(payload).hexdigest()
temporary = scope_path.with_suffix(scope_path.suffix + ".normalized.tmp")
temporary.write_text(json.dumps(scope, indent=2, sort_keys=True) + "\n")
os.replace(temporary, scope_path)
PY
"$python_bin" - "$prereg" "$state_dir/gguf-scope.json" "$state_dir" "$partial" "$candidate" <<'PY'
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

prereg_path, scope_path, state, candidate_file, candidate_path = map(
    Path, sys.argv[1:]
)
scope = json.loads(scope_path.read_text())
if (
    scope.get("valid") is not True
    or scope.get("target_count") != 192
    or scope.get("non_target_count") != 659
    or scope.get("models", {}).get("candidate", {}).get("bytes") != 13893519712
):
    raise SystemExit("v55 scope receipt failed the frozen structural gate")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

receipt = {
    "schema": "qwen38_27b_all_mlp_cq3_template_receipt_v55",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "model": "Qwen/Qwen3.8-27B",
    "repository_commit": (state / "source.commit").read_text().strip(),
    "preregistration": {
        "path": str(prereg_path),
        "sha256": sha256(prereg_path),
    },
    "candidate": scope["models"]["candidate"],
    "scope_receipt": {
        "path": str(scope_path),
        "sha256": sha256(scope_path),
    },
    "host": {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    },
    "checks": {
        "scope_valid": True,
        "exact_target_count": scope["target_count"] == 192,
        "exact_non_target_count": scope["non_target_count"] == 659,
        "exact_candidate_bytes": candidate_file.stat().st_size == 13893519712,
        "all_non_targets_identical": scope["checks"][
            "all_non_target_types_sizes_and_payloads_identical"
        ],
        "all_target_types_valid": scope["checks"]["candidate_target_types_valid"],
        "all_target_payloads_changed": scope["checks"]["all_target_payloads_changed"],
    },
    "claim_boundary": "authenticated 192-slot structural GGUF only; trained payload quality and runtime performance remain unmeasured",
}
receipt["valid"] = all(receipt["checks"].values())
if not receipt["valid"]:
    raise SystemExit("v55 terminal receipt is invalid")
receipt_path = state / "receipt.json"
temporary = state / ".receipt.json.tmp"
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
os.replace(temporary, receipt_path)

artifacts = state / "artifacts.sha256"
artifacts.write_text(
    f'{scope["models"]["candidate"]["sha256"]}  {candidate_path}\n'
    f'{sha256(scope_path)}  {scope_path}\n'
    f'{sha256(receipt_path)}  {receipt_path}\n'
)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

mv "$partial" "$candidate"
trap - EXIT
(cd / && sha256sum -c "$state_dir/artifacts.sha256")
printf 'complete\n' > "$state_dir/COMPLETE"
