#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v64}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
llama_build=${LLAMA_BUILD:-/workspace/build/llama.cpp-qwen38-cq3-aff3-v63}
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime/qwen38_27b_trained_compact_model_graph_preregistration_v64.json"
worker="$runtime/run_qwen38_27b_aff3_llamacpp_model_worker_v51.sh"
v62_state=/workspace/quant-state/qwen38-trained-compact-materialize-v62
v63_state=/workspace/quant-state/qwen38-trained-compact-integration-gate-v63
candidate=/workspace/models/candidates/Qwen3.8-27B-trained-CQ3-AFF3-base-v62.gguf
aff3_root=/workspace/models/candidates/qwen38-trained-aff3-v53
state=${STATE_DIR:-/workspace/quant-state/qwen38-trained-compact-model-graph-v64}
receipt="$state/receipt.json"
stdout_path="$state/generation.stdout"
stderr_path="$state/generation.stderr"
marker="$state/EXPERIMENT_COMPLETE.json"
manifest="$state/artifacts.sha256"
cli="$llama_build/bin/llama-cli"
unit=quant-qwen38-27b-trained-compact-model-graph-v64
prompt='Complete this sentence with a short factual answer: The capital of France is'

expected_prereg=a33e3ddf4e70c139ea5e6325b3e55276672dd4955272fc8ddf38efec187db4ce
expected_worker=49975e98cdf7f021fa4b7dab1e984aca0e88e7e0ef2027f4a095007d8ac7c339

require_sha() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

if [[ -n ${QWEN38_RESIDUAL_DIR+x} || -n ${QWEN38_RESIDUAL_LAYERS+x} ]]; then
  echo "v64 refuses all residual runtime variables" >&2
  exit 2
fi
for active in \
  quant-qwen38-27b-trained-compact-materialize-v62.service \
  quant-qwen38-27b-trained-compact-integration-v63.service; do
  if systemctl is-active --quiet "$active"; then
    echo "refusing to overlap model-graph gate with $active" >&2
    exit 2
  fi
done
test -x "$python_bin"
test -x "$worker"
test -x "$cli"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
require_sha "$prereg" "$expected_prereg"
require_sha "$worker" "$expected_worker"
test -f "$v62_state/EXPERIMENT_COMPLETE.json" -a -f "$v62_state/artifacts.sha256"
test -f "$v63_state/EXPERIMENT_COMPLETE.json" -a -f "$v63_state/artifacts.sha256"
(cd / && sha256sum -c "$v62_state/artifacts.sha256")
(cd / && sha256sum -c "$v63_state/artifacts.sha256")
mkdir -p "$state"

if [[ -f "$marker" || -f "$manifest" || -f "$receipt" ]]; then
  test -f "$marker" -a -f "$manifest" -a -f "$receipt"
  (cd / && sha256sum -c "$manifest")
  "$python_bin" - "$receipt" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1]))
if (
    receipt.get("schema") != "qwen38_27b_trained_compact_model_graph_v64"
    or receipt.get("complete") is not True
    or receipt.get("terminal") is not True
):
    raise SystemExit("existing v64 terminal receipt changed")
print(json.dumps(receipt, indent=2, sort_keys=True))
raise SystemExit(0 if receipt.get("passed") is True else 2)
PY
  exit $?
fi

"$python_bin" - \
  "$v62_state/EXPERIMENT_COMPLETE.json" "$v63_state/receipt.json" \
  "$candidate" "$aff3_root" "$llama_build/bin" \
  >"$state/input-authentication.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


v62_path, v63_path, candidate, aff3_root, bin_dir = map(Path, sys.argv[1:])
v62 = json.loads(v62_path.read_text())
v63 = json.loads(v63_path.read_text())
images = sorted(aff3_root.glob("layer-*.aff3x16"))
runtime_artifacts = v63.get("compile", {}).get("runtime_artifacts", {})
if (
    v62.get("schema") != "qwen38_27b_trained_compact_materialization_marker_v62"
    or v62.get("passed") is not True
    or v62.get("terminal") is not True
    or v62.get("candidate_sha256") != sha256(candidate)
    or v62.get("aff3_images") != 64
    or v62.get("residual_payload_bytes") != 0
    or v63.get("schema") != "qwen38_27b_trained_compact_integration_gate_v63"
    or v63.get("passed") is not True
    or v63.get("complete") is not True
    or v63.get("terminal") is not True
    or v63.get("inputs", {}).get("candidate_sha256") != sha256(candidate)
    or v63.get("inputs", {}).get("aff3_layer_count") != 64
    or v63.get("inputs", {}).get("residual_payload_bytes") != 0
    or len(images) != 64
):
    raise SystemExit("v64 parent gate changed")
for name, descriptor in runtime_artifacts.items():
    path = bin_dir / name
    if not path.exists() or sha256(path) != descriptor.get("sha256"):
        raise SystemExit(f"v63 runtime artifact changed: {name}")
payload = {
    "schema": "qwen38_27b_trained_compact_model_graph_input_authentication_v64",
    "v62_marker_sha256": sha256(v62_path),
    "v63_receipt_sha256": sha256(v63_path),
    "candidate_sha256": sha256(candidate),
    "candidate_bytes": candidate.stat().st_size,
    "aff3_images": len(images),
    "aff3_total_bytes": sum(path.stat().st_size for path in images),
    "runtime_artifacts": {name: sha256(bin_dir / name) for name in sorted(runtime_artifacts)},
    "residual_payload_bytes": 0,
    "passed": True,
    "terminal": True,
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY

rm -f -- "$stdout_path.tmp" "$stderr_path.tmp"
install -m 0644 /dev/null "$stdout_path.tmp"
install -m 0644 /dev/null "$stderr_path.tmp"
status=0
started_ns=$(date +%s%N)
systemd-run --quiet --wait \
  --unit="$unit" \
  --property=MemoryMax=30000000000 \
  --property=MemorySwapMax=0 \
  --property=CPUQuota=400% \
  --property=AllowedCPUs=0,2,4,6 \
  --property=Restart=no \
  --property=RuntimeMaxSec=420 \
  --property=LimitFSIZE=1048576 \
  --setenv=LD_LIBRARY_PATH="$llama_build/bin" \
  --setenv=QWEN38_AFF3_DIR="$aff3_root" \
  --setenv=QWEN38_AFF3_LAYERS=0-63 \
  /usr/bin/taskset -c 0,2,4,6 \
  "$worker" "$stdout_path.tmp" "$stderr_path.tmp" \
  "$cli" \
  --model "$candidate" \
  --prompt "$prompt" \
  --n-predict 8 \
  --threads 4 \
  --threads-batch 4 \
  --ctx-size 128 \
  --batch-size 128 \
  --ubatch-size 64 \
  --seed 20260819 \
  --temp 0 \
  --ignore-eos \
  --no-conversation \
  --single-turn \
  --no-display-prompt \
  --simple-io \
  --no-warmup \
  --no-perf || status=$?
ended_ns=$(date +%s%N)
elapsed_ns=$((ended_ns - started_ns))
service_result=$(systemctl show "$unit.service" -p Result --value 2>/dev/null || true)
service_exec_status=$(systemctl show "$unit.service" -p ExecMainStatus --value 2>/dev/null || true)
service_memory_peak=$(systemctl show "$unit.service" -p MemoryPeak --value 2>/dev/null || true)
test -f "$stdout_path.tmp" -a -f "$stderr_path.tmp"
mv "$stdout_path.tmp" "$stdout_path"
mv "$stderr_path.tmp" "$stderr_path"

"$python_bin" - \
  "$receipt" "$prereg" "$repo" "$candidate" "$aff3_root" "$cli" \
  "$v62_state/EXPERIMENT_COMPLETE.json" "$v63_state/receipt.json" \
  "$state/input-authentication.json" "$stdout_path" "$stderr_path" \
  "$status" "$service_result" "$service_exec_status" "$service_memory_peak" \
  "$elapsed_ns" "$prompt" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


(
    receipt_path, prereg, repo, candidate, aff3_root, cli, v62_marker,
    v63_receipt, input_auth, stdout_path, stderr_path, exit_code,
    service_result, service_exec_status, service_memory_peak, elapsed_ns, prompt,
) = sys.argv[1:]
receipt_path = Path(receipt_path)
stdout_path = Path(stdout_path)
stderr_path = Path(stderr_path)
stdout_bytes = stdout_path.read_bytes()
stdout_text = stdout_bytes.decode(errors="replace")
stderr_text = stderr_path.read_text(errors="replace")
performance_markers = (
    "llama_perf_context_print", "eval time =", "prompt eval time =",
    "tokens per second",
)
interactive_markers = ("available commands:", "\n> ", "\n>\n")
performance_timing_ran = any(
    marker in stdout_text or marker in stderr_text for marker in performance_markers
)
interactive_mode_ran = any(marker in stdout_text for marker in interactive_markers)
stream_bounds_respected = (
    len(stdout_bytes) <= 1048576 and stderr_path.stat().st_size <= 1048576
)
service_status = int(service_exec_status) if service_exec_status else -1
passed = (
    int(exit_code) == 0
    and service_result == "success"
    and service_status == 0
    and bool(stdout_bytes.strip())
    and stream_bounds_respected
    and not interactive_mode_ran
    and not performance_timing_ran
)
payload = {
    "schema": "qwen38_27b_trained_compact_model_graph_v64",
    "model": "Qwen/Qwen3.8-27B",
    "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "repository_commit": subprocess.check_output(
        ["git", "-C", repo, "rev-parse", "HEAD"], text=True
    ).strip(),
    "preregistration_sha256": sha256(Path(prereg)),
    "inputs": {
        "candidate_sha256": sha256(Path(candidate)),
        "candidate_bytes": Path(candidate).stat().st_size,
        "aff3_images": len(list(Path(aff3_root).glob("layer-*.aff3x16"))),
        "active_aff3_layers": "0-63",
        "residual_payload_bytes": 0,
        "llama_cli_sha256": sha256(Path(cli)),
        "v62_marker_sha256": sha256(Path(v62_marker)),
        "v63_receipt_sha256": sha256(Path(v63_receipt)),
        "input_authentication_sha256": sha256(Path(input_auth)),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    },
    "execution": {
        "exit_code": int(exit_code),
        "service_result": service_result,
        "service_exec_status": service_status,
        "service_memory_peak_bytes": int(service_memory_peak) if service_memory_peak.isdigit() else None,
        "elapsed_seconds": int(elapsed_ns) / 1e9,
        "requested_generated_tokens": 8,
        "threads": 4,
        "physical_cpu_list": [0, 2, 4, 6],
        "context": 128,
        "batch": 128,
        "ubatch": 64,
        "seed": 20260819,
        "greedy": True,
        "ignore_eos": True,
        "no_warmup": True,
        "no_conversation": True,
        "single_turn": True,
        "runtime_max_seconds": 420,
        "file_size_max_bytes_per_stream": 1048576,
        "stream_bounds_respected": stream_bounds_respected,
        "interactive_mode_ran": interactive_mode_ran,
        "performance_timing_ran": performance_timing_ran,
        "bounded_free_generation_nonempty": bool(stdout_bytes.strip()),
        "actual_all_layer_model_graph_execution_passed": (
            int(exit_code) == 0 and service_result == "success" and service_status == 0
        ),
    },
    "stdout": {
        "bytes": len(stdout_bytes),
        "sha256": sha256(stdout_path),
        "preview": stdout_text[:512],
    },
    "stderr": {
        "bytes": stderr_path.stat().st_size,
        "sha256": sha256(stderr_path),
    },
    "whole_model_quality_ran": False,
    "performance_measurement_ran": False,
    "high_precision_weight_materialization": False,
    "claim_boundary": "Bounded all-64-layer trained CQ3 plus AFF3 graph execution only; semantic correctness, benchmark quality, throughput, memory efficiency, energy efficiency, novelty, and scientific breakthrough remain unmeasured.",
    "passed": passed,
    "complete": True,
    "terminal": True,
}
with tempfile.NamedTemporaryFile(mode="w", dir=receipt_path.parent, prefix=f".{receipt_path.name}.", delete=False) as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    temporary = Path(stream.name)
os.replace(temporary, receipt_path)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

"$python_bin" - "$state" "$manifest" "$marker" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


state, manifest, marker = map(Path, sys.argv[1:])
files = sorted(path for path in state.iterdir() if path.is_file() and path not in {manifest, marker})
with tempfile.NamedTemporaryFile(mode="w", dir=state, prefix=f".{manifest.name}.", delete=False) as stream:
    for path in files:
        stream.write(f"{sha256(path)}  {path.resolve()}\n")
    stream.flush()
    os.fsync(stream.fileno())
    temporary = Path(stream.name)
os.replace(temporary, manifest)
receipt = json.loads((state / "receipt.json").read_text())
payload = {
    "schema": "qwen38_27b_trained_compact_model_graph_marker_v64",
    "executed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "receipt_sha256": sha256(state / "receipt.json"),
    "passed": receipt["passed"],
    "terminal": True,
}
with tempfile.NamedTemporaryFile(mode="w", dir=state, prefix=f".{marker.name}.", delete=False) as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    temporary = Path(stream.name)
os.replace(temporary, marker)
with manifest.open("a") as stream:
    stream.write(f"{sha256(marker)}  {marker.resolve()}\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

(cd / && sha256sum -c "$manifest")
systemctl reset-failed "$unit.service" >/dev/null 2>&1 || true
"$python_bin" - "$receipt" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["passed"] else 2)
PY
