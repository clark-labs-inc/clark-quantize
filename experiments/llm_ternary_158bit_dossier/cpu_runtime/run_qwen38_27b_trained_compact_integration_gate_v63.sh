#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v63}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
llama_source=${LLAMA_SOURCE:-/workspace/src/llama.cpp-qwen38-cq3-aff3-v54-script}
llama_build=${LLAMA_BUILD:-/workspace/build/llama.cpp-qwen38-cq3-aff3-v63}
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime/qwen38_27b_trained_compact_integration_gate_preregistration_v63.json"
v54_transform="$runtime/receipts/qwen38_27b_cq3_aff3_llamacpp_v54-transform.json"
v62_state=/workspace/quant-state/qwen38-trained-compact-materialize-v62
v62_marker="$v62_state/EXPERIMENT_COMPLETE.json"
v62_manifest="$v62_state/artifacts.sha256"
candidate=/workspace/models/candidates/Qwen3.8-27B-trained-CQ3-AFF3-base-v62.gguf
aff3_root=/workspace/models/candidates/qwen38-trained-aff3-v53
cq3_materialization="$v62_state/cq3-materialization.json"
aff3_materialization="$v62_state/aff3-materialization.json"
state=${STATE_DIR:-/workspace/quant-state/qwen38-trained-compact-integration-gate-v63}
receipt="$state/receipt.json"
marker="$state/EXPERIMENT_COMPLETE.json"
manifest="$state/artifacts.sha256"
standalone="$llama_build/bin/qwen38-aff3-runtime-smoke-v63"

expected_prereg=7797caa49c8553e545edf7a3b1990bb4935818792868d765046dd99e69132125
expected_v62_prereg=146a554da68fb3bec2f24a0c2230a6e282511af32f84cf6a0431652fd044674b
expected_v54_transform=6f2fc0d2877e430926c5dce039e44e59a6acb6994217f43a6a3e70773d739401
expected_llama_revision=4df29be4f4c3673f428170fda944a5b19f743bb8
expected_llama_diff=f76b42067337f35d8f5721b92faffcff90f4e1cd15afe8e4aa22ec4fdf67eb4a

require_sha() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

if [[ -n ${QWEN38_RESIDUAL_DIR+x} || -n ${QWEN38_RESIDUAL_LAYERS+x} ]]; then
  echo "v63 refuses all residual runtime variables" >&2
  exit 2
fi
if systemctl is-active --quiet quant-qwen38-27b-trained-compact-materialize-v62.service; then
  echo "v62 materialization is still active" >&2
  exit 2
fi
test -x "$python_bin"
test -d "$llama_source/.git" -o -f "$llama_source/.git"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
mkdir -p "$state" "$llama_build"
require_sha "$prereg" "$expected_prereg"
require_sha "$runtime/qwen38_27b_trained_compact_x8_runtime_preregistration_v62.json" "$expected_v62_prereg"
require_sha "$v54_transform" "$expected_v54_transform"
test "$(git -C "$llama_source" rev-parse HEAD)" = "$expected_llama_revision"
test "$(git -C "$llama_source" diff --binary | sha256sum | awk '{print $1}')" = "$expected_llama_diff"
test -f "$v62_marker" -a -f "$v62_manifest"
(cd / && sha256sum -c "$v62_manifest")

if [[ -f "$marker" || -f "$manifest" || -f "$receipt" ]]; then
  test -f "$marker" -a -f "$manifest" -a -f "$receipt"
  (cd / && sha256sum -c "$manifest")
  "$python_bin" - "$receipt" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1]))
if (
    receipt.get("schema") != "qwen38_27b_trained_compact_integration_gate_v63"
    or receipt.get("passed") is not True
    or receipt.get("complete") is not True
    or receipt.get("terminal") is not True
):
    raise SystemExit("existing v63 terminal receipt changed")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
  exit 0
fi

available=$(df -B1 --output=avail "$state" | tail -1)
if (( available < 8000000000 )); then
  echo "v63 compile gate requires at least 8 GB free" >&2
  exit 2
fi

"$python_bin" - \
  "$v54_transform" "$llama_source" "$v62_marker" "$candidate" \
  "$cq3_materialization" "$aff3_materialization" "$aff3_root" \
  >"$state/source-and-artifact-authentication.json" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


transform_path, source, marker_path, candidate, cq3_path, aff3_path, aff3_root = map(Path, sys.argv[1:])
transform = json.loads(transform_path.read_text())
marker = json.loads(marker_path.read_text())
cq3 = json.loads(cq3_path.read_text())
aff3 = json.loads(aff3_path.read_text())
status = subprocess.check_output(
    ["git", "-C", str(source), "status", "--porcelain"], text=True
)
observed_paths = set()
for line in status.splitlines():
    relative = line[3:]
    path = source / relative
    if path.is_dir():
        observed_paths.update(
            str(item.relative_to(source)) for item in path.rglob("*") if item.is_file()
        )
    elif path.is_file():
        observed_paths.add(relative)
expected_files = transform.get("working_tree_files_sha256", {})
if observed_paths != set(expected_files):
    raise SystemExit("authenticated v54 source file denominator changed")
for relative, expected in expected_files.items():
    if sha256(source / relative) != expected:
        raise SystemExit(f"authenticated v54 source changed: {relative}")
images = sorted(aff3_root.glob("layer-*.aff3x16"))
if (
    marker.get("schema") != "qwen38_27b_trained_compact_materialization_marker_v62"
    or marker.get("passed") is not True
    or marker.get("terminal") is not True
    or marker.get("residual_payload_bytes") != 0
    or marker.get("aff3_images") != 64
    or marker.get("candidate_sha256") != sha256(candidate)
    or cq3.get("valid") is not True
    or len(cq3.get("targets", {})) != 192
    or not all(cq3.get("checks", {}).values())
    or cq3.get("candidate", {}).get("sha256") != sha256(candidate)
    or aff3.get("complete") is not True
    or aff3.get("terminal") is not True
    or aff3.get("all_images_authenticated") is not True
    or aff3.get("layer_count") != 64
    or aff3.get("persistent_dense_weight_bytes") != 0
    or len(images) != 64
):
    raise SystemExit("v62 materialized artifact gate changed")
payload = {
    "schema": "qwen38_27b_trained_compact_v63_source_artifact_authentication",
    "llama_revision": subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip(),
    "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(
        ["git", "-C", str(source), "diff", "--binary"]
    )).hexdigest(),
    "authenticated_working_tree_files": len(expected_files),
    "candidate_sha256": sha256(candidate),
    "candidate_bytes": candidate.stat().st_size,
    "cq3_target_tensors": len(cq3["targets"]),
    "aff3_images": len(images),
    "aff3_total_bytes": sum(path.stat().st_size for path in images),
    "residual_payload_bytes": 0,
    "passed": True,
    "terminal": True,
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY

: >"$state/build.log"
cmake -S "$llama_source" -B "$llama_build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=ON \
  2>&1 | tee -a "$state/build.log"
cmake --build "$llama_build" --parallel 4 \
  --target test-cq3 llama-cli llama-server llama-perplexity \
  2>&1 | tee -a "$state/build.log"
"$repo/scripts/cpu/build_qwen38_aff3_llamacpp_runtime_smoke_v50.sh" \
  "$llama_source" "$llama_build" "$standalone" \
  2>&1 | tee -a "$state/build.log"

env \
  -u CLARK_QWEN38_27B_CQ3_PAIR_BENCH \
  -u CLARK_QWEN38_27B_CQ3_X8_BENCH \
  -u QWEN38_AFF3_DIR \
  -u QWEN38_AFF3_LAYERS \
  -u QWEN38_RESIDUAL_DIR \
  -u QWEN38_RESIDUAL_LAYERS \
  LD_LIBRARY_PATH="$llama_build/bin" \
  taskset -c 0,2,4,6 "$llama_build/bin/test-cq3" \
  >"$state/test-cq3.stdout" 2>"$state/test-cq3.stderr"

for layer in 0 32 63; do
  for tokens in 1 8; do
    stem=$(printf 'aff3-layer-%02d-tokens-%d' "$layer" "$tokens")
    env \
      -u QWEN38_RESIDUAL_DIR \
      -u QWEN38_RESIDUAL_LAYERS \
      LD_LIBRARY_PATH="$llama_build/bin" \
      QWEN38_AFF3_DIR="$aff3_root" \
      QWEN38_AFF3_LAYERS="$layer" \
      taskset -c 0,2,4,6 "$standalone" \
      --tokens "$tokens" --layer "$layer" --validate-only \
      >"$state/$stem.json" 2>"$state/$stem.stderr"
  done
done

{
  uname -a
  lscpu
  cmake --version
  g++ --version
} >"$state/environment.txt"

"$python_bin" - \
  "$receipt" "$prereg" "$repo" "$llama_source" "$llama_build" \
  "$candidate" "$aff3_root" "$v62_marker" "$state" <<'PY'
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


receipt_path, prereg, repo, source, build, candidate, aff3_root, v62_marker, state = map(Path, sys.argv[1:])
validations = []
for layer in (0, 32, 63):
    for tokens in (1, 8):
        path = state / f"aff3-layer-{layer:02d}-tokens-{tokens}.json"
        payload = json.loads(path.read_text())
        if (
            payload.get("schema") != "qwen38_27b_aff3_llamacpp_runtime_smoke_v49"
            or payload.get("model") != "Qwen/Qwen3.8-27B"
            or payload.get("layer") != layer
            or payload.get("tokens") != tokens
            or payload.get("observed_task_mask") != 15
            or payload.get("output_bit_exact_vs_direct_v47") is not True
            or payload.get("maximum_absolute_error_vs_direct_v47") != 0
            or payload.get("persistent_dense_weight_bytes") != 0
            or payload.get("decoded_group_scale_matrix_bytes") != 0
            or payload.get("performance_timing_ran") is not False
            or payload.get("passed") is not True
            or payload.get("complete") is not True
            or payload.get("terminal") is not True
        ):
            raise SystemExit(f"AFF3 v63 parity failed for layer {layer}, tokens {tokens}")
        validations.append({
            "layer": layer,
            "tokens": tokens,
            "receipt_sha256": sha256(path),
            "stderr_sha256": sha256(state / f"aff3-layer-{layer:02d}-tokens-{tokens}.stderr"),
            "bit_exact": True,
            "task_mask": 15,
        })
binary_names = (
    "test-cq3", "llama-cli", "llama-server", "llama-perplexity",
    "qwen38-aff3-runtime-smoke-v63", "libllama.so", "libggml-cpu.so",
    "libggml-base.so", "libggml.so",
)
binaries = {}
for name in binary_names:
    path = build / "bin" / name
    if not path.exists():
        raise SystemExit(f"required v63 runtime artifact missing: {name}")
    binaries[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
payload = {
    "schema": "qwen38_27b_trained_compact_integration_gate_v63",
    "model": "Qwen/Qwen3.8-27B",
    "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "repository_commit": subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip(),
    "preregistration_sha256": sha256(prereg),
    "source": {
        "llama_revision": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "-C", str(source), "diff", "--binary"]
        )).hexdigest(),
        "authentication_receipt_sha256": sha256(state / "source-and-artifact-authentication.json"),
    },
    "inputs": {
        "v62_terminal_marker_sha256": sha256(v62_marker),
        "candidate_sha256": sha256(candidate),
        "candidate_bytes": candidate.stat().st_size,
        "aff3_layer_count": len(list(aff3_root.glob("layer-*.aff3x16"))),
        "residual_payload_bytes": 0,
    },
    "compile": {
        "generator": "Ninja",
        "build_type": "Release",
        "parallel_jobs": 4,
        "ggml_native": True,
        "ggml_openmp": True,
        "build_log_sha256": sha256(state / "build.log"),
        "runtime_artifacts": binaries,
    },
    "test_cq3": {
        "exit_code": 0,
        "benchmark_environment_present": False,
        "stdout_sha256": sha256(state / "test-cq3.stdout"),
        "stderr_sha256": sha256(state / "test-cq3.stderr"),
        "passed": True,
    },
    "trained_aff3_graph_parity": validations,
    "whole_model_generation_ran": False,
    "performance_timing_ran": False,
    "high_precision_weight_materialization": False,
    "claim_boundary": "Exact compiled CQ3 primitive and representative trained AFF3 direct-versus-GGML graph parity only; whole-model generation, quality, throughput, memory, energy, novelty, and scientific breakthrough remain unmeasured.",
    "passed": True,
    "complete": True,
    "terminal": True,
}
receipt_path.parent.mkdir(parents=True, exist_ok=True)
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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


state, manifest, marker = map(Path, sys.argv[1:])
files = sorted(
    path for path in state.iterdir()
    if path.is_file() and path not in {manifest, marker}
)
manifest_payload = "".join(f"{sha256(path)}  {path.resolve()}\n" for path in files)
with tempfile.NamedTemporaryFile(mode="w", dir=state, prefix=f".{manifest.name}.", delete=False) as stream:
    stream.write(manifest_payload)
    stream.flush()
    os.fsync(stream.fileno())
    temporary = Path(stream.name)
os.replace(temporary, manifest)
receipt = state / "receipt.json"
payload = {
    "schema": "qwen38_27b_trained_compact_integration_gate_marker_v63",
    "executed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "receipt_sha256": sha256(receipt),
    "artifact_count": len(files),
    "passed": True,
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
