#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v62}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
llama_source=${LLAMA_SOURCE:-/workspace/src/llama.cpp-qwen38-selective-v1}
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime/qwen38_27b_trained_compact_x8_runtime_preregistration_v62.json"
training_receipt=/workspace/quant-state/qwen38-all-layer-mixed-trained-v44/receipt.json
training_root=/workspace/quant-runs/qwen38-all-layer-mixed-trained-v44
direct_receipt=/workspace/quant-state/qwen38-all-layer-mixed-direct-v43/receipt.json
direct_root=/workspace/quant-runs/qwen38-all-layer-mixed-direct-v43
template=/workspace/models/candidates/Qwen3.8-27B-Q4_K_M-all-MLP-CQ3_0-template-v55.gguf
template_scope=/workspace/quant-state/qwen38-27b-all-mlp-cq3-template-v55/gguf-scope.json
template_manifest=/workspace/quant-state/qwen38-27b-all-mlp-cq3-template-v55/artifacts.sha256
candidate=/workspace/models/candidates/Qwen3.8-27B-trained-CQ3-AFF3-base-v62.gguf
aff3_root=/workspace/models/candidates/qwen38-trained-aff3-v53
state=/workspace/quant-state/qwen38-trained-compact-materialize-v62
cq3_receipt="$state/cq3-materialization.json"
aff3_receipt="$state/aff3-materialization.json"
marker="$state/EXPERIMENT_COMPLETE.json"
manifest="$state/artifacts.sha256"

expected_prereg=146a554da68fb3bec2f24a0c2230a6e282511af32f84cf6a0431652fd044674b
expected_v46=ee09826e009a6ec00e50e618f67cbf88cc750b1663dbe067c0295fedd5fd8482
expected_v53=348429e8dad39e7635d2c917e641861afce36395fe917760fde2fd7ab98ab0bc
expected_training=1f33a04cf6d5433724e5d37d12fa0bfa3edfd710ab365033a8db69b517f73c1f
expected_direct=3a3179b1801b024cc1b2692897c450cb0443e8b844d1878dca01aea338280e5b

require_sha() {
  local path=$1
  local expected=$2
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}

if [[ -n ${QWEN38_RESIDUAL_DIR+x} || -n ${QWEN38_RESIDUAL_LAYERS+x} ]]; then
  echo "v62 residual-free candidate refuses residual environment variables" >&2
  exit 2
fi
test -x "$python_bin"
test -d "$llama_source/gguf-py/gguf"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
mkdir -p "$state" "$(dirname "$candidate")" "$aff3_root"
require_sha "$prereg" "$expected_prereg"
require_sha "$runtime/materialize_qwen38_27b_all_layer_trained_cq3_v46.py" "$expected_v46"
require_sha "$runtime/materialize_qwen38_27b_all_layer_trained_aff3_v53.py" "$expected_v53"
require_sha "$training_receipt" "$expected_training"
require_sha "$direct_receipt" "$expected_direct"
test -f /workspace/quant-state/qwen38-27b-all-mlp-cq3-template-v55/COMPLETE
(cd / && sha256sum -c "$template_manifest")

if [[ -f "$marker" || -f "$manifest" ]]; then
  test -f "$marker" -a -f "$manifest"
  (cd / && sha256sum -c "$manifest")
  "$python_bin" - "$marker" <<'PY'
import json, sys
marker = json.load(open(sys.argv[1]))
if marker.get("schema") != "qwen38_27b_trained_compact_materialization_marker_v62" or not marker.get("passed") or not marker.get("terminal"):
    raise SystemExit("v62 terminal marker changed")
PY
  echo "authenticated v62 materialization already complete"
  exit 0
fi

available=$(df -B1 --output=avail "$(dirname "$candidate")" | tail -1)
if (( available < 30000000000 )); then
  echo "v62 materialization requires at least 30 GB free" >&2
  exit 2
fi

if [[ -e "$candidate" || -e "$cq3_receipt" ]]; then
  test -f "$candidate" -a -f "$cq3_receipt"
  "$python_bin" - "$candidate" "$cq3_receipt" <<'PY'
import hashlib, json, sys
from pathlib import Path
candidate, receipt_path = map(Path, sys.argv[1:])
receipt = json.loads(receipt_path.read_text())
digest_state = hashlib.sha256()
with candidate.open("rb") as stream:
    for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        digest_state.update(block)
digest = digest_state.hexdigest()
if receipt.get("schema") != "qwen38_27b_all_layer_trained_cq3_gguf_materialization_v46" or receipt.get("valid") is not True or receipt.get("candidate", {}).get("sha256") != digest:
    raise SystemExit("existing v62 CQ3 materialization changed")
PY
else
  PYTHONPATH="$llama_source/gguf-py" "$python_bin" \
    "$runtime/materialize_qwen38_27b_all_layer_trained_cq3_v46.py" \
    --template "$template" \
    --template-scope-receipt "$template_scope" \
    --training-receipt "$training_receipt" \
    --artifact-root "$training_root" \
    --output "$candidate" \
    --out "$cq3_receipt"
fi

PYTHONPATH="$repo:$llama_source/gguf-py" "$python_bin" \
  "$runtime/materialize_qwen38_27b_all_layer_trained_aff3_v53.py" \
  --training-receipt "$training_receipt" \
  --direct-receipt "$direct_receipt" \
  --trained-artifact-root "$training_root" \
  --direct-artifact-root "$direct_root" \
  --output-root "$aff3_root" \
  --receipt "$aff3_receipt"

"$python_bin" - "$prereg" "$candidate" "$cq3_receipt" "$aff3_root" "$aff3_receipt" "$manifest" "$marker" <<'PY'
import hashlib, json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

prereg, candidate, cq3_receipt, aff3_root, aff3_receipt, manifest, marker = map(Path, sys.argv[1:])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

cq3 = json.loads(cq3_receipt.read_text())
aff3 = json.loads(aff3_receipt.read_text())
images = sorted(aff3_root.glob("layer-*.aff3x16"))
if (
    cq3.get("schema") != "qwen38_27b_all_layer_trained_cq3_gguf_materialization_v46"
    or cq3.get("valid") is not True
    or cq3.get("candidate", {}).get("sha256") != sha256(candidate)
    or aff3.get("schema") != "qwen38_27b_all_layer_trained_aff3_runtime_v53"
    or aff3.get("complete") is not True
    or aff3.get("terminal") is not True
    or aff3.get("layer_count") != 64
    or aff3.get("all_images_authenticated") is not True
    or len(images) != 64
):
    raise SystemExit("v62 materialization terminal gate failed")
manifest_payload = "".join(
    f"{sha256(path)}  {path.resolve()}\n"
    for path in (candidate, cq3_receipt, aff3_receipt, *images)
)
with tempfile.NamedTemporaryFile(mode="w", dir=manifest.parent, prefix=f".{manifest.name}.", delete=False) as stream:
    stream.write(manifest_payload)
    stream.flush()
    os.fsync(stream.fileno())
    manifest_temporary = Path(stream.name)
os.replace(manifest_temporary, manifest)
payload = {
    "schema": "qwen38_27b_trained_compact_materialization_marker_v62",
    "executed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "preregistration_sha256": sha256(prereg),
    "candidate_sha256": sha256(candidate),
    "cq3_receipt_sha256": sha256(cq3_receipt),
    "aff3_receipt_sha256": sha256(aff3_receipt),
    "aff3_images": len(images),
    "residual_payload_bytes": 0,
    "passed": True,
    "terminal": True,
}
with tempfile.NamedTemporaryFile(mode="w", dir=marker.parent, prefix=f".{marker.name}.", delete=False) as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    temporary = Path(stream.name)
os.replace(temporary, marker)
print(json.dumps({"event": "v62_materialization_terminal", "candidate_sha256": payload["candidate_sha256"], "aff3_images": len(images), "terminal": True}, sort_keys=True))
PY

(cd / && sha256sum -c "$manifest")
