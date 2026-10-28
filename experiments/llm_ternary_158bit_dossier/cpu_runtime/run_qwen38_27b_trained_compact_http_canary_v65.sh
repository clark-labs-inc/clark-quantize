#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v65}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
llama_build=${LLAMA_BUILD:-/workspace/build/llama.cpp-qwen38-cq3-aff3-v63}
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime/qwen38_27b_trained_compact_http_canary_preregistration_v65.json"
v64_evidence="$runtime/receipts/qwen38_27b_trained_compact_model_graph_v64"
v62_state=/workspace/quant-state/qwen38-trained-compact-materialize-v62
v63_state=/workspace/quant-state/qwen38-trained-compact-integration-gate-v63-retry1
candidate=/workspace/models/candidates/Qwen3.8-27B-trained-CQ3-AFF3-base-v62.gguf
aff3_root=/workspace/models/candidates/qwen38-trained-aff3-v53
state=${STATE_DIR:-/workspace/quant-state/qwen38-trained-compact-http-canary-v65}
receipt="$state/receipt.json"
response="$state/response.json"
request="$state/request.json"
properties="$state/server-properties.txt"
manifest="$state/artifacts.sha256"
marker="$state/EXPERIMENT_COMPLETE.json"
server="$llama_build/bin/llama-server"
unit=quant-qwen38-27b-trained-compact-http-canary-v65
port=8105
prompt='Complete this sentence with a short factual answer: The capital of France is'

expected_prereg=0e4c7f22b7a5817685ad856774def1d7a67f296f3aaa082ab2421e5bae233e3f
expected_v64_receipt=4fc9d2f873979d038792d98d2ce5c34fd73c183f2cd34d0c5cd2ed40217c53c3
expected_v64_stdout=f55f184a7e33133711c0a7609768120d1c76c9cd41cddad43eb6c77e1ab90ce0
expected_v64_adjudication=6524a6b58bfd662f8e97d4347be3e344b119c61b9d2dc8aba8ddd59b540f1bf3

require_sha() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

cleanup() {
  systemctl stop "$unit.service" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ -n ${QWEN38_RESIDUAL_DIR+x} || -n ${QWEN38_RESIDUAL_LAYERS+x} ]]; then
  echo "v65 refuses all residual runtime variables" >&2
  exit 2
fi
if pgrep -x llama-server >/dev/null || pgrep -x llama-cli >/dev/null; then
  echo "another llama.cpp model workload is active" >&2
  exit 2
fi
test -x "$python_bin"
test -x "$server"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
mkdir -p "$state"
require_sha "$prereg" "$expected_prereg"
require_sha "$v64_evidence/receipt.json" "$expected_v64_receipt"
require_sha "$v64_evidence/generation.stdout" "$expected_v64_stdout"
require_sha "$v64_evidence/adjudication.json" "$expected_v64_adjudication"
test -f "$v62_state/EXPERIMENT_COMPLETE.json" -a -f "$v62_state/artifacts.sha256"
test -f "$v63_state/EXPERIMENT_COMPLETE.json" -a -f "$v63_state/artifacts.sha256"
(cd / && sha256sum -c "$v62_state/artifacts.sha256")
(cd / && sha256sum -c "$v63_state/artifacts.sha256")

if [[ -f "$marker" || -f "$manifest" || -f "$receipt" ]]; then
  test -f "$marker" -a -f "$manifest" -a -f "$receipt"
  (cd / && sha256sum -c "$manifest")
  "$python_bin" - "$receipt" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if r.get("schema") != "qwen38_27b_trained_compact_http_canary_v65" or not r.get("complete") or not r.get("terminal"):
    raise SystemExit("existing v65 receipt changed")
print(json.dumps(r, indent=2, sort_keys=True))
raise SystemExit(0 if r.get("passed") else 2)
PY
  exit $?
fi

"$python_bin" - \
  "$v62_state/EXPERIMENT_COMPLETE.json" "$v63_state/receipt.json" \
  "$v64_evidence/receipt.json" "$v64_evidence/adjudication.json" \
  "$candidate" "$aff3_root" "$llama_build/bin" \
  >"$state/input-authentication.json" <<'PY'
import hashlib, json, sys
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

v62p, v63p, v64p, adjp, candidate, aff3_root, bin_dir = map(Path, sys.argv[1:])
v62, v63, v64, adj = (json.loads(p.read_text()) for p in (v62p, v63p, v64p, adjp))
candidate_sha = sha256(candidate)
artifacts = v63.get("compile", {}).get("runtime_artifacts", {})
if (
    not v62.get("passed") or not v62.get("terminal")
    or v62.get("candidate_sha256") != candidate_sha
    or v62.get("residual_payload_bytes") != 0
    or not v63.get("passed") or not v63.get("terminal")
    or v63.get("inputs", {}).get("candidate_sha256") != candidate_sha
    or len(v63.get("trained_aff3_graph_parity", [])) != 6
    or v64.get("passed") is not False or not v64.get("terminal")
    or v64.get("execution", {}).get("actual_all_layer_model_graph_execution_passed") is not True
    or adj.get("classification") != "terminal_harness_negative_with_valid_all_layer_graph_execution"
    or len(list(aff3_root.glob("layer-*.aff3x16"))) != 64
):
    raise SystemExit("v65 parent authentication failed")
for name, descriptor in artifacts.items():
    if sha256(bin_dir / name) != descriptor["sha256"]:
        raise SystemExit(f"v63 runtime artifact changed: {name}")
print(json.dumps({
    "schema": "qwen38_27b_trained_compact_http_canary_input_authentication_v65",
    "candidate_sha256": candidate_sha,
    "v62_marker_sha256": sha256(v62p),
    "v63_receipt_sha256": sha256(v63p),
    "v64_receipt_sha256": sha256(v64p),
    "v64_adjudication_sha256": sha256(adjp),
    "aff3_images": 64,
    "runtime_artifacts": {name: sha256(bin_dir / name) for name in sorted(artifacts)},
    "residual_payload_bytes": 0,
    "passed": True,
    "terminal": True,
}, indent=2, sort_keys=True))
PY

systemctl reset-failed "$unit.service" >/dev/null 2>&1 || true
systemd-run --quiet \
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
  "$server" \
  --model "$candidate" \
  --threads 4 --threads-batch 4 \
  --ctx-size 128 --parallel 1 \
  --batch-size 128 --ubatch-size 64 \
  --host 127.0.0.1 --port "$port" \
  --no-warmup --log-disable

ready=false
for _ in $(seq 1 180); do
  if curl --silent --fail "http://127.0.0.1:${port}/health" >/dev/null; then
    ready=true
    break
  fi
  if ! systemctl is-active --quiet "$unit.service"; then
    echo "v65 server exited before health" >&2
    exit 2
  fi
  sleep 1
done
if [[ "$ready" != true ]]; then
  echo "v65 server health timeout" >&2
  exit 2
fi

"$python_bin" - "$request" "$response" "$port" "$prompt" <<'PY'
import json, os, sys, tempfile, urllib.request
from pathlib import Path

request_path, response_path = map(Path, sys.argv[1:3])
port, prompt = sys.argv[3:]
body = {
    "model": "local",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 8,
    "temperature": 0,
    "seed": 20260819,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}
request_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=300) as result:
    if result.status != 200:
        raise SystemExit(f"v65 HTTP status changed: {result.status}")
    payload = result.read(1048577)
if len(payload) > 1048576:
    raise SystemExit("v65 response exceeded frozen byte cap")
parsed = json.loads(payload)
with tempfile.NamedTemporaryFile(mode="w", dir=response_path.parent, prefix=f".{response_path.name}.", delete=False) as stream:
    json.dump(parsed, stream, indent=2, sort_keys=True)
    stream.write("\n")
    temporary = Path(stream.name)
os.replace(temporary, response_path)
PY

systemctl show "$unit.service" \
  -p ActiveState -p SubState -p Result -p MainPID -p MemoryCurrent -p MemoryPeak \
  -p CPUUsageNSec -p AllowedCPUs -p MemoryMax -p MemorySwapMax >"$properties"
cleanup
service_result=$(systemctl show "$unit.service" -p Result --value 2>/dev/null || true)

"$python_bin" - \
  "$receipt" "$prereg" "$repo" "$state/input-authentication.json" \
  "$request" "$response" "$properties" "$service_result" "$prompt" <<'PY'
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

receipt, prereg, repo, auth, request, response, properties, service_result, prompt = sys.argv[1:]
receipt, prereg, auth, request, response, properties = map(Path, (receipt, prereg, auth, request, response, properties))
result = json.loads(response.read_text())
choices = result.get("choices", [])
content = choices[0].get("message", {}).get("content", "") if len(choices) == 1 else ""

def keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).lower()
            yield from keys(child)
    elif isinstance(value, list):
        for child in value: yield from keys(child)

forbidden = [key for key in keys(result) if "timing" in key or "tokens_per_second" in key]
passed = (
    len(choices) == 1
    and bool(content.strip())
    and "paris" in content.casefold()
    and not forbidden
    and response.stat().st_size <= 1048576
    and service_result == "success"
)
payload = {
    "schema": "qwen38_27b_trained_compact_http_canary_v65",
    "model": "Qwen/Qwen3.8-27B",
    "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "repository_commit": subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip(),
    "preregistration_sha256": sha256(prereg),
    "input_authentication_sha256": sha256(auth),
    "request_sha256": sha256(request),
    "response_sha256": sha256(response),
    "server_properties_sha256": sha256(properties),
    "service_result_after_stop": service_result,
    "protocol": {
        "endpoint": "/v1/chat/completions",
        "max_tokens": 8,
        "temperature": 0,
        "seed": 20260819,
        "stream": False,
        "enable_thinking": False,
        "active_aff3_layers": "0-63",
        "residual_payload_bytes": 0,
    },
    "result": {
        "choices": len(choices),
        "assistant_content": content,
        "assistant_content_nonempty": bool(content.strip()),
        "semantic_canary_contains_paris": "paris" in content.casefold(),
        "forbidden_performance_keys": forbidden,
        "response_bytes": response.stat().st_size,
        "stream_bounds_respected": response.stat().st_size <= 1048576,
    },
    "performance_measurement_ran": False,
    "whole_model_quality_suite_ran": False,
    "claim_boundary": "One bounded thinking-disabled HTTP semantic canary only; no frozen benchmark, throughput, memory-efficiency, energy-efficiency, novelty, or breakthrough claim.",
    "passed": passed,
    "complete": True,
    "terminal": True,
}
with tempfile.NamedTemporaryFile(mode="w", dir=receipt.parent, prefix=f".{receipt.name}.", delete=False) as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    temporary = Path(stream.name)
os.replace(temporary, receipt)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

"$python_bin" - "$state" "$manifest" "$marker" <<'PY'
import hashlib, json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
state, manifest, marker = map(Path, sys.argv[1:])
files = sorted(path for path in state.iterdir() if path.is_file() and path not in {manifest, marker})
with tempfile.NamedTemporaryFile(mode="w", dir=state, prefix=f".{manifest.name}.", delete=False) as stream:
    for path in files: stream.write(f"{sha256(path)}  {path.resolve()}\n")
    stream.flush(); os.fsync(stream.fileno()); temporary = Path(stream.name)
os.replace(temporary, manifest)
r = json.loads((state / "receipt.json").read_text())
payload = {
    "schema": "qwen38_27b_trained_compact_http_canary_marker_v65",
    "executed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "receipt_sha256": sha256(state / "receipt.json"),
    "passed": r["passed"],
    "terminal": True,
}
with tempfile.NamedTemporaryFile(mode="w", dir=state, prefix=f".{marker.name}.", delete=False) as stream:
    json.dump(payload, stream, indent=2, sort_keys=True); stream.write("\n"); temporary = Path(stream.name)
os.replace(temporary, marker)
with manifest.open("a") as stream: stream.write(f"{sha256(marker)}  {marker.resolve()}\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

(cd / && sha256sum -c "$manifest")
"$python_bin" - "$receipt" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["passed"] else 2)
PY
