#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v66}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
llama_build=${LLAMA_BUILD:-/workspace/build/llama.cpp-qwen38-cq3-aff3-v63}
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime/qwen38_27b_trained_compact_factorial_diagnosis_preregistration_v66.json"
v65_receipt="$runtime/receipts/qwen38_27b_trained_compact_http_canary_v65/receipt.json"
cute_manifest="$runtime/cute_quick_v1.json"
cute_evaluator="$runtime/eval_llama_cpp_cute_quick.py"
cq3_model=/workspace/models/candidates/Qwen3.8-27B-trained-CQ3-AFF3-base-v62.gguf
q4_model=/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf
aff3_root=/workspace/models/candidates/qwen38-trained-aff3-v53
v63_receipt=/workspace/quant-state/qwen38-trained-compact-integration-gate-v63-retry1/receipt.json
state=${STATE_DIR:-/workspace/quant-state/qwen38-trained-compact-factorial-diagnosis-v66}
run=${RUN_DIR:-/workspace/quant-runs/qwen38-trained-compact-factorial-diagnosis-v66}
summary="$state/summary.json"
manifest="$state/artifacts.sha256"
marker="$state/EXPERIMENT_COMPLETE.json"
server="$llama_build/bin/llama-server"
port=8106
current_unit=
prompt='Complete this sentence with a short factual answer: The capital of France is'

expected_prereg=944cfe3b29b568bd8a7bee5702bd04f30d4e450c1333221c9c32072e9d993a58
expected_v65=96ddd46b37ad135193d11774fa214ee0430e9b81e15e6f1f29c28318a1284b9f
expected_cute_manifest=19dd73eedf737d11f7c94d039822de62314abf31e3fb739a9a875fe65c5bbd16
expected_cute_evaluator=9c40a48a0c3f0544156a20ccccdc12f7e90bbae8afd391bf640ead1ff234c8ab
expected_cq3=506505935f82bade12ab4b51bd1ca9b6e1b5cc11eeae99a6e230b9a7b9044ffc
expected_q4=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34

require_sha() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

stop_server() {
  if [[ -n "$current_unit" ]]; then
    systemctl stop "$current_unit.service" >/dev/null 2>&1 || true
    current_unit=
  fi
}
trap stop_server EXIT

start_server() {
  local arm=$1 model=$2 aff3=$3
  current_unit="quant-qwen38-27b-v66-${arm//_/-}"
  systemctl reset-failed "$current_unit.service" >/dev/null 2>&1 || true
  local env_args=(--setenv=LD_LIBRARY_PATH="$llama_build/bin")
  if [[ "$aff3" = all ]]; then
    env_args+=(--setenv=QWEN38_AFF3_DIR="$aff3_root" --setenv=QWEN38_AFF3_LAYERS=0-63)
  fi
  systemd-run --quiet \
    --unit="$current_unit" \
    --property=MemoryMax=30000000000 \
    --property=MemorySwapMax=0 \
    --property=CPUQuota=400% \
    --property=AllowedCPUs=0,2,4,6 \
    --property=Restart=no \
    --property=RuntimeMaxSec=1800 \
    "${env_args[@]}" \
    /usr/bin/taskset -c 0,2,4,6 \
    "$server" --model "$model" \
    --threads 4 --threads-batch 4 \
    --ctx-size 4096 --parallel 1 \
    --batch-size 128 --ubatch-size 64 \
    --host 127.0.0.1 --port "$port" \
    --no-warmup --log-disable
  for _ in $(seq 1 180); do
    if curl --silent --fail "http://127.0.0.1:${port}/health" >/dev/null; then
      return 0
    fi
    if ! systemctl is-active --quiet "$current_unit.service"; then
      echo "$current_unit exited before health" >&2
      return 2
    fi
    sleep 1
  done
  echo "$current_unit health timeout" >&2
  return 2
}

run_canary() {
  local arm=$1 model_sha=$2 out="$state/$1-canary.json"
  [[ -s "$out" ]] && return 0
  "$python_bin" - "$out" "$port" "$prompt" "$arm" "$model_sha" <<'PY'
import json, os, sys, tempfile, urllib.request
from pathlib import Path
out = Path(sys.argv[1]); port, prompt, arm, model_sha = sys.argv[2:]
body = {
    "model": "local",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 8,
    "temperature": 0,
    "seed": 20260819,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST",
)
with urllib.request.urlopen(request, timeout=300) as response:
    result = json.load(response)
choices = result.get("choices", [])
content = choices[0].get("message", {}).get("content", "") if len(choices) == 1 else ""
payload = {
    "schema": "qwen38_27b_v66_france_canary",
    "arm": arm,
    "model_sha256": model_sha,
    "assistant_content": content,
    "contains_paris": "paris" in content.casefold(),
    "choices": len(choices),
    "enable_thinking": False,
    "max_tokens": 8,
    "complete": True,
    "terminal": True,
}
with tempfile.NamedTemporaryFile(mode="w", dir=out.parent, prefix=f".{out.name}.", delete=False) as stream:
    json.dump(payload, stream, indent=2, sort_keys=True); stream.write("\n"); temporary = Path(stream.name)
os.replace(temporary, out)
print(json.dumps(payload, sort_keys=True), flush=True)
PY
}

if [[ -n ${QWEN38_RESIDUAL_DIR+x} || -n ${QWEN38_RESIDUAL_LAYERS+x} ]]; then
  echo "v66 refuses residual environment variables" >&2
  exit 2
fi
if pgrep -x llama-server >/dev/null || pgrep -x llama-cli >/dev/null; then
  echo "another llama.cpp model workload is active" >&2
  exit 2
fi
test -x "$python_bin" -a -x "$server"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
mkdir -p "$state" "$run"
require_sha "$prereg" "$expected_prereg"
require_sha "$v65_receipt" "$expected_v65"
require_sha "$cute_manifest" "$expected_cute_manifest"
require_sha "$cute_evaluator" "$expected_cute_evaluator"
require_sha "$cq3_model" "$expected_cq3"
require_sha "$q4_model" "$expected_q4"

if [[ -f "$marker" || -f "$manifest" || -f "$summary" ]]; then
  test -f "$marker" -a -f "$manifest" -a -f "$summary"
  (cd / && sha256sum -c "$manifest")
  "$python_bin" - "$summary" <<'PY'
import json, sys
r=json.load(open(sys.argv[1])); assert r["complete"] and r["terminal"]
print(json.dumps(r, indent=2, sort_keys=True))
PY
  exit 0
fi

for spec in \
  "cq3_mlp_q4_attention|$cq3_model|$expected_cq3|none" \
  "q4_mlp_aff3_attention|$q4_model|$expected_q4|all"; do
  IFS='|' read -r arm model model_sha aff3 <<<"$spec"
  canary="$state/$arm-canary.json"
  cute="$state/$arm-cute.json"
  if [[ ! -s "$canary" || ! -s "$cute" ]]; then
    start_server "$arm" "$model" "$aff3"
    run_canary "$arm" "$model_sha"
    if [[ ! -s "$cute" ]]; then
      "$python_bin" "$cute_evaluator" \
        --manifest "$cute_manifest" \
        --server "http://127.0.0.1:${port}" \
        --model-label "qwen38-v66-$arm" \
        --model-sha256 "$model_sha" \
        --out "$cute" \
        >"$run/$arm-cute.log" 2>&1
    fi
    systemctl show "$current_unit.service" \
      -p ActiveState -p Result -p MemoryCurrent -p MemoryPeak -p CPUUsageNSec \
      >"$state/$arm-server-properties.txt"
    stop_server
  fi
done

"$python_bin" - "$summary" "$prereg" "$repo" "$state" "$v63_receipt" "$v65_receipt" <<'PY'
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

summary, prereg, repo, state, v63_receipt, v65_receipt = map(Path, sys.argv[1:])
def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
arms = {}
for name in ("cq3_mlp_q4_attention", "q4_mlp_aff3_attention"):
    canary_path = state / f"{name}-canary.json"
    cute_path = state / f"{name}-cute.json"
    canary = json.loads(canary_path.read_text())
    cute = json.loads(cute_path.read_text())
    total = cute["summaries"][-1]
    if (
        canary.get("schema") != "qwen38_27b_v66_france_canary"
        or canary.get("arm") != name
        or not canary.get("complete") or not canary.get("terminal")
        or cute.get("schema") != "cpu_cute_quick_result_v1"
        or len(cute.get("records", [])) != 20
        or total.get("task") != "cute_quick_20" or total.get("n") != 20
    ):
        raise SystemExit(f"v66 denominator changed: {name}")
    functional = bool(canary["contains_paris"] and total["correct"] >= 7)
    arms[name] = {
        "model_sha256": canary["model_sha256"],
        "france_canary_content": canary["assistant_content"],
        "france_canary_contains_paris": canary["contains_paris"],
        "cute_correct": total["correct"],
        "cute_denominator": total["n"],
        "functional_at_frozen_tripwire": functional,
        "canary_sha256": sha256(canary_path),
        "cute_sha256": sha256(cute_path),
    }
cq3 = arms["cq3_mlp_q4_attention"]["functional_at_frozen_tripwire"]
aff3 = arms["q4_mlp_aff3_attention"]["functional_at_frozen_tripwire"]
if not cq3 and aff3:
    classification = "mlp_cq3_primary_collapse_source"
elif cq3 and not aff3:
    classification = "attention_aff3_primary_collapse_source"
elif cq3 and aff3:
    classification = "cross_component_error_interaction"
else:
    classification = "both_approximations_independently_fail"
payload = {
    "schema": "qwen38_27b_trained_compact_factorial_diagnosis_v66",
    "model": "Qwen/Qwen3.8-27B",
    "repository_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
    "preregistration_sha256": sha256(prereg),
    "v63_receipt_sha256": sha256(v63_receipt),
    "v65_receipt_sha256": sha256(v65_receipt),
    "q4_cute_control": {"correct": 9, "n": 20},
    "functional_tripwire": {"requires_france_paris": True, "cute_correct_minimum": 7, "cute_n": 20},
    "arms": arms,
    "classification": classification,
    "full_cq3_aff3_candidate_remains_rejected": True,
    "performance_measurement_ran": False,
    "mmlu_ran": False,
    "claim_boundary": "Causal CUTE-Quick-20 catastrophic-forgetting diagnosis only; no promotion, full benchmark, performance, energy, or novelty claim.",
    "complete": True,
    "terminal": True,
}
with tempfile.NamedTemporaryFile(mode="w", dir=summary.parent, prefix=f".{summary.name}.", delete=False) as stream:
    json.dump(payload, stream, indent=2, sort_keys=True); stream.write("\n"); temporary=Path(stream.name)
os.replace(temporary, summary)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

"$python_bin" - "$state" "$manifest" "$marker" <<'PY'
import hashlib, json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
state, manifest, marker = map(Path, sys.argv[1:])
files=sorted(p for p in state.iterdir() if p.is_file() and p not in {manifest,marker})
with tempfile.NamedTemporaryFile(mode="w",dir=state,prefix=f".{manifest.name}.",delete=False) as s:
    for p in files: s.write(f"{sha256(p)}  {p.resolve()}\n")
    s.flush(); os.fsync(s.fileno()); temporary=Path(s.name)
os.replace(temporary,manifest)
payload={"schema":"qwen38_27b_trained_compact_factorial_diagnosis_marker_v66","executed_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"summary_sha256":sha256(state/"summary.json"),"terminal":True}
with tempfile.NamedTemporaryFile(mode="w",dir=state,prefix=f".{marker.name}.",delete=False) as s:
    json.dump(payload,s,indent=2,sort_keys=True); s.write("\n"); temporary=Path(s.name)
os.replace(temporary,marker)
with manifest.open("a") as s: s.write(f"{sha256(marker)}  {marker.resolve()}\n")
print(json.dumps(payload,indent=2,sort_keys=True))
PY
(cd / && sha256sum -c "$manifest")
