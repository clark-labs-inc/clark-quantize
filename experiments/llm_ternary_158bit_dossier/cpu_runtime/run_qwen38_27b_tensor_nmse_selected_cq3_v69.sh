#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v69}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime/qwen38_27b_tensor_nmse_selected_cq3_preregistration_v69.json"
materializer="$runtime/materialize_qwen38_27b_tensor_nmse_selected_cq3_v69.py"
scope_verifier="$runtime/verify_gguf_selective_conversion.py"
cute_manifest="$runtime/cute_quick_v1.json"
cute_evaluator="$runtime/eval_llama_cpp_cute_quick.py"
training_receipt="$runtime/receipts/qwen38_27b_all_layer_mixed_train_v44/terminal-receipt.json"
training_root=/workspace/quant-runs/qwen38-all-layer-mixed-trained-v44
v67_summary="$runtime/receipts/qwen38_27b_sensitivity_selected_cq3_half_v67/summary.json"
v68_summary="$repo/experiments/qwen38_27b_base3_fused/research/receipts/qwen38_27b_sensitivity_selected_cq3_48_v68/summary.json"
baseline=/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf
candidate=/workspace/models/candidates/Qwen3.8-27B-Q4_K_M-trained-CQ3-tensorNMSE143-v69.gguf
partial="${candidate}.partial"
converter_build=/workspace/build/llama.cpp-qwen38-selective-v1
runtime_build=/workspace/build/llama.cpp-qwen38-cq3-aff3-v63
quantize="$converter_build/bin/llama-quantize"
test_cq3="$converter_build/bin/test-cq3"
server="$runtime_build/bin/llama-server"
state=/workspace/quant-state/qwen38-tensor-nmse-selected-cq3-v69
run=/workspace/quant-runs/qwen38-tensor-nmse-selected-cq3-v69
scope="$state/gguf-scope.json"
materialization="$state/materialization.json"
canary="$state/france-canary.json"
cute="$state/cute.json"
summary="$state/summary.json"
manifest="$state/artifacts.sha256"
marker="$state/EXPERIMENT_COMPLETE.json"
port=${PORT:-8109}
threshold=0.04853973433570873
selected_tensor_count=143
expected_bytes=15190067552
expected_ranking=572e73a55e45f7dd6a76a0ab50c9ad6fcd0c37a76fc23d5981aded6a0a3228da
expected_selection=ae341b2351d56cc0d2ffab1366c719e845c06d5844db4b244760fdee5625fd5e

expected_prereg=d406691ebaf0f2329ad1ec53a21e4acd7b6652afdfea2a63a256687c5bf013fc
expected_materializer=4a26b74f37ed1178ad83076671acb7afa28907eca10a2c3ebbfbca44410e583e
expected_scope_verifier=3a2a864ad8d93b28154c04bddf2410a2f22c6887c373abea08c84c78cd526cce
expected_cute_manifest=19dd73eedf737d11f7c94d039822de62314abf31e3fb739a9a875fe65c5bbd16
expected_cute_evaluator=9c40a48a0c3f0544156a20ccccdc12f7e90bbae8afd391bf640ead1ff234c8ab
expected_training_receipt=1f33a04cf6d5433724e5d37d12fa0bfa3edfd710ab365033a8db69b517f73c1f
expected_v67_summary=0f895cea20c3ed5fa4e7be39dd890b92f45113198d73aed8ea34683fe64452cd
expected_v68_summary=e6c6ef695a27d7e00f6502244f3317a0b8170828990aa592269e0342e700a2c6
expected_baseline=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34
expected_quantize=30c8e7234e1e3486077f238901df5f83accd49248e928b80586f93e84bb87b7d
expected_test_cq3=2c1dfa7f1e1fca79d7fef2ff16de8315349b16a7346865c39a57a70288e01834

current_unit=
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

mkdir -p "$state" "$run" "$(dirname "$candidate")"
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

if [[ -n ${QWEN38_AFF3_DIR+x} || -n ${QWEN38_AFF3_LAYERS+x} || -n ${QWEN38_RESIDUAL_DIR+x} || -n ${QWEN38_RESIDUAL_LAYERS+x} ]]; then
  echo "v69 refuses AFF3 and residual environment variables" >&2
  exit 2
fi
if pgrep -x llama-server >/dev/null || pgrep -x llama-cli >/dev/null; then
  echo "another llama.cpp model workload is active" >&2
  exit 2
fi
test -x "$python_bin" -a -x "$quantize" -a -x "$test_cq3" -a -x "$server"
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
require_sha "$prereg" "$expected_prereg"
require_sha "$materializer" "$expected_materializer"
require_sha "$scope_verifier" "$expected_scope_verifier"
require_sha "$cute_manifest" "$expected_cute_manifest"
require_sha "$cute_evaluator" "$expected_cute_evaluator"
require_sha "$training_receipt" "$expected_training_receipt"
require_sha "$v67_summary" "$expected_v67_summary"
require_sha "$v68_summary" "$expected_v68_summary"
require_sha "$baseline" "$expected_baseline"
require_sha "$quantize" "$expected_quantize"
require_sha "$test_cq3" "$expected_test_cq3"

target_regex=$("$python_bin" - "$prereg" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
by_role=p["selection"]["selected_layers_by_role"]
parts=[]
for role in ("ffn_gate","ffn_up","ffn_down"):
    layers="|".join(str(value) for value in by_role[role])
    parts.append(f"({layers})\\.{role}")
print("^blk\\.("+"|".join(parts)+")\\.weight$")
PY
)

if [[ -e "$partial" ]]; then
  test -f "$partial"
  echo "removing only interrupted v69 partial: $partial"
  rm -f -- "$partial" "$scope" "$materialization"
fi
if [[ -e "$candidate" && ! -s "$materialization" ]]; then
  echo "existing v69 candidate lacks its materialization receipt" >&2
  exit 2
fi

if [[ ! -e "$candidate" ]]; then
  "$python_bin" - "$candidate" "$expected_bytes" <<'PY'
import os, sys
from pathlib import Path
parent=Path(sys.argv[1]).parent
available=os.statvfs(parent).f_bavail*os.statvfs(parent).f_frsize
required=int(sys.argv[2])+2*1024**3
if available < required:
    raise SystemExit(f"v69 needs {required} free bytes, observed {available}")
PY
  "$test_cq3" >"$run/test-cq3.log" 2>&1
  /usr/bin/time -v "$quantize" \
    --allow-requantize \
    --tensor-type "${target_regex}=CQ3_0" \
    "$baseline" "$partial" COPY 4 \
    >"$run/conversion.log" 2>&1
  sync "$partial"
  if [[ "$(stat -c %s "$partial")" != "$expected_bytes" ]]; then
    echo "v69 structural template byte count changed" >&2
    exit 2
  fi
  "$python_bin" - "$run/conversion.log" "$prereg" <<'PY'
import json, re, sys
from pathlib import Path
text=Path(sys.argv[1]).read_text()
p=json.load(open(sys.argv[2]))
expected={
    f"blk.{layer}.{role}.weight"
    for role,layers in p["selection"]["selected_layers_by_role"].items()
    for layer in layers
}
actual=set(re.findall(r"(blk\.[0-9]+\.ffn_(?:down|gate|up)\.weight)\s+- applying selective COPY override", text))
if actual != expected or text.count("applying selective COPY override") != len(expected):
    raise SystemExit(f"v69 override denominator changed: observed={len(actual)} expected={len(expected)}")
PY
  PYTHONPATH=/workspace/src/llama.cpp-qwen38-selective-v1/gguf-py \
    "$python_bin" "$scope_verifier" \
      --baseline "$baseline" \
      --candidate "$partial" \
      --target-regex "$target_regex" \
      --expected-target-count "$selected_tensor_count" \
      --baseline-target-type 12 \
      --candidate-target-type 43 \
      --out "$scope" \
      >"$run/gguf-scope.log" 2>&1
  PYTHONPATH=/workspace/src/llama.cpp-qwen38-selective-v1/gguf-py:"$runtime" \
    "$python_bin" "$materializer" \
      --template-partial "$partial" \
      --template-scope-receipt "$scope" \
      --training-receipt "$training_receipt" \
      --artifact-root "$training_root" \
      --threshold "$threshold" \
      --expected-count "$selected_tensor_count" \
      --expected-ranking-sha256 "$expected_ranking" \
      --expected-selection-sha256 "$expected_selection" \
      --output "$candidate" \
      --out "$materialization" \
      >"$run/materialization.log" 2>&1
fi

"$python_bin" - "$materialization" "$candidate" "$prereg" "$expected_bytes" <<'PY'
import hashlib, json, sys
from pathlib import Path
receipt=Path(sys.argv[1]); candidate=Path(sys.argv[2]); prereg=json.load(open(sys.argv[3])); expected_bytes=int(sys.argv[4])
r=json.loads(receipt.read_text())
h=hashlib.sha256()
with candidate.open("rb") as stream:
    for block in iter(lambda:stream.read(16*1024*1024),b""): h.update(block)
sha=h.hexdigest(); selection=prereg["selection"]
assert r["schema"] == "qwen38_27b_tensor_nmse_selected_cq3_materialization_v69" and r["valid"]
assert r["selection"]["threshold"] == selection["inclusive_threshold"]
assert r["selection"]["ranking_sha256"] == selection["ranking_sha256"]
assert r["selection"]["selection_sha256"] == selection["selection_sha256"]
assert r["selection"]["selected_tensor_count"] == selection["selected_tensor_count"]
assert r["selection"]["selected_layers_by_role"] == selection["selected_layers_by_role"]
assert candidate.stat().st_size == expected_bytes == r["candidate"]["bytes"]
assert sha == r["candidate"]["sha256"]
PY

current_unit=quant-qwen38-27b-v69-server
systemctl reset-failed "$current_unit.service" >/dev/null 2>&1 || true
systemd-run --quiet \
  --unit="$current_unit" \
  --property=MemoryMax=30000000000 \
  --property=MemorySwapMax=0 \
  --property=CPUQuota=400% \
  --property=AllowedCPUs=0,2,4,6 \
  --property=Restart=no \
  --property=RuntimeMaxSec=1800 \
  --setenv=LD_LIBRARY_PATH="$runtime_build/bin" \
  /usr/bin/taskset -c 0,2,4,6 \
  "$server" --model "$candidate" \
    --threads 4 --threads-batch 4 \
    --ctx-size 4096 --parallel 1 \
    --batch-size 128 --ubatch-size 64 \
    --host 127.0.0.1 --port "$port" \
    --no-warmup --log-disable
for _ in $(seq 1 180); do
  if curl --silent --fail "http://127.0.0.1:${port}/health" >/dev/null; then break; fi
  if ! systemctl is-active --quiet "$current_unit.service"; then
    echo "v69 server exited before health" >&2; exit 2
  fi
  sleep 1
done
curl --silent --fail "http://127.0.0.1:${port}/health" >/dev/null

candidate_sha=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate"]["sha256"])' "$materialization")
if [[ ! -s "$canary" ]]; then
  "$python_bin" - "$canary" "$port" "$candidate_sha" <<'PY'
import json, os, sys, tempfile, urllib.request
from pathlib import Path
out=Path(sys.argv[1]); port=sys.argv[2]; model_sha=sys.argv[3]
body={"model":"local","messages":[{"role":"user","content":"Complete this sentence with a short factual answer: The capital of France is"}],"max_tokens":8,"temperature":0,"seed":20260819,"stream":False,"chat_template_kwargs":{"enable_thinking":False}}
request=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"},method="POST")
with urllib.request.urlopen(request,timeout=300) as response: result=json.load(response)
choices=result.get("choices",[]); content=choices[0].get("message",{}).get("content","") if len(choices)==1 else ""
payload={"schema":"qwen38_27b_v69_france_canary","model_sha256":model_sha,"assistant_content":content,"contains_paris":"paris" in content.casefold(),"choices":len(choices),"enable_thinking":False,"max_tokens":8,"complete":True,"terminal":True}
with tempfile.NamedTemporaryFile(mode="w",dir=out.parent,prefix=f".{out.name}.",delete=False) as stream:
    json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n"); temporary=Path(stream.name)
os.replace(temporary,out); print(json.dumps(payload,sort_keys=True),flush=True)
PY
fi
if [[ ! -s "$cute" ]]; then
  "$python_bin" "$cute_evaluator" \
    --manifest "$cute_manifest" \
    --server "http://127.0.0.1:${port}" \
    --model-label qwen38-v69-tensor-nmse-143 \
    --model-sha256 "$candidate_sha" \
    --out "$cute" \
    >"$run/cute.log" 2>&1
fi
systemctl show "$current_unit.service" \
  -p ActiveState -p Result -p MemoryCurrent -p MemoryPeak -p CPUUsageNSec \
  >"$state/server-properties.txt"
stop_server

"$python_bin" - "$summary" "$prereg" "$repo" "$training_receipt" "$materialization" "$canary" "$cute" "$v67_summary" "$v68_summary" <<'PY'
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
summary, prereg, repo, training_receipt, materialization, canary, cute, v67_summary, v68_summary=map(Path,sys.argv[1:10])
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
p=json.loads(prereg.read_text()); m=json.loads(materialization.read_text()); c=json.loads(canary.read_text()); q=json.loads(cute.read_text())
selection=p["selection"]; total=q["summaries"][-1]
if q.get("schema") != "cpu_cute_quick_result_v1" or len(q.get("records",[])) != 20 or total.get("task") != "cute_quick_20" or total.get("n") != 20:
    raise SystemExit("v69 CUTE denominator changed")
if m["selection"]["ranking_sha256"] != selection["ranking_sha256"] or m["selection"]["selection_sha256"] != selection["selection_sha256"]:
    raise SystemExit("v69 frozen tensor selection changed")
functional=bool(c["contains_paris"] and total["correct"] >= 7)
payload={
  "schema":"qwen38_27b_tensor_nmse_selected_cq3_summary_v69",
  "model":"Qwen/Qwen3.8-27B",
  "repository_commit":subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),
  "preregistration_sha256":sha(prereg),
  "training_receipt_sha256":sha(training_receipt),
  "parent_summary_sha256":{"functional_v67":sha(v67_summary),"nonfunctional_v68":sha(v68_summary)},
  "selection":m["selection"],
  "candidate":m["candidate"],
  "france_canary_content":c["assistant_content"],
  "france_canary_contains_paris":c["contains_paris"],
  "cute_correct":total["correct"],
  "cute_denominator":total["n"],
  "cute_is_adaptive_development_evidence":True,
  "functional_at_development_tripwire":functional,
  "decision":"run_held_out_quality_gates_before_paired_throughput" if functional else "reject_scalar_tensor_nmse_separability_at_v67_threshold",
  "performance_measurement_ran":False,
  "held_out_quality_gates_ran":False,
  "claim_boundary":p["claim_boundary"],
  "complete":True,
  "terminal":True,
}
with tempfile.NamedTemporaryFile(mode="w",dir=summary.parent,prefix=f".{summary.name}.",delete=False) as stream:
    json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n"); temporary=Path(stream.name)
os.replace(temporary,summary); print(json.dumps(payload,indent=2,sort_keys=True))
PY

"$python_bin" - "$state" "$manifest" "$marker" "$candidate" <<'PY'
import hashlib, json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b""): h.update(block)
    return h.hexdigest()
state,manifest,marker,candidate=map(Path,sys.argv[1:5])
files=sorted(p for p in state.iterdir() if p.is_file() and p not in {manifest,marker})+[candidate]
with tempfile.NamedTemporaryFile(mode="w",dir=state,prefix=f".{manifest.name}.",delete=False) as stream:
    for path in files: stream.write(f"{sha(path)}  {path.resolve()}\n")
    stream.flush(); os.fsync(stream.fileno()); temporary=Path(stream.name)
os.replace(temporary,manifest)
payload={"schema":"qwen38_27b_tensor_nmse_selected_cq3_marker_v69","executed_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"summary_sha256":sha(state/"summary.json"),"terminal":True}
with tempfile.NamedTemporaryFile(mode="w",dir=state,prefix=f".{marker.name}.",delete=False) as stream:
    json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n"); temporary=Path(stream.name)
os.replace(temporary,marker)
with manifest.open("a") as stream: stream.write(f"{sha(marker)}  {marker.resolve()}\n")
print(json.dumps(payload,indent=2,sort_keys=True))
PY
(cd / && sha256sum -c "$manifest")
