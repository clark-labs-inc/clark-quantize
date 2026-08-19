#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v67}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime/qwen38_27b_sensitivity_selected_cq3_half_preregistration_v67.json"
materializer="$runtime/materialize_qwen38_27b_sensitivity_selected_cq3_v67.py"
scope_verifier="$runtime/verify_gguf_selective_conversion.py"
cute_manifest="$runtime/cute_quick_v1.json"
cute_evaluator="$runtime/eval_llama_cpp_cute_quick.py"
training_receipt="$runtime/receipts/qwen38_27b_all_layer_mixed_train_v44/terminal-receipt.json"
training_root=/workspace/quant-runs/qwen38-all-layer-mixed-trained-v44
baseline=/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf
candidate=/workspace/models/candidates/Qwen3.8-27B-Q4_K_M-trained-CQ3-sensitivity32-v67.gguf
partial="${candidate}.partial"
converter_build=/workspace/build/llama.cpp-qwen38-selective-v1
runtime_build=/workspace/build/llama.cpp-qwen38-cq3-aff3-v63
quantize="$converter_build/bin/llama-quantize"
test_cq3="$converter_build/bin/test-cq3"
server="$runtime_build/bin/llama-server"
state=${STATE_DIR:-/workspace/quant-state/qwen38-sensitivity-selected-cq3-half-v67}
run=${RUN_DIR:-/workspace/quant-runs/qwen38-sensitivity-selected-cq3-half-v67}
scope="$state/gguf-scope.json"
materialization="$state/materialization.json"
canary="$state/france-canary.json"
cute="$state/cute.json"
summary="$state/summary.json"
manifest="$state/artifacts.sha256"
marker="$state/EXPERIMENT_COMPLETE.json"
port=8107
layers='1,3,4,5,7,8,9,10,11,12,13,14,15,16,17,19,20,21,23,24,25,27,28,29,36,37,39,40,41,43,44,45'
layer_regex='(1|3|4|5|7|8|9|10|11|12|13|14|15|16|17|19|20|21|23|24|25|27|28|29|36|37|39|40|41|43|44|45)'
target_regex="^blk\\.${layer_regex}\\.ffn_(down|gate|up)\\.weight$"
expected_bytes=16433695072

expected_prereg=6f4b233192ddb89240d1849212d5ef8d825f39db23cad35cc798e7f56ca4a2ef
expected_materializer=0018bfd81de0791c3de568963d751244dc6595efd497939006e3b9f593aacff9
expected_scope_verifier=3a2a864ad8d93b28154c04bddf2410a2f22c6887c373abea08c84c78cd526cce
expected_cute_manifest=19dd73eedf737d11f7c94d039822de62314abf31e3fb739a9a875fe65c5bbd16
expected_cute_evaluator=9c40a48a0c3f0544156a20ccccdc12f7e90bbae8afd391bf640ead1ff234c8ab
expected_training_receipt=1f33a04cf6d5433724e5d37d12fa0bfa3edfd710ab365033a8db69b517f73c1f
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
  echo "v67 refuses AFF3 and residual environment variables" >&2
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
require_sha "$baseline" "$expected_baseline"
require_sha "$quantize" "$expected_quantize"
require_sha "$test_cq3" "$expected_test_cq3"

if [[ -e "$partial" ]]; then
  test -f "$partial"
  echo "removing only interrupted v67 partial: $partial"
  rm -f -- "$partial" "$scope" "$materialization"
fi
if [[ -e "$candidate" && ! -s "$materialization" ]]; then
  echo "existing v67 candidate lacks its materialization receipt" >&2
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
    raise SystemExit(f"v67 needs {required} free bytes, observed {available}")
PY
  "$test_cq3" >"$run/test-cq3.log" 2>&1
  /usr/bin/time -v "$quantize" \
    --allow-requantize \
    --tensor-type "${target_regex}=CQ3_0" \
    "$baseline" "$partial" COPY 4 \
    >"$run/conversion.log" 2>&1
  sync "$partial"
  if [[ "$(stat -c %s "$partial")" != "$expected_bytes" ]]; then
    echo "v67 structural template byte count changed" >&2
    exit 2
  fi
  "$python_bin" - "$run/conversion.log" "$layers" <<'PY'
import re, sys
from pathlib import Path
text=Path(sys.argv[1]).read_text()
layers=[int(value) for value in sys.argv[2].split(",")]
expected={f"blk.{layer}.ffn_{kind}.weight" for layer in layers for kind in ("down","gate","up")}
actual=set(re.findall(r"(blk\.[0-9]+\.ffn_(?:down|gate|up)\.weight)\s+- applying selective COPY override", text))
if actual != expected or text.count("applying selective COPY override") != len(expected):
    raise SystemExit(f"v67 override denominator changed: observed={len(actual)} expected={len(expected)}")
PY
  PYTHONPATH=/workspace/src/llama.cpp-qwen38-selective-v1/gguf-py \
    "$python_bin" "$scope_verifier" \
      --baseline "$baseline" \
      --candidate "$partial" \
      --target-regex "$target_regex" \
      --expected-target-count 96 \
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
      --layers "$layers" \
      --output "$candidate" \
      --out "$materialization" \
      >"$run/materialization.log" 2>&1
fi

"$python_bin" - "$materialization" "$candidate" "$layers" "$expected_bytes" <<'PY'
import hashlib, json, sys
from pathlib import Path
receipt=Path(sys.argv[1]); candidate=Path(sys.argv[2]); layers=[int(x) for x in sys.argv[3].split(",")]; expected_bytes=int(sys.argv[4])
r=json.loads(receipt.read_text())
h=hashlib.sha256()
with candidate.open("rb") as stream:
    for block in iter(lambda:stream.read(16*1024*1024),b""): h.update(block)
sha=h.hexdigest()
assert r["valid"] and r["selected_layers"] == layers and r["selected_tensor_count"] == 96
assert candidate.stat().st_size == expected_bytes == r["candidate"]["bytes"]
assert sha == r["candidate"]["sha256"]
PY

current_unit=quant-qwen38-27b-v67-server
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
    echo "v67 server exited before health" >&2; exit 2
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
payload={"schema":"qwen38_27b_v67_france_canary","model_sha256":model_sha,"assistant_content":content,"contains_paris":"paris" in content.casefold(),"choices":len(choices),"enable_thinking":False,"max_tokens":8,"complete":True,"terminal":True}
with tempfile.NamedTemporaryFile(mode="w",dir=out.parent,prefix=f".{out.name}.",delete=False) as stream:
    json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n"); temporary=Path(stream.name)
os.replace(temporary,out); print(json.dumps(payload,sort_keys=True),flush=True)
PY
fi
if [[ ! -s "$cute" ]]; then
  "$python_bin" "$cute_evaluator" \
    --manifest "$cute_manifest" \
    --server "http://127.0.0.1:${port}" \
    --model-label qwen38-v67-sensitivity32 \
    --model-sha256 "$candidate_sha" \
    --out "$cute" \
    >"$run/cute.log" 2>&1
fi
systemctl show "$current_unit.service" \
  -p ActiveState -p Result -p MemoryCurrent -p MemoryPeak -p CPUUsageNSec \
  >"$state/server-properties.txt"
stop_server

"$python_bin" - "$summary" "$prereg" "$repo" "$training_receipt" "$materialization" "$canary" "$cute" <<'PY'
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
summary, prereg, repo, training_receipt, materialization, canary, cute=map(Path,sys.argv[1:])
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
p=json.loads(prereg.read_text()); t=json.loads(training_receipt.read_text()); m=json.loads(materialization.read_text()); c=json.loads(canary.read_text()); q=json.loads(cute.read_text())
ranking=[]
for layer in t["layers"]:
    score=max(item["aggregate_calibration_metrics"]["selected_calibration_nmse"] for item in layer["cq3_training"])
    ranking.append({"layer":layer["layer"],"score":score})
ranking.sort(key=lambda item:(item["score"],item["layer"]))
ranking_sha=hashlib.sha256(json.dumps(ranking,sort_keys=True,separators=(",",":")).encode()).hexdigest()
selected=sorted(item["layer"] for item in ranking[:32]); total=q["summaries"][-1]
if selected != p["selection"]["admitted_layers"] or ranking_sha != p["selection"]["ranking_sha256"]:
    raise SystemExit("v67 frozen sensitivity ranking changed")
if q.get("schema") != "cpu_cute_quick_result_v1" or len(q.get("records",[])) != 20 or total.get("task") != "cute_quick_20" or total.get("n") != 20:
    raise SystemExit("v67 CUTE denominator changed")
functional=bool(c["contains_paris"] and total["correct"] >= 7)
payload={
  "schema":"qwen38_27b_sensitivity_selected_cq3_half_summary_v67",
  "model":"Qwen/Qwen3.8-27B",
  "repository_commit":subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),
  "preregistration_sha256":sha(prereg),
  "training_receipt_sha256":sha(training_receipt),
  "ranking_sha256":ranking_sha,
  "selected_layers":selected,
  "selected_layer_count":32,
  "selected_tensor_count":96,
  "candidate":m["candidate"],
  "france_canary_content":c["assistant_content"],
  "france_canary_contains_paris":c["contains_paris"],
  "cute_correct":total["correct"],
  "cute_denominator":total["n"],
  "functional_at_frozen_tripwire":functional,
  "decision":"promote_to_more_aggressive_sensitivity_threshold" if functional else "reduce_admitted_layer_count_under_same_frozen_ranking",
  "performance_measurement_ran":False,
  "mmlu_ran":False,
  "claim_boundary":"Frozen 32-layer CUTE screening result only; no perplexity, MATH, code, MMLU, throughput, energy, novelty, or breakthrough claim.",
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
state,manifest,marker,candidate=map(Path,sys.argv[1:])
files=sorted(p for p in state.iterdir() if p.is_file() and p not in {manifest,marker})+[candidate]
with tempfile.NamedTemporaryFile(mode="w",dir=state,prefix=f".{manifest.name}.",delete=False) as stream:
    for path in files: stream.write(f"{sha(path)}  {path.resolve()}\n")
    stream.flush(); os.fsync(stream.fileno()); temporary=Path(stream.name)
os.replace(temporary,manifest)
payload={"schema":"qwen38_27b_sensitivity_selected_cq3_half_marker_v67","executed_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"summary_sha256":sha(state/"summary.json"),"terminal":True}
with tempfile.NamedTemporaryFile(mode="w",dir=state,prefix=f".{marker.name}.",delete=False) as stream:
    json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n"); temporary=Path(stream.name)
os.replace(temporary,marker)
with manifest.open("a") as stream: stream.write(f"{sha(marker)}  {marker.resolve()}\n")
print(json.dumps(payload,indent=2,sort_keys=True))
PY
(cd / && sha256sum -c "$manifest")
