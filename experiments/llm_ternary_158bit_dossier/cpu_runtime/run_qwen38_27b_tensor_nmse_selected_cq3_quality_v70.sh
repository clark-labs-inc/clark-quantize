#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v70}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
runtime="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime"
prereg="$runtime/qwen38_27b_tensor_nmse_selected_cq3_quality_preregistration_v70.json"
v69_state=/workspace/quant-state/qwen38-tensor-nmse-selected-cq3-v69
v69_summary="$v69_state/summary.json"
v69_materialization="$v69_state/materialization.json"
v69_manifest="$v69_state/artifacts.sha256"
v69_marker="$v69_state/EXPERIMENT_COMPLETE.json"
baseline_summary=/workspace/quant-state/qwen38-27b-cpu-q4-baseline-v1/summary.json
model=/workspace/models/candidates/Qwen3.8-27B-Q4_K_M-trained-CQ3-tensorNMSE143-v69.gguf
build=/workspace/build/llama.cpp-qwen38-cq3-aff3-v63
server="$build/bin/llama-server"
perplexity="$build/bin/llama-perplexity"
math_code_manifest="$runtime/math_code_canary_v1.json"
mmlu_manifest="$runtime/qwen35_4b_cpu_layer30_mmlu_v1.json"
wiki_corpus=/workspace/quant-runs/cq3-4b-ptq-scope-20260817/wikitext-2-raw/wiki.test.raw
state=/workspace/quant-state/qwen38-tensor-nmse-selected-cq3-quality-v70
run=/workspace/quant-runs/qwen38-tensor-nmse-selected-cq3-quality-v70
math_code="$state/math-code.json"
mmlu="$state/mmlu.json"
wiki="$state/wikitext.json"
summary="$state/summary.json"
manifest="$state/artifacts.sha256"
marker="$state/QUALITY_COMPLETE.json"
port=${PORT:-8110}
server_unit=quant-qwen38-27b-v70-quality-server

expected_prereg=40873d754fd8e647de916a083a197f838029f85000a09703c589b42d0abe4b9a
expected_parent_prereg=d406691ebaf0f2329ad1ec53a21e4acd7b6652afdfea2a63a256687c5bf013fc
expected_baseline_summary=1557f1aac13f6c557e89006427aa829e2707f07ad73aa29611763e754c409a64
expected_math_code_manifest=1ec0b41ce3ca3597d65013ac0c8a48c208c4e03b05f0109be61319252c09dab0
expected_mmlu_manifest=e3e38ccb1bf35c56b600f1cf39bbd0ea38072f2b4bdd5fc36146c143fb658dba
expected_wiki_corpus=173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08
expected_math_code_evaluator=fa3d0f870b3b0ed1d7d907cc8d146854b164e6d75cfa9d2bd20743979af74677
expected_mmlu_evaluator=a604ca9a528231fa17de78ab2a7ffed7c1b9a0b9a9ca74459c2d894f1f3612d3
expected_wiki_evaluator=54458dc377a1d07aaf26df90823f65f973b625f55a6ac73895d0c6ec9a666b07
expected_server=58d611c2a2bb38fdb8d32121d4b7a54180ad89233020037a11cdd67b6774177d
expected_perplexity=be9149ccdc77343087fff7de7d2370f3285cb0082ac23c20b458445053101c34
expected_model_bytes=15190067552

require_sha() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}
stop_server() {
  systemctl stop "$server_unit.service" >/dev/null 2>&1 || true
}
trap stop_server EXIT

mkdir -p "$state" "$run"
if [[ -f "$marker" || -f "$manifest" || -f "$summary" ]]; then
  test -f "$marker" -a -f "$manifest" -a -f "$summary"
  (cd / && sha256sum -c "$manifest")
  "$python_bin" - "$summary" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); assert r["complete"] and r["terminal"]
print(json.dumps(r,indent=2,sort_keys=True))
PY
  exit 0
fi

if pgrep -x llama-server >/dev/null || pgrep -x llama-cli >/dev/null || pgrep -x llama-perplexity >/dev/null; then
  echo "another llama.cpp model workload is active" >&2
  exit 2
fi
test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
test -f "$v69_marker" -a -f "$v69_manifest" -a -f "$v69_summary" -a -f "$v69_materialization" -a -f "$model"
test -x "$server" -a -x "$perplexity" -a -x "$python_bin"
require_sha "$prereg" "$expected_prereg"
require_sha "$runtime/qwen38_27b_tensor_nmse_selected_cq3_preregistration_v69.json" "$expected_parent_prereg"
require_sha "$baseline_summary" "$expected_baseline_summary"
require_sha "$math_code_manifest" "$expected_math_code_manifest"
require_sha "$mmlu_manifest" "$expected_mmlu_manifest"
require_sha "$wiki_corpus" "$expected_wiki_corpus"
require_sha "$runtime/eval_llama_cpp_math_code_canary.py" "$expected_math_code_evaluator"
require_sha "$runtime/eval_llama_cpp_mmlu.py" "$expected_mmlu_evaluator"
require_sha "$runtime/eval_llama_cpp_wikitext_quick.py" "$expected_wiki_evaluator"
require_sha "$server" "$expected_server"
require_sha "$perplexity" "$expected_perplexity"
if [[ "$(stat -c %s "$model")" != "$expected_model_bytes" ]]; then
  echo "unexpected v69 candidate byte count" >&2
  exit 2
fi
(cd / && sha256sum -c "$v69_manifest")

model_sha=$("$python_bin" - "$prereg" "$v69_summary" "$v69_materialization" "$v69_marker" "$model" <<'PY'
import hashlib,json,sys
from pathlib import Path
prereg,summary_path,materialization_path,marker_path,model=map(Path,sys.argv[1:])
p=json.loads(prereg.read_text()); s=json.loads(summary_path.read_text()); m=json.loads(materialization_path.read_text()); marker=json.loads(marker_path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b""): h.update(block)
    return h.hexdigest()
actual=sha(model)
if s.get("schema") != p["conditional_admission"]["required_parent_summary_schema"] or not s.get("functional_at_development_tripwire"):
    raise SystemExit("v69 did not pass its preregistered development screen")
if m.get("schema") != p["conditional_admission"]["required_materialization_schema"] or not m.get("valid"):
    raise SystemExit("v69 materialization receipt is not valid")
if m["candidate"]["bytes"] != p["conditional_admission"]["required_candidate_bytes"] or m["candidate"]["sha256"] != actual:
    raise SystemExit("v69 candidate identity changed")
if marker.get("summary_sha256") != hashlib.sha256(summary_path.read_bytes()).hexdigest() or not marker.get("terminal"):
    raise SystemExit("v69 terminal marker does not authenticate summary")
print(actual)
PY
)

git -C "$repo" rev-parse HEAD >"$state/source.commit"
sha256sum "$prereg" "$v69_summary" "$v69_materialization" "$v69_marker" \
  "$baseline_summary" "$math_code_manifest" "$mmlu_manifest" "$wiki_corpus" \
  "$runtime/eval_llama_cpp_math_code_canary.py" \
  "$runtime/eval_llama_cpp_mmlu.py" \
  "$runtime/eval_llama_cpp_wikitext_quick.py" \
  "$server" "$perplexity" "$model" >"$state/inputs.sha256"

systemctl reset-failed "$server_unit.service" >/dev/null 2>&1 || true
systemd-run --quiet \
  --unit="$server_unit" \
  --property=MemoryMax=30000000000 \
  --property=MemorySwapMax=0 \
  --property=CPUQuota=400% \
  --property=AllowedCPUs=0,2,4,6 \
  --property=Restart=no \
  --property=RuntimeMaxSec=21600 \
  --setenv=LD_LIBRARY_PATH="$build/bin" \
  /usr/bin/taskset -c 0,2,4,6 \
  "$server" --model "$model" \
    --threads 4 --threads-batch 4 \
    --ctx-size 8192 --parallel 1 \
    --batch-size 512 --ubatch-size 512 \
    --host 127.0.0.1 --port "$port" \
    --no-warmup --log-file "$run/server.log"
for _ in $(seq 1 180); do
  if curl --silent --fail "http://127.0.0.1:${port}/health" >/dev/null; then break; fi
  if ! systemctl is-active --quiet "$server_unit.service"; then
    echo "v70 server exited before health" >&2; exit 2
  fi
  sleep 1
done
curl --silent --fail "http://127.0.0.1:${port}/health" >/dev/null

if [[ ! -s "$math_code" ]]; then
  "$python_bin" "$runtime/eval_llama_cpp_math_code_canary.py" \
    --manifest "$math_code_manifest" \
    --server "http://127.0.0.1:${port}" \
    --model-label qwen38-v69-tensor-nmse-143 \
    --model-sha256 "$model_sha" \
    --out "$math_code" \
    >"$run/math-code.log" 2>&1
fi

if [[ ! -s "$mmlu" ]]; then
  resume_args=()
  if [[ -s "$state/mmlu.json.partial" ]]; then
    resume_args+=(--resume)
  fi
  PYTHONPATH="$repo" "$python_bin" "$runtime/eval_llama_cpp_mmlu.py" \
    --manifest "$mmlu_manifest" \
    --model "$model" \
    --model-label qwen38-v69-tensor-nmse-143 \
    --model-sha256 "$model_sha" \
    --server "http://127.0.0.1:${port}" \
    --workers 1 \
    --timeout 900 \
    --checkpoint-every 4 \
    --out "$mmlu" \
    "${resume_args[@]}" \
    >"$run/mmlu.log" 2>&1
fi
stop_server

if [[ ! -s "$wiki" ]]; then
  LD_LIBRARY_PATH="$build/bin" /usr/bin/taskset -c 0,2,4,6 \
    "$python_bin" "$runtime/eval_llama_cpp_wikitext_quick.py" \
      --model "$model" \
      --model-label qwen38-v69-tensor-nmse-143 \
      --corpus "$wiki_corpus" \
      --llama-perplexity "$perplexity" \
      --chunks 4 --context 512 --threads 4 \
      --out "$wiki" \
      >"$run/wikitext.log" 2>&1
fi

"$python_bin" - "$summary" "$prereg" "$baseline_summary" "$v69_summary" "$math_code" "$mmlu" "$wiki" "$model_sha" "$repo" <<'PY'
import hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
summary,prereg,baseline_path,parent_path,math_path,mmlu_path,wiki_path=map(Path,sys.argv[1:8]); model_sha=sys.argv[8]; repo=Path(sys.argv[9])
p=json.loads(prereg.read_text()); baseline=json.loads(baseline_path.read_text()); parent=json.loads(parent_path.read_text()); math_code=json.loads(math_path.read_text()); mmlu=json.loads(mmlu_path.read_text()); wiki=json.loads(wiki_path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def by_task(rows): return {row["task"]:row for row in rows}
bmath=by_task(baseline["math_code"]); cmath=by_task(math_code["summaries"]); gate=p["promotion_gate"]
receipts_valid=(
    math_code.get("model_sha256")==model_sha and len(math_code.get("records",[]))==14
    and mmlu.get("valid") and mmlu.get("status")=="completed" and mmlu.get("summary",{}).get("n")==64
    and mmlu.get("contract",{}).get("model_sha256")==model_sha
    and wiki.get("valid") and wiki.get("model_sha256")==model_sha
)
observed={
  "development_cute_correct":parent["cute_correct"],
  "development_cute_denominator":parent["cute_denominator"],
  "math500_correct":int(cmath["math500"]["correct"]),
  "math500_denominator":int(cmath["math500"]["n"]),
  "code_trace_correct":int(cmath["code_trace"]["correct"]),
  "code_trace_denominator":int(cmath["code_trace"]["n"]),
  "mmlu_correct":int(mmlu["summary"]["correct"]),
  "mmlu_denominator":int(mmlu["summary"]["n"]),
  "wikitext_ppl":float(wiki["final_ppl"]),
  "wikitext_ppl_ratio":float(wiki["final_ppl"])/float(baseline["wikitext_ppl"]),
}
checks={
  "all_receipts_valid":receipts_valid,
  "math500":observed["math500_correct"]>=gate["math500_correct_min"],
  "code_trace":observed["code_trace_correct"]>=gate["code_trace_correct_min"],
  "mmlu":observed["mmlu_correct"]>=gate["mmlu_correct_min"],
  "wikitext":observed["wikitext_ppl_ratio"]<=gate["wikitext_ppl_ratio_max"],
}
payload={
  "schema":"qwen38_27b_tensor_nmse_selected_cq3_quality_summary_v70",
  "repository_commit":subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(),
  "preregistration_sha256":sha(prereg),
  "parent_summary_sha256":sha(parent_path),
  "candidate_model_sha256":model_sha,
  "baseline":{
    "summary_sha256":sha(baseline_path),
    "math500_correct":bmath["math500"]["correct"],
    "code_trace_correct":bmath["code_trace"]["correct"],
    "mmlu_correct":baseline["mmlu"]["correct"],
    "wikitext_ppl":baseline["wikitext_ppl"],
  },
  "observed":observed,
  "checks":checks,
  "promotion_pass":all(checks.values()),
  "decision":"admit_to_paired_performance" if all(checks.values()) else "reject_before_performance",
  "cute_excluded_from_promotion_as_adaptive":True,
  "claim_boundary":p["claim_boundary"],
  "complete":True,
  "terminal":True,
}
with tempfile.NamedTemporaryFile(mode="w",dir=summary.parent,prefix=f".{summary.name}.",delete=False) as stream:
    json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n"); temporary=Path(stream.name)
os.replace(temporary,summary); print(json.dumps(payload,indent=2,sort_keys=True))
PY

"$python_bin" - "$state" "$manifest" "$marker" "$model" <<'PY'
import hashlib,json,os,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b""): h.update(block)
    return h.hexdigest()
state,manifest,marker,model=map(Path,sys.argv[1:5])
files=sorted(p for p in state.iterdir() if p.is_file() and p not in {manifest,marker})+[model]
with tempfile.NamedTemporaryFile(mode="w",dir=state,prefix=f".{manifest.name}.",delete=False) as stream:
    for path in files: stream.write(f"{sha(path)}  {path.resolve()}\n")
    stream.flush(); os.fsync(stream.fileno()); temporary=Path(stream.name)
os.replace(temporary,manifest)
summary=json.loads((state/"summary.json").read_text())
payload={"schema":"qwen38_27b_tensor_nmse_selected_cq3_quality_marker_v70","executed_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"summary_sha256":sha(state/"summary.json"),"promotion_pass":summary["promotion_pass"],"terminal":True}
with tempfile.NamedTemporaryFile(mode="w",dir=state,prefix=f".{marker.name}.",delete=False) as stream:
    json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n"); temporary=Path(stream.name)
os.replace(temporary,marker)
with manifest.open("a") as stream: stream.write(f"{sha(marker)}  {marker.resolve()}\n")
print(json.dumps(payload,indent=2,sort_keys=True))
PY
(cd / && sha256sum -c "$manifest")
