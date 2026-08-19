#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v49-smoke}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
cli=${LLAMA_CLI:-/workspace/build/llama.cpp-qwen38-aff3-v49c/bin/llama-cli}
libllama=${LIBLLAMA:-/workspace/build/llama.cpp-qwen38-aff3-v49c/bin/libllama.so}
model=${MODEL:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf}
sidecar=${SIDECAR:-/workspace/quant-runs/qwen38-aff3-rotated-v47/layer-00-direct.aff3x16}
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-aff3-llamacpp-model-smoke-v49}
preregistration="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime/qwen38_27b_aff3_llamacpp_v49_development_preregistration.json"
receipt="$state_dir/receipt.json"
stdout_path="$state_dir/generation.stdout"
stderr_path="$state_dir/generation.stderr"
sidecar_dir="$state_dir/sidecars"
sidecar_alias="$sidecar_dir/layer-00.aff3x16"
unit=quant-qwen38-27b-aff3-llamacpp-model-smoke-v49
prompt='Complete this sentence with a short factual answer: The capital of France is'

require_sha256() {
  local path=$1 expected=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

if [[ -s "$receipt" ]]; then
  "$python_bin" - "$receipt" "$stdout_path" "$stderr_path" <<'PY'
import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


receipt_path, stdout_path, stderr_path = map(Path, sys.argv[1:])
receipt = json.loads(receipt_path.read_text())
if (
    receipt.get("schema") != "qwen38_27b_aff3_llamacpp_model_smoke_v49"
    or receipt.get("stdout", {}).get("sha256") != sha256(stdout_path)
    or receipt.get("stderr", {}).get("sha256") != sha256(stderr_path)
    or receipt.get("complete") is not True
    or receipt.get("terminal") is not True
):
    raise SystemExit("existing v49 model-smoke receipt changed")
print(json.dumps(receipt, indent=2, sort_keys=True))
raise SystemExit(0 if receipt.get("passed") is True else 2)
PY
  exit $?
fi

for active in \
  quant-qwen38-27b-all-layer-mixed-direct-v43.service \
  quant-qwen38-27b-all-layer-mixed-trained-v44.service; do
  if systemctl is-active --quiet "$active"; then
    echo "refusing to overlap actual-model smoke with $active" >&2
    exit 2
  fi
done
if systemctl list-units --type=service --state=running --no-legend \
  | grep -Eiq 'qwen35|qwen3[.]5|4b'; then
  echo "a Qwen 4B-era service is active; refusing to share the node" >&2
  exit 2
fi

test -x "$python_bin"
test -x "$cli"
test -f "$libllama"
test -f "$model"
test -f "$sidecar"
test -f "$preregistration"
require_sha256 "$preregistration" 85dce05fb4306d4fd4974c64df7b478f1b949ced81cb3af0ab201e8d147cacfe
require_sha256 "$cli" 1bd2b6d3d5f0b1b988508bb6682cd9af896b40da46422200a47b887a7c4ae600
require_sha256 "$libllama" dc4860ba93c9e2df68e001aa7dba159b062edc7297555b80ad67651e978bd83f
require_sha256 "$model" 31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34
require_sha256 "$sidecar" a4cec7372ac21a0093d047c1452277508583ebca1e3f4f66c73a8a74142ccf48

mkdir -p "$state_dir" "$sidecar_dir"
if [[ -e "$sidecar_alias" || -L "$sidecar_alias" ]]; then
  if [[ $(readlink -f "$sidecar_alias") != $(readlink -f "$sidecar") ]]; then
    echo "existing layer-0 sidecar alias changed" >&2
    exit 2
  fi
else
  ln -s "$sidecar" "$sidecar_alias"
fi

status=0
QWEN38_AFF3_DIR="$sidecar_dir" QWEN38_AFF3_LAYERS=0 \
systemd-run --quiet --wait --pipe --collect \
  --unit="$unit" \
  --property=MemoryMax=30000000000 \
  --property=MemorySwapMax=0 \
  --property=CPUQuota=400% \
  --property=AllowedCPUs=0,2,4,6 \
  --property=Restart=no \
  --setenv=QWEN38_AFF3_DIR="$sidecar_dir" \
  --setenv=QWEN38_AFF3_LAYERS=0 \
  /usr/bin/taskset -c 0,2,4,6 \
  "$cli" \
  --model "$model" \
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
  --no-display-prompt \
  --simple-io \
  --no-warmup \
  --no-perf \
  >"$stdout_path.tmp" 2>"$stderr_path.tmp" || status=$?
mv "$stdout_path.tmp" "$stdout_path"
mv "$stderr_path.tmp" "$stderr_path"

"$python_bin" - \
  "$receipt" "$stdout_path" "$stderr_path" "$status" \
  "$repo" "$preregistration" "$cli" "$libllama" "$model" "$sidecar" \
  "$prompt" <<'PY'
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
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


(
    receipt_path,
    stdout_path,
    stderr_path,
    exit_code,
    repo,
    preregistration,
    cli,
    libllama,
    model,
    sidecar,
    prompt,
) = sys.argv[1:]
receipt_path = Path(receipt_path)
stdout_path = Path(stdout_path)
stderr_path = Path(stderr_path)
repo = Path(repo)
stderr_text = stderr_path.read_text(errors="replace")
stdout_bytes = stdout_path.read_bytes()
performance_markers = (
    "llama_perf_context_print",
    "eval time =",
    "prompt eval time =",
    "tokens per second",
)
performance_timing_ran = any(marker in stderr_text for marker in performance_markers)
passed = int(exit_code) == 0 and bool(stdout_bytes.strip()) and not performance_timing_ran
payload = {
    "schema": "qwen38_27b_aff3_llamacpp_model_smoke_v49",
    "model": "Qwen/Qwen3.8-27B",
    "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "repository_commit": subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip(),
    "preregistration_sha256": sha256(Path(preregistration)),
    "inputs": {
        "llama_cli_sha256": sha256(Path(cli)),
        "libllama_sha256": sha256(Path(libllama)),
        "q4_control_gguf_sha256": sha256(Path(model)),
        "direct_v43_layer0_aff3_sha256": sha256(Path(sidecar)),
        "active_aff3_layers": [0],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    },
    "execution": {
        "exit_code": int(exit_code),
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
        "performance_timing_ran": performance_timing_ran,
        "bounded_free_generation_nonempty": bool(stdout_bytes.strip()),
        "actual_qwen_model_graph_execution_passed": int(exit_code) == 0,
    },
    "stdout": {
        "path": str(stdout_path),
        "bytes": len(stdout_bytes),
        "sha256": sha256(stdout_path),
    },
    "stderr": {
        "path": str(stderr_path),
        "bytes": stderr_path.stat().st_size,
        "sha256": sha256(stderr_path),
    },
    "claim_boundary": (
        "This development smoke can establish only that the exact v49 llama.cpp "
        "binary loads the actual Qwen3.8-27B Q4 graph with one authenticated "
        "direct-v43 AFF3 projection and completes bounded free generation. It is "
        "not trained-v44, quality, throughput, memory, energy, novelty, or "
        "scientific-breakthrough evidence."
    ),
    "passed": passed,
    "complete": True,
    "terminal": True,
}
receipt_path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(
    mode="w", dir=receipt_path.parent, prefix=f".{receipt_path.name}.", delete=False
) as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    temporary = Path(stream.name)
os.replace(temporary, receipt_path)
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(0 if passed else 2)
PY
