#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-/workspace/src/clark-quantize-qwen38-v50}
python_bin=${PYTHON_BIN:-/workspace/venvs/quant-cpu/bin/python}
llama_source=${LLAMA_SOURCE:-/workspace/src/llama.cpp-qwen38-aff3-v50}
runtime_lib_dir=${RUNTIME_LIB_DIR:-/workspace/build/llama.cpp-qwen38-aff3-v50/bin}
cli=${LLAMA_CLI:-$runtime_lib_dir/llama-cli}
standalone=${STANDALONE:-$runtime_lib_dir/qwen38-aff3-runtime-smoke-v50}
model=${MODEL:-/workspace/models/controls/Qwen3.8-27B-Q4_K_M.gguf}
sidecar=${SIDECAR:-/workspace/quant-runs/qwen38-aff3-rotated-v47/layer-00-direct.aff3x16}
state_dir=${STATE_DIR:-/workspace/quant-state/qwen38-aff3-llamacpp-model-smoke-v50}
one_token=${ONE_TOKEN_RECEIPT:-/workspace/quant-state/qwen38-aff3-llamacpp-repair-v50-one-token.json}
eight_token=${EIGHT_TOKEN_RECEIPT:-/workspace/quant-state/qwen38-aff3-llamacpp-repair-v50-eight-token.json}
preregistration="$repo/experiments/llm_ternary_158bit_dossier/cpu_runtime/qwen38_27b_aff3_llamacpp_v50_repair_preregistration.json"
receipt="$state_dir/receipt.json"
stdout_path="$state_dir/generation.stdout"
stderr_path="$state_dir/generation.stderr"
sidecar_dir="$state_dir/sidecars"
sidecar_alias="$sidecar_dir/layer-00.aff3x16"
unit=quant-qwen38-27b-aff3-llamacpp-model-smoke-v50
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
    receipt.get("schema") != "qwen38_27b_aff3_llamacpp_model_smoke_v50"
    or receipt.get("stdout", {}).get("sha256") != sha256(stdout_path)
    or receipt.get("stderr", {}).get("sha256") != sha256(stderr_path)
    or receipt.get("complete") is not True
    or receipt.get("terminal") is not True
):
    raise SystemExit("existing v50 model-smoke receipt changed")
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

for path in "$python_bin" "$cli" "$standalone" "$model" "$sidecar" \
  "$preregistration" "$one_token" "$eight_token" \
  "$llama_source/src/models/qwen35.cpp"; do
  test -f "$path"
done
test -x "$python_bin"
test -x "$cli"
test -x "$standalone"

require_sha256 "$preregistration" a6ad20f285f4c47e7e094acf8b4ebc8bd1f9d4c9d1649efec882fcd0a3e09381
require_sha256 "$llama_source/src/models/qwen35.cpp" 6953d033c467029db896efa528b6261eb04999b94d3e731952b75f4ffa5e36d5
require_sha256 "$cli" e55a49bfc03e7ae6346a7f3e8240a18cddd64a176a8cc6d61e9ab7f1a38ff92c
require_sha256 "$standalone" b8033fe6c240ccef2e2f76d0d1249f7e24a33c929451ec565bf0b97f1ce3fe84
require_sha256 "$runtime_lib_dir/libllama.so" 942ab23f53a5339a0665af73fa661d4bd7331850e39f0a1ef2bb2e12785a5231
require_sha256 "$runtime_lib_dir/libggml-cpu.so" 9e8c8eef658d5ea18f661ce932566c1870d2defe0c5b9c2b31f0245002d30d9c
require_sha256 "$runtime_lib_dir/libggml-base.so" 1352cf541623e130645a5d4730cd044c472fb851ebb48107173f6aa2c78e191b
require_sha256 "$runtime_lib_dir/libggml.so" bd8cb25cba47bf950247cf270855c9bc1f35eb77c655e0865014dad7851b7ddc
require_sha256 "$runtime_lib_dir/libllama-cli-impl.so" 2091d2889a8a65a23a92c1a374875783c61a0fdf26c355c683161ca9b935de5b
require_sha256 "$runtime_lib_dir/libllama-common.so" b0867f3a78495d0329ff95cf379d5d8e0815e448aaf0a834a7935ac81caceb18
require_sha256 "$runtime_lib_dir/libllama-server-impl.so" 4fc5d3602d2c7ce0e93733c7658349c07cb8bd99f9e4fe72d7e3e15a8908a70d
require_sha256 "$runtime_lib_dir/libmtmd.so" fd20155baa1583971686b86b34435c59579b928924dd7d6f3459f112e7f4f830
require_sha256 "$model" 31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34
require_sha256 "$sidecar" a4cec7372ac21a0093d047c1452277508583ebca1e3f4f66c73a8a74142ccf48
require_sha256 "$one_token" 8205a610522110f839787bdc82b36402ce85582929f076f9e52a6e45908231d6
require_sha256 "$eight_token" 6300e5ab7c8e6fec83b720dff9ed685986158d47c9ebb9aa3d50f8efc0fe7088

"$python_bin" - "$one_token" "$eight_token" <<'PY'
import json
from pathlib import Path
import sys

for expected_tokens, path in zip((1, 8), map(Path, sys.argv[1:])):
    receipt = json.loads(path.read_text())
    if (
        receipt.get("model") != "Qwen/Qwen3.8-27B"
        or receipt.get("tokens") != expected_tokens
        or receipt.get("observed_task_mask") != 15
        or receipt.get("output_bit_exact_vs_direct_v47") is not True
        or receipt.get("performance_timing_ran") is not False
        or receipt.get("passed") is not True
        or receipt.get("complete") is not True
        or receipt.get("terminal") is not True
    ):
        raise SystemExit("v50 standalone algebra receipt changed")
PY

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
  "$receipt" "$stdout_path" "$stderr_path" "$status" "$repo" \
  "$preregistration" "$cli" "$standalone" "$runtime_lib_dir" "$model" \
  "$sidecar" "$one_token" "$eight_token" "$prompt" <<'PY'
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
    receipt_path, stdout_path, stderr_path, exit_code, repo, preregistration,
    cli, standalone, runtime_lib_dir, model, sidecar, one_token, eight_token,
    prompt,
) = sys.argv[1:]
receipt_path = Path(receipt_path)
stdout_path = Path(stdout_path)
stderr_path = Path(stderr_path)
repo = Path(repo)
runtime_lib_dir = Path(runtime_lib_dir)
stdout_bytes = stdout_path.read_bytes()
stderr_text = stderr_path.read_text(errors="replace")
performance_markers = (
    "llama_perf_context_print",
    "eval time =",
    "prompt eval time =",
    "tokens per second",
)
performance_timing_ran = any(marker in stderr_text for marker in performance_markers)
passed = int(exit_code) == 0 and bool(stdout_bytes.strip()) and not performance_timing_ran
payload = {
    "schema": "qwen38_27b_aff3_llamacpp_model_smoke_v50",
    "model": "Qwen/Qwen3.8-27B",
    "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "repository_commit": subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip(),
    "preregistration_sha256": sha256(Path(preregistration)),
    "inputs": {
        "llama_cli_sha256": sha256(Path(cli)),
        "standalone_smoke_sha256": sha256(Path(standalone)),
        "standalone_receipts_sha256": {
            "one_token": sha256(Path(one_token)),
            "eight_token": sha256(Path(eight_token)),
        },
        "runtime_library_sha256": {
            name: sha256(runtime_lib_dir / name)
            for name in (
                "libllama.so", "libggml-cpu.so", "libggml-base.so",
                "libggml.so", "libllama-cli-impl.so", "libllama-common.so",
                "libllama-server-impl.so", "libmtmd.so",
            )
        },
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
        "preview": stdout_bytes.decode(errors="replace")[:512],
    },
    "stderr": {
        "path": str(stderr_path),
        "bytes": stderr_path.stat().st_size,
        "sha256": sha256(stderr_path),
    },
    "repair": {
        "classification": "callback_userdata_lifetime_violation",
        "owner": "llm_graph_result",
        "v49_failure_preserved": True,
    },
    "claim_boundary": (
        "This repair smoke can establish only lifetime-safe bounded free "
        "generation through the actual Qwen3.8-27B graph with one unchanged "
        "AFF3 projection. It is not trained-v44, quality, throughput, memory, "
        "energy, novelty, or scientific-breakthrough evidence."
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
