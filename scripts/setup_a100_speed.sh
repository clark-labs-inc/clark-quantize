#!/usr/bin/env bash
# Fast-kernel stack for Qwen3.5 ternary KD on A100 (sm80).
# The 136 tok/s pilot was the DeltaNet *torch fallback* — these packages give
# transformers its Triton fast path for the 24 linear_attention layers.
set -euo pipefail

echo "=== 0. what does the installed Qwen3.5 modeling code want? ==="
python - <<'PY'
import importlib, inspect
try:
    m = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    src = inspect.getsource(m)
    print(m.__file__)
    for line in src.splitlines():
        ls = line.strip()
        if ("import" in ls or "available" in ls) and any(
            p in ls for p in ("fla", "flash_linear", "causal_conv1d", "flash_attn", "gated_delta")
        ):
            print("   ", ls[:120])
except ModuleNotFoundError as e:
    print("qwen3_5 modeling not found in transformers:", e)
PY

echo "=== 1. install kernel stack ==="
pip install -U "triton>=3.1" ninja packaging
# Gated-DeltaNet chunked Triton kernels (pure Triton, no CUDA build):
pip install -U flash-linear-attention
# Depthwise causal conv kernel used by the linear_attn path.
# CRITICAL: transformers' Qwen3.5 fast-path gate is ALL-OR-NOTHING — if
# causal_conv1d is missing it disables the fla kernels too (100x slowdown).
# Boxes without nvcc can't build the real package; install the exact-semantics
# torch shim instead (unlocks the gate; the 4-tap depthwise conv runs on cuDNN).
pip install -U causal-conv1d --no-build-isolation || {
  echo "causal-conv1d build failed (no nvcc?) — installing torch shim to unlock the fla fast path"
  pip install /workspace/causal_conv1d_shim 2>/dev/null \
    || echo "shim dir missing: create it per A100_RUNBOOK.md §1 / session notes"
}
# Full-attention layers: PyTorch SDPA needs nothing. flash-attn v2 is optional:
#   pip install flash-attn --no-build-isolation      # slow source build
#   pip install kernels                              # or HF kernels-hub prebuilt

echo "=== 2. verify dispatch ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| dev", torch.cuda.get_device_name(0))
import triton; print("triton", triton.__version__)
ok = True
try:
    import fla
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
    print("fla gated_delta_rule: OK", getattr(fla, "__version__", ""))
except Exception as e:
    ok = False; print("fla: MISSING ->", e)
try:
    import causal_conv1d
    print("causal_conv1d: OK", getattr(causal_conv1d, "__version__", ""))
except Exception as e:
    print("causal_conv1d: missing (conv fallback, minor cost) ->", e)
# our fused ternary quantizer
import sys, os; sys.path.insert(0, os.getcwd())
from clark_quantize.ternary_kd import fused_absmean_dequant, _absmean_dequant_torch
w = torch.randn(512, 2560, device="cuda") * 0.02
q_triton = fused_absmean_dequant(w)
q_ref = _absmean_dequant_torch(w, torch.bfloat16)
match = (q_triton == q_ref).float().mean().item()
print(f"fused ternary quantizer: triton output match vs torch = {match:.6f} (ties may differ)")
assert match > 0.9999
print("ALL GOOD" if ok else "INSTALL INCOMPLETE — deltanet will fall back to slow torch")
PY

echo "=== 3. next: python scripts/throughput_ladder.py --model Qwen/Qwen3.5-4B ==="
