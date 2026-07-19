"""Smoke test: load Clark Air Sana via our actual ComfyUI node, run the injected SanaMS forward."""
import importlib.util, sys, types
import torch

# Test-harness only: this minimal venv lacks torchaudio, so ExtraModels' eager __init__
# (which imports its audio VAE) would fail. Stub the parent package so only the Sana
# submodule loads. A real ComfyUI env imports ExtraModels normally (torchaudio present).
_em = types.ModuleType("ComfyUI_ExtraModels")
_em.__path__ = ["/workspace/ComfyUI/custom_nodes/ComfyUI_ExtraModels"]
sys.modules["ComfyUI_ExtraModels"] = _em

NODE_DIR = "/workspace/ComfyUI/custom_nodes/ComfyUI-ClarkAirSana"
spec = importlib.util.spec_from_file_location(
    "ComfyUI_ClarkAirSana", NODE_DIR + "/__init__.py", submodule_search_locations=[NODE_DIR])
mod = importlib.util.module_from_spec(spec)
sys.modules["ComfyUI_ClarkAirSana"] = mod
spec.loader.exec_module(mod)

loader = mod.NODE_CLASS_MAPPINGS["ClarkAirSanaLoader"]()
(patcher,) = loader.load("clark_air_sana_gemlite_comfy.safetensors", "SanaMS_1600M_P1_D20", "bfloat16")
dm = patcher.model.diffusion_model
ng = sum(1 for _, m in dm.named_modules() if type(m).__name__ in ("_GemliteLin", "_GemliteConv1x1"))
print(f"[loaded] {type(patcher).__name__} | diffusion_model {type(dm).__name__} | GemLite modules: {ng}")
assert ng == 140, f"expected 140 GemLite modules, got {ng}"

x = torch.randn(1, 32, 16, 16, device="cuda", dtype=torch.bfloat16)
t = torch.tensor([500.0], device="cuda")
ctx = torch.randn(1, 1, 300, 2304, device="cuda", dtype=torch.bfloat16)  # [B,1,L,C]
mask = torch.ones(1, 300, device="cuda")
with torch.no_grad():
    out = dm(x, t, ctx, mask=mask)
print(f"[forward] out {tuple(out.shape)} {out.dtype} finite={torch.isfinite(out).all().item()}")
assert tuple(out.shape) == (1, 32, 16, 16) and torch.isfinite(out).all().item()
print("[OK] node loads + GemLite trunk runs under ComfyUI's SanaMS forward")
