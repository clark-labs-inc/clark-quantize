"""End-to-end ComfyUI render: real Gemma text encode -> our GemLite Sana under KSampler -> DC-AE decode."""
import sys, types, importlib.util, importlib.machinery
import numpy as np
import torch
from PIL import Image

# torchaudio has no torch-2.12 build; stub it (ComfyUI's audio VAE is unused for Sana).
# Needs a valid __spec__ so transformers' importlib.util.find_spec check doesn't raise.
for n in ("torchaudio", "torchaudio.transforms", "torchaudio.functional"):
    mod = types.ModuleType(n)
    mod.__spec__ = importlib.machinery.ModuleSpec(n, loader=None)
    mod.__path__ = []
    sys.modules.setdefault(n, mod)
sys.path.insert(0, "/workspace/ComfyUI")
sys.path.insert(0, "/workspace/ComfyUI/custom_nodes")

import nodes as comfy_nodes  # ComfyUI core nodes
from ComfyUI_ExtraModels.Gemma.nodes import GemmaLoader, GemmaTextEncode
from ComfyUI_ExtraModels.VAE.nodes import ExtraVAELoader
from ComfyUI_ExtraModels.Sana.nodes import EmptySanaLatentImage

NODE_DIR = "/workspace/ComfyUI/custom_nodes/ComfyUI-ClarkAirSana"
spec = importlib.util.spec_from_file_location("ComfyUI_ClarkAirSana", NODE_DIR + "/__init__.py", submodule_search_locations=[NODE_DIR])
m = importlib.util.module_from_spec(spec); sys.modules["ComfyUI_ClarkAirSana"] = m; spec.loader.exec_module(m)
ClarkAirSanaLoader = m.NODE_CLASS_MAPPINGS["ClarkAirSanaLoader"]

GEMMA_DIR = "/workspace/ComfyUI/models/text_encoders/models-efficient-large-model--gemma-2-2b-it"
PROMPT = "a corgi astronaut floating in space, photorealistic, 8k"
SAMPLER = sys.argv[1] if len(sys.argv) > 1 else "euler"
SCHED = sys.argv[2] if len(sys.argv) > 2 else "normal"

# 1) text conditioning via the real GemmaTextEncode, then free Gemma.
# Build GemmaLoader's exact GEMMA dict from our local download (its loader hardcodes repo IDs).
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(GEMMA_DIR); tok.padding_side = "right"
tem = AutoModelForCausalLM.from_pretrained(GEMMA_DIR, torch_dtype=torch.bfloat16)
gemma = {"tokenizer": tok, "text_encoder": tem.get_decoder().to("cuda"), "text_encoder_model": tem}
pos = GemmaTextEncode().encode(PROMPT, gemma)[0]
neg = GemmaTextEncode().encode("", gemma)[0]
del gemma, tem; torch.cuda.empty_cache()
print("[cond] encoded pos/neg, gemma freed", flush=True)

# 2) our GemLite Sana model + empty latent
model = ClarkAirSanaLoader().load("clark_air_sana_gemlite_comfy.safetensors", "SanaMS_1600M_P1_D20", "bfloat16")[0]
latent = {"samples": torch.zeros([1, 32, 16, 16])}  # 512x512 / DC-AE 32x
print("[model] loaded + latent", flush=True)

# 3) native KSampler
out = comfy_nodes.KSampler().sample(model, 9000, 20, 4.5, SAMPLER, SCHED, pos, neg, latent, 1.0)[0]
print(f"[ksampler] {SAMPLER}/{SCHED} done, latent {tuple(out['samples'].shape)}", flush=True)

# 4) DC-AE decode -> image
vae = ExtraVAELoader().load_vae("mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers", "dcae-f32c32-sana-1.1-diffusers", "bfloat16")[0]
img = comfy_nodes.VAEDecode().decode(vae, out)[0]
arr = (img[0].detach().float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(arr).save("/workspace/clark-quantize/comfy_render.png")
print("[saved] comfy_render.png", flush=True)
