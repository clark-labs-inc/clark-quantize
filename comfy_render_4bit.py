"""ComfyUI render using the 4-bit (bnb) Gemma text encoder — verify quality at ~2GB vs 9.8GB."""
import sys, types, importlib.util, importlib.machinery
import numpy as np
import torch
from PIL import Image

for n in ("torchaudio", "torchaudio.transforms", "torchaudio.functional"):
    mod = types.ModuleType(n); mod.__spec__ = importlib.machinery.ModuleSpec(n, loader=None); mod.__path__ = []
    sys.modules.setdefault(n, mod)
sys.path.insert(0, "/workspace/ComfyUI")
sys.path.insert(0, "/workspace/ComfyUI/custom_nodes")

import nodes as comfy_nodes
from ComfyUI_ExtraModels.Gemma.nodes import GemmaTextEncode
from ComfyUI_ExtraModels.VAE.nodes import ExtraVAELoader

NODE_DIR = "/workspace/ComfyUI/custom_nodes/ComfyUI-ClarkAirSana"
spec = importlib.util.spec_from_file_location("ComfyUI_ClarkAirSana", NODE_DIR + "/__init__.py", submodule_search_locations=[NODE_DIR])
m = importlib.util.module_from_spec(spec); sys.modules["ComfyUI_ClarkAirSana"] = m; spec.loader.exec_module(m)
ClarkAirSanaLoader = m.NODE_CLASS_MAPPINGS["ClarkAirSanaLoader"]

GEMMA4 = "/workspace/ComfyUI/models/text_encoders/unsloth-gemma-2-2b-it-bnb-4bit"
PROMPT = "a corgi astronaut floating in space, photorealistic, 8k"

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(GEMMA4); tok.padding_side = "right"
tem = AutoModelForCausalLM.from_pretrained(GEMMA4, device_map={"": 0})  # bnb 4-bit, on GPU
gemma = {"tokenizer": tok, "text_encoder": tem.get_decoder(), "text_encoder_model": tem}
pos = GemmaTextEncode().encode(PROMPT, gemma)[0]
neg = GemmaTextEncode().encode("", gemma)[0]
del gemma, tem; torch.cuda.empty_cache()
print("[cond] encoded with 4-bit gemma", flush=True)

model = ClarkAirSanaLoader().load("clark_air_sana_gemlite_comfy.safetensors", "SanaMS_1600M_P1_D20", "bfloat16")[0]
latent = {"samples": torch.zeros([1, 32, 16, 16])}
out = comfy_nodes.KSampler().sample(model, 9000, 20, 4.5, "euler", "normal", pos, neg, latent, 1.0)[0]
print("[ksampler] done", flush=True)

vae = ExtraVAELoader().load_vae("mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers", "dcae-f32c32-sana-1.1-diffusers", "bfloat16")[0]
img = comfy_nodes.VAEDecode().decode(vae, out)[0]
arr = (img[0].detach().float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(arr).save("/workspace/clark-quantize/comfy_render_4bitgemma.png")
print("[saved] comfy_render_4bitgemma.png", flush=True)
