"""Verify the self-contained GemLite repo loads via from_pretrained + trust_remote_code and renders.

ENV: REPO=clark-air-sana-1.6b-gemlite-2bit OUT_RENDER=render_gemlite_repo.png  (run on CUDA)
"""
import os
import torch
os.environ["HF_HUB_OFFLINE"] = "1"
from diffusers import DiffusionPipeline
from PIL import Image

REPO = os.environ.get("REPO", "clark-air-sana-1.6b-gemlite-2bit")
OUT = os.environ.get("OUT_RENDER", "render_gemlite_repo.png")
K = int(os.environ.get("STEPS", "20")); GUID = float(os.environ.get("GUID", "4.5"))
SEED = int(os.environ.get("SEED", "9000"))
PROMPTS = [
    "a corgi astronaut floating in space, photorealistic, 8k",
    "a half-empty bottle of red wine on a wooden table, soft light",
    "emoji of a pixelart heart, high quality",
    "a little poodle puppy, photorealistic, 8k, close up",
]

pipe = DiffusionPipeline.from_pretrained(REPO, trust_remote_code=True, torch_dtype=torch.bfloat16)
pipe.to("cuda")
print("[pipe]", type(pipe).__name__, "| transformer", type(pipe.transformer).__name__, flush=True)

# confirm the trunk is actually running GemLite kernels
ng = sum(1 for _, m in pipe.transformer.named_modules() if type(m).__name__ in ("_GemliteLin", "_GemliteConv1x1"))
print(f"[gemlite] {ng} GemLite trunk modules live", flush=True)
assert ng > 0, "no GemLite modules found — load path broken"

imgs = [pipe(prompt=p, num_inference_steps=K, height=512, width=512, guidance_scale=GUID,
             generator=torch.Generator("cuda").manual_seed(SEED + j)).images[0]
        for j, p in enumerate(PROMPTS)]
grid = Image.new("RGB", (512 * len(imgs), 512), (20, 20, 20))
for j, im in enumerate(imgs): grid.paste(im, (512 * j, 0))
grid.save(OUT)
print(f"[render] {OUT}", flush=True)
