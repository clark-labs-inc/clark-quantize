"""Quantize a Sana diffusion transformer with Clark-Quantize and generate images.

Examples
--------
  # full-precision baseline
  python scripts/quantize_sana.py --mode fp16 --tag fp16

  # best data-free ternary (optimal scale, group 32)
  python scripts/quantize_sana.py --method optimal --group 32 --tag opt_g32

  # absmean baseline (the naive 1.58-bit default -> noise)
  python scripts/quantize_sana.py --method absmean --group 128 --tag absmean
"""
import argparse
import torch
from diffusers import SanaPipeline

from clark_quantize import quantize_module

PROMPTS = [
    "a red panda wearing a tiny astronaut helmet, studio photo, sharp focus",
    "a bowl of ramen on a wooden table, steam rising, warm light, photoreal",
    "an oil painting of a lighthouse on a cliff at sunset, dramatic sky",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Efficient-Large-Model/Sana_1600M_512px_diffusers")
    ap.add_argument("--mode", default="quant", choices=["fp16", "quant"])
    ap.add_argument("--method", default="optimal", choices=["optimal", "absmean"])
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--error-feedback", action="store_true")
    ap.add_argument("--row-gain", action="store_true")
    ap.add_argument("--keep-blocks", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    pipe = SanaPipeline.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    pipe.vae.to(torch.float32)                      # DC-AE decoder prefers fp32

    if a.mode == "quant":
        quantize_module(pipe.transformer, group=a.group, method=a.method,
                        error_feedback=a.error_feedback, row_gain=a.row_gain,
                        keep_blocks=a.keep_blocks)

    pipe.enable_model_cpu_offload()                 # fit modest GPUs (<=24GB)

    import os
    os.makedirs(a.out, exist_ok=True)
    for i, prompt in enumerate(PROMPTS):
        img = pipe(prompt=prompt, height=512, width=512,
                   num_inference_steps=a.steps, guidance_scale=a.guidance,
                   generator=torch.Generator("cuda").manual_seed(a.seed)).images[0]
        path = f"{a.out}/{a.tag}_{i}.png"
        img.save(path)
        print("saved", path)


if __name__ == "__main__":
    main()
