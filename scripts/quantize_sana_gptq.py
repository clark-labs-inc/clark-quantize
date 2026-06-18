"""Activation-aware ternary quantization of Sana with GPTQ (+ optional rotation),
self-calibrated on the model's own generations. Reaches near-FP16 at 2.1 bits.

  python scripts/quantize_sana_gptq.py --group 128 --rotate --tag gptqrot_g128
"""
import argparse, os, time, torch, torch.nn as nn
from diffusers import SanaPipeline

from clark_quantize import HessianCollector, apply_gptq, random_orthogonal
from clark_quantize.ternary import DEFAULT_PROTECT

ROT_DIM = 2240                       # dominant linear input width in Sana 1.6B
EVAL = ["a red panda wearing a tiny astronaut helmet, studio photo, sharp focus",
        "a bowl of ramen on a wooden table, steam rising, warm light, photoreal",
        "an oil painting of a lighthouse on a cliff at sunset, dramatic sky"]
CALIB = ["a golden retriever puppy running on a beach at sunrise",
         "a steaming cup of coffee next to an open book, cozy cafe",
         "a futuristic city skyline at night, neon lights, rain",
         "a still life of fruit in a ceramic bowl, soft window light",
         "a snowy mountain village, wooden chalets, blue hour",
         "a portrait of an elderly fisherman, weathered face, detailed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Efficient-Large-Model/Sana_1600M_512px_diffusers")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--rotate", action="store_true")
    ap.add_argument("--calib-steps", type=int, default=8)
    ap.add_argument("--out", default="out")
    ap.add_argument("--tag", default="gptqrot_g128")
    a = ap.parse_args()
    dev = "cuda"

    pipe = SanaPipeline.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    pipe.vae.to(torch.float32)
    pipe.enable_model_cpu_offload()
    tf = pipe.transformer
    targets = {n: m for n, m in tf.named_modules()
               if isinstance(m, nn.Linear) and not any(p in n for p in DEFAULT_PROTECT)}

    print(f"[calib] {len(CALIB)} self-generated prompts ...")
    t0 = time.time()
    with HessianCollector(targets, dev) as hc:
        for p in CALIB:
            pipe(prompt=p, height=512, width=512, num_inference_steps=a.calib_steps,
                 guidance_scale=4.5, generator=torch.Generator(dev).manual_seed(123)).images
    print(f"[calib] {time.time()-t0:.0f}s")

    R = random_orthogonal(ROT_DIM, dev) if a.rotate else None
    t0 = time.time()
    info = apply_gptq(targets, hc.H, group=a.group, rotation=R, rot_dim=ROT_DIM)
    print(f"[gptq] {time.time()-t0:.0f}s  {info}")

    os.makedirs(a.out, exist_ok=True)
    for i, p in enumerate(EVAL):
        img = pipe(prompt=p, height=512, width=512, num_inference_steps=20,
                   guidance_scale=4.5, generator=torch.Generator(dev).manual_seed(0)).images[0]
        img.save(f"{a.out}/{a.tag}_{i}.png")
        print("saved", f"{a.out}/{a.tag}_{i}.png")


if __name__ == "__main__":
    main()
