"""Quantize the Sana 1.6B Gemma-2 text encoder to bitsandbytes NF4 (4-bit) and save.

The quantization_config is embedded in the saved config.json, so diffusers' pipeline loader
reconstructs the 4-bit model automatically (requires bitsandbytes + CUDA).

ENV: BASE_TE=models/Sana_1600M_512px/text_encoder  OUT_TE=<dir>   (run on CUDA)
"""
import os, torch
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import Gemma2Model, BitsAndBytesConfig

BASE = os.environ.get("BASE_TE", "models/Sana_1600M_512px/text_encoder")
OUT = os.environ["OUT_TE"]

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
te = Gemma2Model.from_pretrained(BASE, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0})
te.save_pretrained(OUT)
print("[nf4] saved", OUT, flush=True)
