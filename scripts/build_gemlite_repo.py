"""Assemble a self-contained, `from_pretrained`-loadable Clark Air Sana 1.6B GemLite INT2 repo.

Builds the *faithful* gemlite pack by re-quantizing the trained MASTER weights at each layer's
largest power-of-two group (g64 attn / g32 FFN) — no double-quant. Bundles text_encoder + vae
(bf16) + tokenizer + scheduler + a custom transformer class so that:

    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(REPO, trust_remote_code=True).to("cuda")
    img = pipe("a corgi astronaut", num_inference_steps=20).images[0]

renders end-to-end on real GemLite INT2 kernels.  (run on CUDA)

ENV: STATE=<master-weight .pt>  MODEL=models/Sana_1600M_512px  OUT=clark-air-sana-1.6b-gemlite-2bit
"""
import json, os, shutil
import numpy as np, torch, torch.nn as nn
os.environ["HF_HUB_OFFLINE"] = "1"
from diffusers import SanaPipeline
from safetensors.torch import save_file

STATE = os.environ.get("STATE", "state_sana_t112_air_teacher_behavior_s43679_v1.pt")
MODEL = os.environ.get("MODEL", "models/Sana_1600M_512px")
OUT = os.environ.get("OUT", "clark-air-sana-1.6b-gemlite-2bit")
HERE = os.path.dirname(os.path.abspath(__file__))


def is_trunk_lin(n, m):
    return isinstance(m, nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) \
        and any(s in n for s in ("to_q", "to_k", "to_v", "to_out.0"))
def is_trunk_conv(n, m):
    return isinstance(m, nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def predicate(n, m): return is_trunk_lin(n, m) or is_trunk_conv(n, m)


def pow2_group(inn, cap=128):
    g = cap
    while g >= 8:
        if inn % g == 0: return g
        g //= 2
    return 8


@torch.no_grad()
def mse_ternary(W, g):  # W [out,in] fp32 -> codes {-1,0,1} [out,in], scales [out, in/g]
    out, inn = W.shape
    Wg = W.reshape(out, -1, g); A = Wg.abs()
    a, _ = A.sort(-1, descending=True)
    cs = a.cumsum(-1); ks = torch.arange(1, g + 1, device=W.device, dtype=torch.float32)
    k = (cs * cs / ks).argmax(-1, keepdim=True)
    s = cs.gather(-1, k) / (k + 1)
    t = torch.sign(Wg) * (A >= a.gather(-1, k)).float()
    return t.reshape(out, -1), s.squeeze(-1)


def pack2b(code):  # int8/float [out,in] in {-1,0,1} -> uint8 [out, in/4] (levels code+1, 4/byte)
    lev = (code + 1).to(torch.uint8).cpu().numpy()
    return (lev[:, 0::4] | (lev[:, 1::4] << 2) | (lev[:, 2::4] << 4) | (lev[:, 3::4] << 6)).astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"[load] base pipeline {MODEL} (bf16)", flush=True)
    pipe = SanaPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    tf = pipe.transformer
    base_sd = tf.state_dict()

    print(f"[load] master-weight state {STATE}", flush=True)
    st = torch.load(STATE, map_location="cpu", weights_only=False, mmap=True)
    tf_state = st["transformer"] if "transformer" in st else st

    trunk = {n for n, m in tf.named_modules() if predicate(n, m)}
    trunk_w = {n + ".weight" for n in trunk}

    pack = {}
    # 1) islands: trained non-trunk tensors (norms/embeds/modulation/depthwise-conv/biases) as bf16
    n_isl = 0
    for k, v in tf_state.items():
        if k.endswith(".weight_master") or k.endswith(".log_scales"): continue
        if k in trunk_w: continue  # trunk weights become gemlite codes, not stored
        if k in base_sd and base_sd[k].shape == v.shape:
            pack[k] = v.to(torch.bfloat16).contiguous(); n_isl += 1
    print(f"[islands] {n_isl} bf16 tensors", flush=True)

    # 2) trunk: re-quantize master weights at per-layer pow2 group -> 2-bit codes + fp16 scales
    by_g = {}
    for n in sorted(trunk):
        W = tf_state[n + ".weight_master"].float()
        if W.dim() == 4: W = W.reshape(W.shape[0], -1)  # 1x1 conv -> matrix
        out, inn = W.shape; g = pow2_group(inn)
        code, scale = mse_ternary(W.cuda(), g)
        pack[n + "::shape"] = torch.tensor([out, inn], dtype=torch.int32)
        pack[n + "::codes2b"] = torch.from_numpy(pack2b(code))
        pack[n + "::scale"] = scale.to(torch.float16).cpu().contiguous()
        by_g[g] = by_g.get(g, 0) + 1
    print(f"[gemlite] {len(trunk)} trunk modules packed | groups {dict(sorted(by_g.items()))}", flush=True)

    # --- write transformer/ ---
    tdir = os.path.join(OUT, "transformer"); os.makedirs(tdir, exist_ok=True)
    save_file(pack, os.path.join(tdir, "gemlite_pack.safetensors"))
    cfg = dict(tf.config)
    cfg["_class_name"] = "ClarkAirSanaGemliteTransformer"
    for k in list(cfg):
        if k.startswith("_") and k not in ("_class_name", "_diffusers_version"): cfg.pop(k)
    json.dump(cfg, open(os.path.join(tdir, "config.json"), "w"), indent=2)
    shutil.copy(os.path.join(HERE, "clark_air_gemlite", "transformer_clark_gemlite.py"),
                os.path.join(tdir, "transformer_clark_gemlite.py"))

    # --- bundle the rest of the pipeline (bf16) ---
    print("[bundle] text_encoder + vae (bf16) + tokenizer + scheduler", flush=True)
    pipe.text_encoder.to(torch.bfloat16).save_pretrained(os.path.join(OUT, "text_encoder"))
    pipe.vae.to(torch.bfloat16).save_pretrained(os.path.join(OUT, "vae"))
    pipe.tokenizer.save_pretrained(os.path.join(OUT, "tokenizer"))
    pipe.scheduler.save_pretrained(os.path.join(OUT, "scheduler"))

    # --- model_index.json: transformer -> custom remote-code component ---
    mi = {
        "_class_name": "SanaPipeline",
        "_diffusers_version": cfg.get("_diffusers_version", "0.38.0"),
        "scheduler": ["diffusers", "DPMSolverMultistepScheduler"],
        "text_encoder": ["transformers", "Gemma2Model"],
        "tokenizer": ["transformers", "GemmaTokenizerFast"],
        "transformer": ["transformer_clark_gemlite", "ClarkAirSanaGemliteTransformer"],
        "vae": ["diffusers", "AutoencoderDC"],
    }
    json.dump(mi, open(os.path.join(OUT, "model_index.json"), "w"), indent=2)
    print(f"[done] repo at {OUT}", flush=True)


if __name__ == "__main__":
    main()
