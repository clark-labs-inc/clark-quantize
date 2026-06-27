"""Build a GemLite INT2 (ternary in 2-bit slots) Sana transformer from the trained
master weights, and render to verify. GemLite needs power-of-two groups, and Sana's
dims (2240, 5600) are not 128-friendly, so each layer is re-quantized at the largest
power-of-two group that divides its in_features (g64 for 2240, g32 for 5600, ...).

ENV: STATE=state_...g112.pt OUT_RENDER=render_gemlite.png  (run on CUDA)
"""
import json, os, sys
import torch, torch.nn as nn
os.environ["HF_HUB_OFFLINE"] = "1"
from diffusers import SanaPipeline
from PIL import Image
from gemlite import GemLiteLinear, DType

STATE = os.environ.get("STATE", "state_sana_t112_science_fresh_muon_g112.pt")
MODEL = os.environ.get("MODEL", "models/Sana_1600M_512px")
OUT_RENDER = os.environ.get("OUT_RENDER", "render_gemlite.png")
SAVE_DIR = os.environ.get("SAVE_DIR", "")
K = int(os.environ.get("STEPS", "20")); GUID = float(os.environ.get("GUID", "4.5"))
NREND = int(os.environ.get("NREND", "4")); SEED0 = int(os.environ.get("SEED", "9000"))
DEV = "cuda"


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
def mse_ternary(W, g):  # MSE-optimal ternary, group g along in-dim -> codes[out,in], scales[out,ng]
    out, inn = W.shape
    Wg = W.reshape(out, -1, g); A = Wg.abs()
    a, _ = A.sort(-1, descending=True)
    cs = a.cumsum(-1); ks = torch.arange(1, g + 1, device=W.device, dtype=torch.float32)
    k = (cs * cs / ks).argmax(-1, keepdim=True)
    s = cs.gather(-1, k) / (k + 1)
    t = torch.sign(Wg) * (A >= a.gather(-1, k)).float()
    return t.reshape(out, -1), s.squeeze(-1)


def make_gemlite(W, bias):  # W [out,in] fp32 (master) -> GemLiteLinear (ternary@pow2 group)
    out, inn = W.shape; g = pow2_group(inn)
    code, scale = mse_ternary(W, g)
    Wq = (code + 1).to(torch.uint8).to(DEV)
    zeros = torch.ones(out, inn // g, device=DEV)
    gl = GemLiteLinear(W_nbits=2, group_size=g, in_features=inn, out_features=out,
                       input_dtype=DType.FP16, output_dtype=DType.FP16)
    gl.pack(Wq, scale.to(DEV), zeros, bias=(bias.to(DEV).half() if bias is not None else None), fma_mode=False)
    return gl, g


class GemliteLinearWrap(nn.Module):
    def __init__(self, gl): super().__init__(); self.gl = gl
    def forward(self, x):
        s = x.shape; y = self.gl(x.reshape(-1, s[-1]).half()); return y.reshape(*s[:-1], y.shape[-1]).to(x.dtype)


class GemliteConv1x1Wrap(nn.Module):  # 1x1 conv as channel matmul
    def __init__(self, gl): super().__init__(); self.gl = gl
    def forward(self, x):  # x [B,Cin,H,W]
        B, C, H, Wd = x.shape
        xf = x.permute(0, 2, 3, 1).reshape(-1, C).half()
        y = self.gl(xf).reshape(B, H, Wd, -1).permute(0, 3, 1, 2)
        return y.to(x.dtype)


def main():
    st = torch.load(STATE, map_location="cpu", weights_only=False, mmap=True)
    tf_state = st["transformer"]
    pipe = SanaPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16); pipe.to(DEV)
    tf = pipe.transformer
    # 1) load non-trunk (islands/norms/biases/embeds) from trained state
    base_sd = tf.state_dict(); loaded = 0
    for k, v in tf_state.items():
        if k.endswith(".weight_master") or k.endswith(".log_scales"): continue
        if k in base_sd and base_sd[k].shape == v.shape:
            base_sd[k].copy_(v.to(base_sd[k].dtype)); loaded += 1
    tf.load_state_dict(base_sd, strict=False)
    print(f"[islands] loaded {loaded} non-trunk tensors from trained state", flush=True)
    # 2) replace trunk modules with gemlite wrappers built from master weights
    mods = [(n, m) for n, m in tf.named_modules() if predicate(n, m)]
    by_g = {}
    for n, m in mods:
        W = tf_state[n + ".weight_master"].float()
        if W.dim() == 4: W = W.reshape(W.shape[0], -1)
        bias = getattr(m, "bias", None)
        gl, g = make_gemlite(W, bias.detach() if bias is not None else None)
        wrap = GemliteConv1x1Wrap(gl) if isinstance(m, nn.Conv2d) else GemliteLinearWrap(gl)
        parent = tf.get_submodule(n.rsplit(".", 1)[0]); parent.add_module(n.rsplit(".", 1)[1], wrap)
        by_g[g] = by_g.get(g, 0) + 1
    print(f"[gemlite] packed {len(mods)} trunk modules | group histogram: {dict(sorted(by_g.items()))}", flush=True)

    hard = json.load(open(os.environ.get("PROMPTS", "shared_hard.json")))[:NREND]
    imgs = [pipe(prompt=p, num_inference_steps=K, height=512, width=512, guidance_scale=GUID,
                 generator=torch.Generator(DEV).manual_seed(SEED0 + j)).images[0] for j, p in enumerate(hard)]
    grid = Image.new("RGB", (512 * len(imgs), 512), (20, 20, 20))
    for j, im in enumerate(imgs): grid.paste(im, (512 * j, 0))
    grid.save(OUT_RENDER); print(f"[render] {OUT_RENDER} (gemlite-int2 model)", flush=True)


if __name__ == "__main__":
    main()
