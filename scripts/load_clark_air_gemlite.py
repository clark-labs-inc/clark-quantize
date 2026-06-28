"""Self-contained loader for the Clark Air Sana GemLite INT2 artifact: reads the packed
safetensors (2-bit ternary codes + scales + FP8 islands), builds the GemLite INT2 Sana
transformer, and renders. No master-weight/state file needed.

ENV: PACKED=clark_air_sana_gemlite_2bit.safetensors OUT_RENDER=render.png PROMPTS=shared_hard.json (run on CUDA)
"""
import json, os
import numpy as np, torch, torch.nn as nn
os.environ["HF_HUB_OFFLINE"] = "1"
from diffusers import SanaPipeline
from safetensors.torch import load_file
from gemlite import GemLiteLinear, DType
from PIL import Image

PACKED = os.environ.get("PACKED", "clark_air_sana_gemlite_2bit.safetensors")
MODEL = os.environ.get("MODEL", "models/Sana_1600M_512px")
OUT_RENDER = os.environ.get("OUT_RENDER", "render_gemlite.png")
K = int(os.environ.get("STEPS", "20")); GUID = float(os.environ.get("GUID", "4.5"))
NREND = int(os.environ.get("NREND", "4")); SEED0 = int(os.environ.get("SEED", "9000"))
DEV = "cuda"


def is_trunk_lin(n, m):
    return isinstance(m, nn.Linear) and n.startswith("transformer_blocks") and (".attn1." in n or ".attn2." in n) \
        and any(s in n for s in ("to_q", "to_k", "to_v", "to_out.0"))
def is_trunk_conv(n, m):
    return isinstance(m, nn.Conv2d) and n.startswith("transformer_blocks") and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
def predicate(n, m): return is_trunk_lin(n, m) or is_trunk_conv(n, m)


class GLin(nn.Module):
    def __init__(self, gl): super().__init__(); self.gl = gl
    def forward(self, x):
        s = x.shape; return self.gl(x.reshape(-1, s[-1]).half()).reshape(*s[:-1], -1).to(x.dtype)
class GConv(nn.Module):
    def __init__(self, gl): super().__init__(); self.gl = gl
    def forward(self, x):
        B, C, H, W = x.shape
        y = self.gl(x.permute(0, 2, 3, 1).reshape(-1, C).half())
        return y.reshape(B, H, W, -1).permute(0, 3, 1, 2).to(x.dtype)


def unpack2b(packed, out, inn):  # uint8 [out, in/4] -> ternary int8 [out,in]
    b = packed.astype(np.uint8); lev = np.empty((out, inn), np.int8)
    lev[:, 0::4] = b & 3; lev[:, 1::4] = (b >> 2) & 3; lev[:, 2::4] = (b >> 4) & 3; lev[:, 3::4] = (b >> 6) & 3
    return lev.astype(np.int8) - 1


def main():
    sd = load_file(PACKED)

    def island(name):  # -> bf16 tensor or None
        if name not in sd: return None
        v = sd[name]
        if name + "::fp8shape" in sd:
            shp = tuple(int(x) for x in sd[name + "::fp8shape"].tolist())
            return v.view(torch.float8_e4m3fn).reshape(shp).to(torch.bfloat16)
        return v.to(torch.bfloat16)

    pipe = SanaPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16); pipe.to(DEV); tf = pipe.transformer
    base = tf.state_dict()
    for k in list(sd.keys()):
        if "::" in k: continue
        isl = island(k)
        if k in base and isl is not None and base[k].shape == isl.shape:
            base[k].copy_(isl.to(base[k].dtype))
    tf.load_state_dict(base, strict=False)

    for n, m in tf.named_modules():
        if not predicate(n, m): continue
        out_, inn = [int(x) for x in sd[n + "::shape"].tolist()]
        code = unpack2b(sd[n + "::codes2b"].numpy(), out_, inn)
        scale = sd[n + "::scale"].float(); ng = scale.shape[1]; g = inn // ng
        Wq = torch.from_numpy((code + 1).astype(np.uint8)).to(DEV)
        b = island(n + ".bias")
        gl = GemLiteLinear(W_nbits=2, group_size=g, in_features=inn, out_features=out_, input_dtype=DType.FP16, output_dtype=DType.FP16)
        gl.pack(Wq, scale.to(DEV), torch.ones(out_, ng, device=DEV), bias=(b.to(DEV).half() if b is not None else None), fma_mode=False)
        parent = tf.get_submodule(n.rsplit(".", 1)[0]); parent.add_module(n.rsplit(".", 1)[1], GConv(gl) if isinstance(m, nn.Conv2d) else GLin(gl))
    print("[loaded] GemLite INT2 transformer built from packed safetensors", flush=True)

    prompts = json.load(open(os.environ.get("PROMPTS", "shared_hard.json")))[:NREND]
    imgs = [pipe(prompt=p, num_inference_steps=K, height=512, width=512, guidance_scale=GUID,
                 generator=torch.Generator(DEV).manual_seed(SEED0 + j)).images[0] for j, p in enumerate(prompts)]
    grid = Image.new("RGB", (512 * len(imgs), 512), (20, 20, 20))
    for j, im in enumerate(imgs): grid.paste(im, (512 * j, 0))
    grid.save(OUT_RENDER); print(f"[render] {OUT_RENDER}", flush=True)


if __name__ == "__main__":
    main()
