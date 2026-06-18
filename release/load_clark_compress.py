"""Load the Clark-Compress Sana-1.6B Mixed artifact and build a runnable pipeline.
The artifact stores the quantized transformer only; the standard Sana text encoder
and VAE are pulled from the base repo.

    pipe = load("path/to/artifact")           # dir with the .safetensors + manifest.json
    pipe("a bowl of ramen, steam rising").images[0].save("out.png")
"""
import json
import torch
from safetensors.torch import load_file
from diffusers import SanaPipeline, SanaTransformer2DModel

BASE = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
G = 128


def _unpack2(codes, scale, out, inf, pad):                # ternary 2-bit -> {-1,0,1}
    c = torch.zeros(out, inf + pad, dtype=torch.uint8)
    c[:, 0::4] = codes & 3; c[:, 1::4] = (codes >> 2) & 3
    c[:, 2::4] = (codes >> 4) & 3; c[:, 3::4] = (codes >> 6) & 3
    t = c.float() - 1
    return (t.reshape(out, -1, G) * scale.float()[..., None]).reshape(out, -1)[:, :inf]


def _unpack4(codes, scale, out, inf, pad):                # 4-bit -> {-7..7}
    c = torch.zeros(out, inf + pad, dtype=torch.uint8)
    c[:, 0::2] = codes & 15; c[:, 1::2] = (codes >> 4) & 15
    q = c.float() - 7
    return (q.reshape(out, -1, G) * scale.float()[..., None]).reshape(out, -1)[:, :inf]


def load(artifact_dir, base=BASE, dtype=torch.bfloat16):
    st = load_file(f"{artifact_dir}/clark_compress_mixed.safetensors")
    man = json.load(open(f"{artifact_dir}/manifest.json"))
    tf = SanaTransformer2DModel.from_pretrained(base, subfolder="transformer", torch_dtype=dtype)

    sd = {n: st[n] for n in man["fp16"]}
    for n, info in man["ternary"].items():
        W = _unpack2(st[f"{n}.codes"], st[f"{n}.scale"], info["out"], info["in"], info["pad"])
        sd[f"{n}.weight"] = W.to(dtype)
    for n, info in man["int4"].items():
        W = _unpack4(st[f"{n}.codes"], st[f"{n}.scale"], info["out"], info["in"], info["pad"])
        sd[f"{n}.weight"] = W.reshape(info["out"], info["in"], 1, 1).to(dtype)   # 1x1 conv
    tf.load_state_dict(sd, strict=True)

    R = st["_rotation"].float()
    def rot_hook(_m, inp): return ((inp[0].float() @ R.t()).to(inp[0].dtype),) + inp[1:]
    for n in man["rot_hooks"]:
        tf.get_submodule(n).register_forward_pre_hook(rot_hook)

    pipe = SanaPipeline.from_pretrained(base, transformer=tf, torch_dtype=dtype)
    pipe.vae.to(torch.float32)
    return pipe


if __name__ == "__main__":
    import sys
    p = load(sys.argv[1] if len(sys.argv) > 1 else ".")
    p.enable_model_cpu_offload()
    p("a red panda wearing a tiny astronaut helmet, studio photo").images[0].save("sample.png")
    print("wrote sample.png")
