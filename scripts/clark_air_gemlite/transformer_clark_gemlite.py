"""Custom diffusers transformer for Clark Air Sana 1.6B — GemLite INT2 (ternary) trunk.

Loaded automatically by `DiffusionPipeline.from_pretrained(repo, trust_remote_code=True)`:
the repo's `model_index.json` names the transformer component
`["transformer_clark_gemlite", "ClarkAirSanaGemliteTransformer"]`, and diffusers imports
this file from the `transformer/` subfolder.

The attention + GLU-FFN trunk runs on real GemLite INT2 kernels (ternary {-1,0,+1} packed in
2-bit slots, per-group fp scales). Islands (embeds/norms/modulation/depthwise conv) stay bf16.

Requires CUDA + the `gemlite` package. The packed weights live in `gemlite_pack.safetensors`.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from diffusers import SanaTransformer2DModel


# --- which trunk modules were quantized (must match the build script) ---
def _is_trunk_lin(n, m):
    return (
        isinstance(m, nn.Linear)
        and n.startswith("transformer_blocks")
        and (".attn1." in n or ".attn2." in n)
        and any(s in n for s in ("to_q", "to_k", "to_v", "to_out.0"))
    )


def _is_trunk_conv(n, m):
    return (
        isinstance(m, nn.Conv2d)
        and n.startswith("transformer_blocks")
        and (".ff.conv_inverted" in n or ".ff.conv_point" in n)
    )


def _predicate(n, m):
    return _is_trunk_lin(n, m) or _is_trunk_conv(n, m)


# --- shape adapters around the GemLite kernel (it expects 2D fp16 [tokens, in]) ---
class _GemliteLin(nn.Module):
    def __init__(self, gl):
        super().__init__()
        self.gl = gl

    def forward(self, x):
        s = x.shape
        return self.gl(x.reshape(-1, s[-1]).half()).reshape(*s[:-1], -1).to(x.dtype)


class _GemliteConv1x1(nn.Module):  # 1x1 conv as a channel matmul
    def __init__(self, gl):
        super().__init__()
        self.gl = gl

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.gl(x.permute(0, 2, 3, 1).reshape(-1, c).half())
        return y.reshape(b, h, w, -1).permute(0, 3, 1, 2).to(x.dtype)


def _unpack2b(packed, out, inn):  # uint8 [out, in/4] -> ternary int8 [out, in]
    b = packed.astype(np.uint8)
    lev = np.empty((out, inn), np.int8)
    lev[:, 0::4] = b & 3
    lev[:, 1::4] = (b >> 2) & 3
    lev[:, 2::4] = (b >> 4) & 3
    lev[:, 3::4] = (b >> 6) & 3
    return lev.astype(np.int8) - 1


class ClarkAirSanaGemliteTransformer(SanaTransformer2DModel):
    """SanaTransformer2DModel whose trunk runs on GemLite INT2 ternary kernels."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from gemlite import GemLiteLinear, DType

        if not torch.cuda.is_available():
            raise RuntimeError(
                "ClarkAirSanaGemliteTransformer requires a CUDA GPU (GemLite INT2 kernels are "
                "CUDA-only). Use the bf16 'unpacked' repo on CPU/Apple Silicon."
            )
        path = pretrained_model_name_or_path
        sub = kwargs.get("subfolder")
        if sub:
            path = os.path.join(path, sub)
        dtype = kwargs.get("torch_dtype") or kwargs.get("dtype") or torch.bfloat16
        dev = "cuda"

        config = cls.load_config(path)
        model = cls.from_config(config).to(dtype)

        sd = load_file(os.path.join(path, "gemlite_pack.safetensors"))

        # 1) islands (norms/embeds/modulation/depthwise-conv/biases) -> bf16 skeleton
        base = model.state_dict()
        for k, v in sd.items():
            if "::" in k:
                continue
            if k in base and base[k].shape == v.shape:
                base[k].copy_(v.to(base[k].dtype))
        model.load_state_dict(base, strict=False)
        model = model.to(dev)

        # 2) replace trunk modules with packed GemLite INT2 kernels
        for n, m in list(model.named_modules()):
            if not _predicate(n, m):
                continue
            out_, inn = [int(x) for x in sd[n + "::shape"].tolist()]
            code = _unpack2b(sd[n + "::codes2b"].numpy(), out_, inn)
            scale = sd[n + "::scale"].float()
            ng = scale.shape[1]
            g = inn // ng
            wq = torch.from_numpy((code + 1).astype(np.uint8)).to(dev)
            b = getattr(m, "bias", None)
            gl = GemLiteLinear(
                W_nbits=2, group_size=g, in_features=inn, out_features=out_,
                input_dtype=DType.FP16, output_dtype=DType.FP16,
            )
            gl.pack(
                wq, scale.to(dev), torch.ones(out_, ng, device=dev),
                bias=(b.detach().to(dev).half() if b is not None else None), fma_mode=False,
            )
            parent = model.get_submodule(n.rsplit(".", 1)[0])
            wrap = _GemliteConv1x1(gl) if isinstance(m, nn.Conv2d) else _GemliteLin(gl)
            parent.add_module(n.rsplit(".", 1)[1], wrap)

        model._clark_air_gemlite = True
        return model
