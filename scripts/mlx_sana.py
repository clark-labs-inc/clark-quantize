"""Sana 1.6B transformer ported to MLX (Apple Silicon) — first pure-MLX Sana runtime,
the foundation for mlx-2bit / mlx-1bit Clark Air artifacts. Weights are loaded from a
diffusers state dict (keyed by the original PyTorch names). FP16 first (port parity),
then per-layer MLX quantization of the trunk.

Config (Sana 1.6B 512px): dim 2240 = 70 heads x 32; cross 20 heads x 112; 20 layers;
mlp_ratio 2.5 (GLU hidden 5600); in/out 32 latent ch; patch 1; sample 16.
"""
import math
import mlx.core as mx
import mlx.nn as nn

DIM = 2240
HEADS = 70
HEAD_DIM = 32
X_HEADS = 20
X_HEAD_DIM = 112
N_LAYERS = 20
HIDDEN = 5600  # int(2.5 * 2240)
EPS = 1e-6


def timestep_embedding(t, dim=256, max_period=10000):
    # diffusers Timesteps: flip_sin_to_cos=True, downscale_freq_shift=0
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t.astype(mx.float32)[:, None] * freqs[None]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)  # flip_sin_to_cos


def layernorm(x, eps=EPS):  # no affine
    x = x.astype(mx.float32)
    x = (x - x.mean(-1, keepdims=True)) * mx.rsqrt(x.var(-1, keepdims=True) + eps)
    return x


def silu(x):
    return x * mx.sigmoid(x)


def gelu_tanh(x):
    return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


class QLinear:
    """Linear backed by an MLX-quantized weight (or plain fp16 if not quantized)."""
    def __init__(self, w, bias=None, bits=None, group_size=64):
        self.bias = bias
        self.q = None
        if bits is None:
            self.w = w  # [out, in] fp16
        else:
            self.q = mx.quantize(w, group_size=group_size, bits=bits)  # (wq, scales, biases)
            self.gs, self.bits = group_size, bits

    def __call__(self, x):
        if self.q is None:
            y = x @ self.w.T
        else:
            wq, scales, biases = self.q
            y = mx.quantized_matmul(x, wq, scales=scales, biases=biases,
                                    transpose=True, group_size=self.gs, bits=self.bits)
        if self.bias is not None:
            y = y + self.bias
        return y


def pow2_group(inn, cap=128):
    g = cap
    while g >= 32:
        if inn % g == 0:
            return g
        g //= 2
    return 32


class SanaMLX:
    def __init__(self, sd, quant_bits=None):
        # sd: dict of mx.array keyed by diffusers PyTorch names. quant_bits: None|2|1
        self.sd = sd
        self.qbits = quant_bits
        g = lambda k: sd[k]
        self.g = g
        # trunk (quantizable) linear weights, with per-layer pow2 group
        def trunk(k_w, k_b=None):
            w = sd[k_w]
            if w.ndim == 4:  # 1x1 conv -> [out,in]
                w = w.reshape(w.shape[0], w.shape[1])
            b = sd[k_b] if (k_b and k_b in sd) else None
            if quant_bits is None:
                return QLinear(w, b)
            return QLinear(w, b, bits=quant_bits, group_size=pow2_group(w.shape[1]))
        def island(k_w, k_b=None):
            w = sd[k_w]
            if w.ndim == 4:
                w = w.reshape(w.shape[0], w.shape[1])
            return QLinear(w, sd[k_b] if (k_b and k_b in sd) else None)
        self.trunk, self.island = trunk, island

    def block(self, x, enc, temb, H, W, i):
        p = f"transformer_blocks.{i}."
        g = self.g
        ss = g(p + "scale_shift_table")[None] + temb.reshape(temb.shape[0], 6, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [ss[:, j] for j in range(6)]
        # 1. linear self-attention
        h = layernorm(x) * (1 + scale_msa) + shift_msa
        q = self.trunk(p + "attn1.to_q.weight")(h)
        k = self.trunk(p + "attn1.to_k.weight")(h)
        v = self.trunk(p + "attn1.to_v.weight")(h)
        B, N, _ = q.shape
        q = q.reshape(B, N, HEADS, HEAD_DIM).transpose(0, 2, 3, 1)  # [B,heads,hd,N]
        k = k.reshape(B, N, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)  # [B,heads,N,hd]
        v = v.reshape(B, N, HEADS, HEAD_DIM).transpose(0, 2, 3, 1)  # [B,heads,hd,N]
        q = nn.relu(q).astype(mx.float32); k = nn.relu(k).astype(mx.float32); v = v.astype(mx.float32)
        ones = mx.ones((B, HEADS, 1, N), dtype=mx.float32)
        v = mx.concatenate([v, ones], axis=2)  # [B,heads,hd+1,N]
        scores = v @ k                          # [B,heads,hd+1,hd]
        o = scores @ q                          # [B,heads,hd+1,N]
        o = o[:, :, :-1] / (o[:, :, -1:] + 1e-15)  # [B,heads,hd,N]
        o = o.transpose(0, 3, 1, 2).reshape(B, N, HEADS * HEAD_DIM).astype(x.dtype)
        o = self.trunk(p + "attn1.to_out.0.weight", p + "attn1.to_out.0.bias")(o)
        x = x + gate_msa * o
        # 2. cross-attention (SDPA)
        q = self.trunk(p + "attn2.to_q.weight", p + "attn2.to_q.bias")(x)
        k = self.trunk(p + "attn2.to_k.weight", p + "attn2.to_k.bias")(enc)
        v = self.trunk(p + "attn2.to_v.weight", p + "attn2.to_v.bias")(enc)
        M = enc.shape[1]
        q = q.reshape(B, N, X_HEADS, X_HEAD_DIM).transpose(0, 2, 1, 3)
        k = k.reshape(B, M, X_HEADS, X_HEAD_DIM).transpose(0, 2, 1, 3)
        v = v.reshape(B, M, X_HEADS, X_HEAD_DIM).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(X_HEAD_DIM))
        o = o.transpose(0, 2, 1, 3).reshape(B, N, X_HEADS * X_HEAD_DIM)
        o = self.trunk(p + "attn2.to_out.0.weight", p + "attn2.to_out.0.bias")(o)
        x = x + o
        # 3. GLU FFN (operate per-token for 1x1, NHWC for depthwise)
        h = layernorm(x) * (1 + scale_mlp) + shift_mlp
        h = self.trunk(p + "ff.conv_inverted.weight", p + "ff.conv_inverted.bias")(h)  # [B,N,11200]
        h = silu(h)
        # depthwise 3x3 (island), NHWC
        himg = h.reshape(B, H, W, 2 * HIDDEN)
        dw = self.g(p + "ff.conv_depth.weight")  # [11200,1,3,3] torch -> need [11200,3,3,1]
        dw = dw.reshape(2 * HIDDEN, 1, 3, 3).transpose(0, 2, 3, 1)
        himg = mx.conv2d(himg, dw, stride=1, padding=1, groups=2 * HIDDEN)
        db = self.g(p + "ff.conv_depth.bias")
        himg = himg + db
        h = himg.reshape(B, N, 2 * HIDDEN)
        a, gate = h[..., :HIDDEN], h[..., HIDDEN:]
        h = a * silu(gate)
        h = self.trunk(p + "ff.conv_point.weight")(h)  # [B,N,2240], no bias
        x = x + gate_mlp * h
        return x

    def __call__(self, latent, enc, t):
        # latent [B,32,16,16] (NCHW torch convention from caller); enc [B,M,2304]; t [B]
        g = self.g
        B = latent.shape[0]
        H = W = latent.shape[-1]
        # patch_embed: 1x1 conv 32->2240 + flatten
        x = latent.transpose(0, 2, 3, 1).reshape(B, H * W, 32)  # tokens [B,256,32]
        x = self.island("patch_embed.proj.weight", "patch_embed.proj.bias")(x)  # [B,256,2240]
        # time embed (AdaLN single)
        tproj = timestep_embedding(t, 256)
        te = self.island("time_embed.emb.timestep_embedder.linear_1.weight", "time_embed.emb.timestep_embedder.linear_1.bias")(tproj)
        te = silu(te)
        emb = self.island("time_embed.emb.timestep_embedder.linear_2.weight", "time_embed.emb.timestep_embedder.linear_2.bias")(te)  # embedded_timestep [B,2240]
        temb = self.island("time_embed.linear.weight", "time_embed.linear.bias")(silu(emb))  # [B,13440]
        # caption projection + rms norm
        enc = self.island("caption_projection.linear_1.weight", "caption_projection.linear_1.bias")(enc)
        enc = gelu_tanh(enc)
        enc = self.island("caption_projection.linear_2.weight", "caption_projection.linear_2.bias")(enc)
        cw = g("caption_norm.weight")
        enc = (enc.astype(mx.float32) * mx.rsqrt((enc.astype(mx.float32) ** 2).mean(-1, keepdims=True) + 1e-5)).astype(enc.dtype) * cw
        for i in range(N_LAYERS):
            x = self.block(x, enc, temb, H, W, i)
        # norm_out (modulated) + proj_out
        ss = g("scale_shift_table")[None] + emb[:, None]
        shift, scale = ss[:, 0:1], ss[:, 1:2]
        x = layernorm(x) * (1 + scale) + shift
        x = self.island("proj_out.weight", "proj_out.bias")(x)  # [B,256,32]
        x = x.reshape(B, H, W, 32).transpose(0, 3, 1, 2)  # [B,32,16,16]
        return x
