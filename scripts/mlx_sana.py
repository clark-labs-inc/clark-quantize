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


def mse_ternary(W, g):
    """MSE-optimal ternary, group g along in-dim -> codes[out,in]{-1,0,1}, scales[out,ng]."""
    out, inn = W.shape
    Wg = W.reshape(out, inn // g, g).astype(mx.float32)
    A = mx.abs(Wg)
    a = -mx.sort(-A, axis=-1)  # descending
    cs = mx.cumsum(a, axis=-1)
    ks = mx.arange(1, g + 1, dtype=mx.float32)
    k = mx.argmax(cs * cs / ks, axis=-1, keepdims=True)
    s = mx.take_along_axis(cs, k, axis=-1) / (k.astype(mx.float32) + 1)
    thr = mx.take_along_axis(a, k, axis=-1)
    t = mx.sign(Wg) * (A >= thr)
    return t.reshape(out, inn), s.squeeze(-1)


def pack_ternary_mlx(code, scale):
    """Inject our ternary into MLX 2-bit slots. levels=code+1{0,1,2}, scale/bias chosen
    so MLX affine dequant (level*scale+bias) reproduces code*scale exactly."""
    out, inn = code.shape
    lev = (code + 1).astype(mx.uint32).reshape(out, inn // 16, 16)
    pow4 = mx.array([4], dtype=mx.uint32) ** mx.arange(16, dtype=mx.uint32)
    wq = (lev * pow4[None, None, :]).sum(axis=2).astype(mx.uint32)
    return wq, scale, -scale


def is_trunk_key(k):  # diffusers weight key -> is it a quantizable trunk linear/1x1-conv?
    if not (k.startswith("transformer_blocks") and k.endswith(".weight")):
        return False
    attn = (".attn1." in k or ".attn2." in k) and any(s + ".weight" in k for s in ("to_q", "to_k", "to_v", "to_out.0"))
    ff = (".ff.conv_inverted.weight" in k or ".ff.conv_point.weight" in k)
    return attn or ff


def build_packed(sd):
    """Build a serializable mlx-2bit dict: trunk -> ternary in MLX 2-bit (k::wq/scales/biases/gs),
    everything else copied fp16."""
    out = {}
    for k, w in sd.items():
        if is_trunk_key(k):
            ww = w.reshape(w.shape[0], w.shape[1]) if w.ndim == 4 else w
            g = pow2_group(ww.shape[1])
            code, scale = mse_ternary(ww.astype(mx.float32), g)
            wq, sc, _bi = pack_ternary_mlx(code, scale.astype(mx.float16))  # bias == -scale, derived on load
            out[k + "::wq"] = wq
            out[k + "::scales"] = sc
            out[k + "::gs"] = mx.array([g], dtype=mx.int32)
        else:
            out[k] = w
    return out


class QLinear:
    """Linear: plain fp16 (islands) or ternary injected into MLX 2-bit (trunk)."""
    def __init__(self, w=None, bias=None, ternary=False, group_size=64, packed=None):
        self.bias = bias
        if packed is not None:
            self.wq, self.scales, self.biases, self.gs = packed
            self.q = True
        elif not ternary:
            self.q = None
            self.w = w
        else:
            code, scale = mse_ternary(w.astype(mx.float32), group_size)
            self.wq, self.scales, self.biases = pack_ternary_mlx(code, scale.astype(mx.float16))
            self.gs = group_size
            self.q = True

    def __call__(self, x):
        if self.q is None:
            y = x @ self.w.T
        else:
            y = mx.quantized_matmul(x, self.wq, scales=self.scales, biases=self.biases,
                                    transpose=True, group_size=self.gs, bits=2)
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
    def __init__(self, sd, quantize=False):
        # sd: dict of mx.array keyed by diffusers PyTorch names. quantize: ternary trunk -> MLX 2-bit
        self.sd = sd
        self.quantize = quantize
        g = lambda k: sd[k]
        self.g = g
        self._cache = {}
        # trunk (quantizable) linear weights, with per-layer pow2 group (cached across forwards)
        def trunk(k_w, k_b=None):
            if k_w in self._cache:
                return self._cache[k_w]
            b = sd[k_b] if (k_b and k_b in sd) else None
            if k_w + "::wq" in sd:  # pre-packed mlx-2bit artifact (bias == -scale)
                sc = sd[k_w + "::scales"]
                ql = QLinear(bias=b, packed=(sd[k_w + "::wq"], sc, -sc, int(sd[k_w + "::gs"][0].item())))
            else:
                w = sd[k_w]
                if w.ndim == 4:  # 1x1 conv -> [out,in]
                    w = w.reshape(w.shape[0], w.shape[1])
                ql = QLinear(w, b, ternary=True, group_size=pow2_group(w.shape[1])) if quantize else QLinear(w, b)
            self._cache[k_w] = ql
            return ql
        def island(k_w, k_b=None):
            w = sd[k_w]
            if w.ndim == 4:
                w = w.reshape(w.shape[0], w.shape[1])
            return QLinear(w, sd[k_b] if (k_b and k_b in sd) else None)
        self.trunk, self.island = trunk, island

    def block(self, x, enc, enc_bias, temb, H, W, i):
        p = f"transformer_blocks.{i}."
        g = self.g
        ss = g(p + "scale_shift_table")[None] + temb.reshape(temb.shape[0], 6, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [ss[:, j:j + 1] for j in range(6)]
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
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(X_HEAD_DIM), mask=enc_bias)
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

    def __call__(self, latent, enc, t, enc_mask=None):
        # latent [B,32,16,16] (NCHW torch convention from caller); enc [B,M,2304]; t [B]
        g = self.g
        B = latent.shape[0]
        enc_bias = None
        if enc_mask is not None:
            enc_bias = ((1.0 - enc_mask.astype(mx.float32)) * -1e4)[:, None, None, :]  # [B,1,1,M]
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
            x = self.block(x, enc, enc_bias, temb, H, W, i)
        # norm_out (modulated) + proj_out
        ss = g("scale_shift_table")[None] + emb[:, None]
        shift, scale = ss[:, 0:1], ss[:, 1:2]
        x = layernorm(x) * (1 + scale) + shift
        x = self.island("proj_out.weight", "proj_out.bias")(x)  # [B,256,32]
        x = x.reshape(B, H, W, 32).transpose(0, 3, 1, 2)  # [B,32,16,16]
        return x
