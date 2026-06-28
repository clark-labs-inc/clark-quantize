"""General per-group affine N-bit fake-quantization (naive PTQ baseline).

Used to render a quality-vs-bitwidth comparison: this is the *naive* post-training
quantizer (per-group asymmetric min-max affine) that any low-bit knob would apply,
to contrast against our *trained* ternary (~1.58-bit) QAT. Bit-width-agnostic, so
3/4/5/6/7/8-bit are all available; it returns a dequantized tensor (fake-quant) for
rendering — no kernel packing.
"""
import torch


def pow2_group(inn: int, cap: int = 128) -> int:
    """Largest power-of-two group size (<= cap) that divides `inn`."""
    g = cap
    while g >= 8:
        if inn % g == 0:
            return g
        g //= 2
    return 8


@torch.no_grad()
def quantize_affine_nbits(W: torch.Tensor, n_bits: int, group_size: int | None = None) -> torch.Tensor:
    """Per-group asymmetric min-max affine N-bit quantize, returned dequantized.

    W: [out, in] (any float dtype). Codes live in [0, 2**n_bits - 1] with a per-group
    scale + zero-point along the input dimension. Returns the fake-quantized tensor in
    W's dtype. `group_size=None` picks the largest power-of-two group dividing `in`.
    """
    out, inn = W.shape
    g = group_size or pow2_group(inn)
    if inn % g != 0:
        g = pow2_group(inn)
    qmax = (1 << n_bits) - 1

    Wg = W.reshape(out, -1, g).float()
    wmin = Wg.amin(dim=-1, keepdim=True)
    wmax = Wg.amax(dim=-1, keepdim=True)
    scale = (wmax - wmin).clamp_(min=1e-8) / qmax
    zero = wmin

    q = ((Wg - zero) / scale).round_().clamp_(0, qmax)
    deq = q * scale + zero
    return deq.reshape(out, inn).to(W.dtype)
