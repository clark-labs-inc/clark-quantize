"""Clark-Quantize core: data-free group-wise ternary quantization.

A weight matrix W in R^{out x in} is quantized so that every weight takes a value
in {-s_g, 0, +s_g}, where s_g is a single FP16 scale shared by a contiguous group
of `group` weights along the input dimension:

    w_i ~= s_g * t_i,   t_i in {-1, 0, +1}.

The only thing that matters for quality is how (s_g, t) are chosen. We support:

  method="optimal"  -- the MSE-optimal symmetric ternary quantizer (default).
                       For a group, minimizing sum_i (w_i - s t_i)^2 over the
                       ternary support S (the indices set to +/-1) and scale s
                       has the closed form
                           s* = mean_{i in S} |w_i|,
                       and the optimal support of size k is the top-k by |w|.
                       Choosing k reduces to MAXIMIZING (sum_{top-k} |w|)^2 / k.
                       Fully data-free, O(g log g) per group.

  method="absmean"  -- s = mean_i |w_i| over the whole group, t = clip(round(w/s)).
                       The common 1.58-bit default; provided as a baseline. It is
                       provably suboptimal because it never chooses the support.

Optional, data-free refinements (off by default; see paper for why they do not
help diffusion trunks): error feedback (sigma-delta rounding) and per-output-row
least-squares gain correction.
"""
from __future__ import annotations
import re
import torch
import torch.nn as nn


@torch.no_grad()
def _optimal_scale_rows(Wblk: torch.Tensor) -> torch.Tensor:
    """Per-row MSE-optimal ternary scale for a block [out, g]: returns [out, 1].
    Maximizes (sum_top-k |w|)^2 / k, the closed-form minimum-MSE ternary scale.
    """
    a, _ = torch.sort(Wblk.abs(), dim=1, descending=True)
    cs = a.cumsum(1)
    ks = torch.arange(1, Wblk.shape[1] + 1, device=Wblk.device, dtype=cs.dtype)
    kbest = (cs * cs / ks).argmax(1)
    return (cs.gather(1, kbest[:, None]) / (kbest[:, None].to(cs.dtype) + 1)).clamp_min(1e-8)


@torch.no_grad()
def ternarize(W: torch.Tensor, group: int = 128, method: str = "optimal",
              error_feedback: bool = False, row_gain: bool = False):
    """Ternarize a 2D weight [out, in]; groups run along the input dimension.

    Returns (Wq, stats) where Wq is the dequantized fp32 weight (same shape) and
    stats is a dict with zero_frac, sign_flip and weight_rel_err.
    """
    dev = W.device
    out, in_features = W.shape
    pad = (-in_features) % group
    if pad:
        W = torch.cat([W, torch.zeros(out, pad, device=dev, dtype=W.dtype)], dim=1)
    G = W.shape[1] // group
    Wg = W.reshape(out, G, group)
    A = Wg.abs()

    if method == "absmean":
        s = A.mean(-1, keepdim=True).clamp_min(1e-8)
        thr = 0.5 * s
    elif method == "optimal":
        a, _ = torch.sort(A, dim=-1, descending=True)
        cs = a.cumsum(-1)
        ks = torch.arange(1, group + 1, device=dev, dtype=cs.dtype)
        ratio = cs * cs / ks                                # maximize -> min MSE
        kbest = ratio.argmax(-1)                            # index = k-1
        s = (cs.gather(-1, kbest[..., None]) /
             (kbest[..., None].to(cs.dtype) + 1)).clamp_min(1e-8)
        thr = a.gather(-1, kbest[..., None])                # k-th largest |w|
    else:
        raise ValueError(f"unknown method {method!r}")

    if error_feedback:                                      # sigma-delta rounding
        t = torch.zeros_like(Wg)
        err = torch.zeros(out, G, device=dev, dtype=Wg.dtype)
        ss = s.squeeze(-1)
        for k in range(group):
            v = Wg[..., k] + err
            tk = torch.clamp(torch.round(v / ss), -1, 1)
            t[..., k] = tk
            err = v - tk * ss
    else:
        t = torch.sign(Wg) * (A >= thr).to(Wg.dtype)        # threshold assignment

    Wq = (t * s).reshape(out, G * group)[:, :in_features]
    Wt = W[:, :in_features]
    if row_gain:                                            # per-row LS gain
        num = (Wt * Wq).sum(1)
        den = (Wq * Wq).sum(1).clamp_min(1e-12)
        Wq = Wq * (num / den)[:, None]

    nz = t != 0
    stats = {
        "zero_frac": (t == 0).float().mean().item(),
        "sign_flip": ((nz & (torch.sign(t) != torch.sign(Wg))).float().sum()
                      / nz.float().sum().clamp_min(1)).item(),
        "weight_rel_err": ((Wq - Wt).norm() / Wt.norm().clamp_min(1e-9)).item(),
    }
    return Wq.contiguous(), stats


# Layers whose ternarization disproportionately hurts a diffusion transformer:
# adaptive-norm modulation, text/time conditioning, and the final projection.
# They are a small fraction of parameters but gate every block, so we keep them
# in the original precision by default (see paper, Section "Layer sensitivity").
DEFAULT_PROTECT = ("time_embed", "caption_projection", "proj_out",
                   "adaln", "modulation", "pos_embed", "patch_embed")


@torch.no_grad()
def quantize_module(model: nn.Module, group: int = 128, method: str = "optimal",
                    error_feedback: bool = False, row_gain: bool = False,
                    protect=DEFAULT_PROTECT, keep_blocks: int = 0,
                    block_regex: str = r"(?:transformer_blocks|blocks|layers)\.(\d+)\.",
                    verbose: bool = True):
    """In-place fake-quantize every nn.Linear in `model` (weights replaced by their
    dequantized ternary values, dtype preserved, so the model runs unchanged).

    `protect`     : substrings of module names kept at original precision.
    `keep_blocks` : keep the first/last K transformer blocks at original precision.
    Returns aggregate stats.
    """
    n_blocks = 0
    for name, _ in model.named_modules():
        m = re.search(block_regex, name + ".")
        if m:
            n_blocks = max(n_blocks, int(m.group(1)) + 1)
    keep = set(range(keep_blocks)) | set(range(n_blocks - keep_blocks, n_blocks))

    n = skip = 0
    psum = z = f = rel = 0.0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(p in name for p in protect):
            skip += 1
            continue
        bm = re.search(block_regex, name + ".")
        if keep_blocks and bm and int(bm.group(1)) in keep:
            skip += 1
            continue
        W = mod.weight.data.float()
        Wq, st = ternarize(W, group, method, error_feedback, row_gain)
        mod.weight.data.copy_(Wq.to(mod.weight.dtype))
        p = mod.weight.numel()
        n += 1
        psum += p
        z += st["zero_frac"] * p
        f += st["sign_flip"] * p
        rel += st["weight_rel_err"] * p

    agg = {"ternary_linears": n, "fp16_linears": skip,
           "params_ternary_M": psum / 1e6,
           "zero_frac": z / max(psum, 1), "sign_flip": f / max(psum, 1),
           "weight_rel_err": rel / max(psum, 1)}
    if verbose:
        print(f"[clark-quantize] method={method} group={group} ef={error_feedback} "
              f"row_gain={row_gain} keep_blocks={keep_blocks}")
        print(f"  ternary linears={n}  kept={skip}  params={agg['params_ternary_M']:.0f}M"
              f"  zero={agg['zero_frac']*100:.1f}%  flip={agg['sign_flip']*100:.1f}%"
              f"  weight_rel_err={agg['weight_rel_err']*100:.1f}%")
    return agg
