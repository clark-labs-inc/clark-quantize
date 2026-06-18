"""Activation-aware ternary: GPTQ + incoherence rotation.

Optimal-scale ternary (see ternary.py) minimizes the per-weight error
||W - Wq||. To reach near-FP16 quality at ternary precision we instead minimize
the per-OUTPUT error ||(W - Wq) x|| using two ideas:

  1. GPTQ -- quantize columns left-to-right; after quantizing a column, push its
     rounding error onto the not-yet-quantized columns weighted by the inverse
     Hessian H^{-1} (H = sum_x x x^T), so error lands where activations absorb it.
  2. Incoherence rotation -- a single shared random orthogonal R rotates every
     same-width linear's input space (W -> W R^T, x -> x R^T, exact since
     R^T R = I). This spreads outliers so a single group scale suffices. The
     Hessian rotates analytically: H -> R H R^T (no re-calibration).

Calibration is "self-generated": the Hessians are collected from the model's own
outputs on generic prompts, so no external dataset is required.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .ternary import _optimal_scale_rows  # per-row optimal ternary scale


@torch.no_grad()
def random_orthogonal(d: int, device, seed: int = 0) -> torch.Tensor:
    """Haar-distributed orthogonal matrix via QR (sign-fixed)."""
    g = torch.randn(d, d, device=device, generator=torch.Generator(device).manual_seed(seed))
    R, _ = torch.linalg.qr(g)
    return R * torch.sign(torch.diag(R))[:, None]


@torch.no_grad()
def gptq_quantize(W: torch.Tensor, H: torch.Tensor, group: int = 128,
                  percdamp: float = 0.01) -> torch.Tensor:
    """GPTQ with a ternary, per-group optimal-scale column quantizer.

    W [out, in] (float), H [in, in] (the input second moment). Returns the
    dequantized ternary weight Wq [out, in] minimizing ||(W - Wq) x||^2.
    """
    W = W.clone().float().to(H.device)
    out, cols = W.shape
    H = H.clone().float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0
    H[range(cols), range(cols)] += percdamp * torch.diag(H).mean()
    Hinv = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Hinv)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)         # upper Cholesky of H^{-1}

    Q = torch.zeros_like(W)
    for i1 in range(0, cols, group):
        i2 = min(i1 + group, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        s = _optimal_scale_rows(W1).squeeze(1)             # per-row group scale
        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hi[i, i]
            q = torch.clamp(torch.round(w / s), -1, 1) * s
            err = (w - q) / d
            W1[:, i:] -= err[:, None] * Hi[i, i:][None, :]
            Q1[:, i] = q
            E[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E @ Hinv[i1:i2, i2:]
    return Q


class HessianCollector:
    """Forward-pre-hook collector for H = sum_x x x^T over chosen Linear modules."""

    def __init__(self, modules: dict, device, max_tokens: int = 4096):
        self.modules = modules
        self.H = {n: torch.zeros(m.weight.shape[1], m.weight.shape[1], device=device)
                  for n, m in modules.items()}
        self.cnt = {n: 0 for n in modules}
        self.max_tokens = max_tokens
        self._handles = []

    def _hook(self, name):
        def f(_mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            if x.shape[0] > self.max_tokens:
                idx = torch.randperm(x.shape[0], device=x.device)[:self.max_tokens]
                x = x[idx]
            self.H[name].addmm_(x.t(), x)
            self.cnt[name] += x.shape[0]
        return f

    def __enter__(self):
        self._handles = [m.register_forward_pre_hook(self._hook(n))
                         for n, m in self.modules.items()]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        for n in self.H:
            self.H[n] /= max(self.cnt[n], 1)


@torch.no_grad()
def apply_gptq(modules: dict, hessians: dict, group: int = 128,
               rotation: torch.Tensor | None = None, rot_dim: int | None = None):
    """Quantize each module in `modules` with GPTQ; optionally rotate (incoherence).

    If `rotation` (R, [rot_dim, rot_dim]) is given, every Linear whose input width
    equals `rot_dim` is quantized in the rotated basis and gets a forward-pre-hook
    that applies x -> x R^T at inference (so the maths stays exact).
    """
    def rot_hook(_mod, inp):
        x = inp[0]
        return ((x.float() @ rotation.t()).to(x.dtype),) + inp[1:]

    n_rot = 0
    for name, m in modules.items():
        W = m.weight.data.float().to(hessians[name].device)
        H = hessians[name]
        rotate = rotation is not None and W.shape[1] == rot_dim
        if rotate:
            W = W @ rotation.t()
            H = rotation @ H @ rotation.t()
        Q = gptq_quantize(W, H, group=group)
        m.weight.data.copy_(Q.to(m.weight.dtype))
        if rotate:
            m.register_forward_pre_hook(rot_hook)
            n_rot += 1
        torch.cuda.empty_cache()
    return {"quantized": len(modules), "rotated": n_rot}
