"""Bonsai-style quantization-aware training (QAT) library.

Reverse-engineered from Prism's Bonsai checkpoints:
  - STE ternary forward, continuous master weights backward
  - Optimal-scale ternary initialization (from Clark-Quantize)
  - Proximal pull: after optimizer step, drag master weights toward codebook
  - Island protection: keep sensitive layers in FP16
  - Selective island training: freeze proj_out, train norm_out hard

Supports nn.Linear and nn.Conv2d (1x1 pointwise only for group quantization).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math
from typing import Dict, List, Optional, Tuple, Set


# ---------------------------------------------------------------------------
# Optimal-scale ternary (from Clark-Quantize, reused for initialization)
# ---------------------------------------------------------------------------

@torch.no_grad()
def optimal_ternary_codes(W: torch.Tensor, group: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute optimal-scale ternary codes and scales for W [out, in].

    Returns (t, s) where t in {-1,0,+1} [out, in] and s [out, 1] (per-row scale).
    For 2D weights, groups run along the input dimension.
    """
    if W.dim() == 4:
        # Conv2d: [out, in, kh, kw] -> [out, in*kh*kw]
        out, inp, kh, kw = W.shape
        Wf = W.reshape(out, inp * kh * kw)
    else:
        Wf = W.float()

    out, inp = Wf.shape
    pad = (-inp) % group
    if pad:
        Wf = torch.cat([Wf, torch.zeros(out, pad, device=Wf.device, dtype=Wf.dtype)], dim=1)
    G = Wf.shape[1] // group
    Wg = Wf.reshape(out, G, group)

    A = Wg.abs()
    a, _ = torch.sort(A, dim=-1, descending=True)
    cs = a.cumsum(-1)
    ks = torch.arange(1, group + 1, device=Wf.device, dtype=torch.float32)
    ratio = cs * cs / ks
    kbest = ratio.argmax(-1)
    s = (cs.gather(-1, kbest[..., None]) / (kbest[..., None].float() + 1)).clamp_min(1e-8)
    thr = a.gather(-1, kbest[..., None])
    t = torch.sign(Wg) * (A >= thr).float()

    # Return codes in original shape and per-row scales
    t = t.reshape(out, G * group)[:, :inp]
    if W.dim() == 4:
        t = t.reshape(out, inp, kh, kw)
    s = s.mean(-1)  # average across groups -> per-row scale
    return t, s.unsqueeze(-1)


# ---------------------------------------------------------------------------
# STE Ternary Layer wrapper
# ---------------------------------------------------------------------------

class TernaryLinear(nn.Module):
    """nn.Linear with STE ternary quantization.

    Keeps a trainable master weight (u). Forward uses hard ternary:
        t = optimal_scale_ternary(u)
        W_fwd = s * t + (u - u.detach())   # STE

    The group scale s and codes t are recomputed every forward pass from u,
    so they track the master weight as it trains.

    Args:
        original: the nn.Linear to replace
        group: group size for ternary quantization (default 128)
    """

    def __init__(self, original: nn.Linear, group: int = 128):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.group = group
        self.has_bias = original.bias is not None

        # Master weight (trainable, continuous)
        self.u = nn.Parameter(original.weight.data.clone().float())
        if self.has_bias:
            self.bias = nn.Parameter(original.bias.data.clone().float())
        else:
            self.register_parameter('bias', None)

        # Cached quantized weight (updated on forward, used by .weight property)
        self._wq_cached = None

        # Store initial codes for tracking
        with torch.no_grad():
            t0, s0 = optimal_ternary_codes(self.u.data, group)
            self.register_buffer('_init_codes', t0.clone(), persistent=False)

    @property
    def weight(self) -> torch.Tensor:
        """Return quantized weight (STE). Allows compatibility with modules that
        access .weight directly (e.g. nn.MultiheadAttention.out_proj)."""
        if self._wq_cached is not None:
            return self._wq_cached + (self.u - self.u.detach())
        return self.u

    def _quantize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ternary codes and scales from master weight u."""
        W = self.u
        if W.dim() == 4:
            out, inp, kh, kw = W.shape
            Wf = W.reshape(out, inp * kh * kw)
        else:
            Wf = W

        out, inp = Wf.shape
        group = self.group
        pad = (-inp) % group
        if pad:
            Wf = torch.cat([Wf, torch.zeros(out, pad, device=Wf.device, dtype=Wf.dtype)], dim=1)
        G = Wf.shape[1] // group
        Wg = Wf.reshape(out, G, group)

        A = Wg.abs()
        a, _ = torch.sort(A, dim=-1, descending=True)
        cs = a.cumsum(-1)
        ks = torch.arange(1, group + 1, device=Wf.device, dtype=torch.float32)
        ratio = cs * cs / ks
        kbest = ratio.argmax(-1)
        s = (cs.gather(-1, kbest[..., None]) / (kbest[..., None].float() + 1)).clamp_min(1e-8)
        thr = a.gather(-1, kbest[..., None])
        t = torch.sign(Wg) * (A >= thr).float()

        Wq = (t * s).reshape(out, G * group)[:, :inp]
        if W.dim() == 4:
            Wq = Wq.reshape(out, inp, kh, kw)
        return Wq, t.reshape(out, G * group)[:, :inp]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Wq, t = self._quantize()
        # STE: forward uses quantized weight, gradient passes through to u
        W_fwd = Wq + (self.u - self.u.detach())
        self._wq_cached = Wq  # cache for .weight property
        return F.linear(x, W_fwd, self.bias)

    @torch.no_grad()
    def proximal_pull(self, rho: float):
        """Pull master weight toward quantized codebook: u <- (1-rho)*u + rho*Wq."""
        if rho <= 0:
            return
        Wq, _ = self._quantize()
        self.u.data.mul_(1 - rho).add_(Wq, alpha=rho)

    @torch.no_grad()
    def get_quantized_weight(self) -> torch.Tensor:
        """Return the dequantized ternary weight (for evaluation/packing)."""
        Wq, _ = self._quantize()
        return Wq

    @torch.no_grad()
    def code_stats(self) -> dict:
        """Return statistics about current ternary codes vs initial."""
        Wq, t = self._quantize()
        t_init = self._init_codes
        if t.dim() > 2:
            t = t.reshape(t.shape[0], -1)
            t_init = t_init.reshape(t_init.shape[0], -1)
        same = (t == t_init).float().mean().item()
        sign_flip = ((t * t_init < 0)).float().mean().item()
        zero_frac = (t == 0).float().mean().item()
        init_zero_frac = (t_init == 0).float().mean().item()
        activated = ((t_init == 0) & (t != 0)).float().mean().item()
        deactivated = ((t_init != 0) & (t == 0)).float().mean().item()
        return {
            'same': same, 'sign_flip': sign_flip,
            'zero_frac': zero_frac, 'init_zero_frac': init_zero_frac,
            'activated': activated, 'deactivated': deactivated,
        }


# ---------------------------------------------------------------------------
# Island protection and model wrapping
# ---------------------------------------------------------------------------

DEFAULT_ISLAND_KEYWORDS = (
    "time_embed", "caption_projection", "caption_norm", "proj_out",
    "adaln", "modulation", "pos_embed", "patch_embed", "x_embedder",
    "context_embedder", "norm_out", "scale_shift_table",
    "conv_depth", "norm_q", "norm_k", "norm_added",
)


def is_island(name: str, island_keywords: Tuple[str, ...] = DEFAULT_ISLAND_KEYWORDS) -> bool:
    """Check if a module name matches any island keyword."""
    return any(kw in name for kw in island_keywords)


def wrap_model_ternary(
    model: nn.Module,
    group: int = 128,
    island_keywords: Tuple[str, ...] = DEFAULT_ISLAND_KEYWORDS,
    verbose: bool = True,
) -> Tuple[int, int, List[str]]:
    """Replace all nn.Linear (not in island) with TernaryLinear.

    Returns (n_ternary, n_island, ternary_layer_names).
    """
    n_ternary = 0
    n_island = 0
    ternary_names = []

    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if is_island(name, island_keywords):
            n_island += 1
            continue

        # Replace with TernaryLinear
        parent, _, leaf = _get_parent(model, name)
        new_mod = TernaryLinear(mod, group=group)
        setattr(parent, leaf, new_mod)
        n_ternary += 1
        ternary_names.append(name)

    if verbose:
        print(f"[bonsai] wrapped {n_ternary} trunk linears as TernaryLinear, "
              f"kept {n_island} island linears in FP16")

    return n_ternary, n_island, ternary_names


def _get_parent(model: nn.Module, dotted_name: str) -> Tuple[nn.Module, str, str]:
    """Get parent module, parent name, and leaf name from dotted path."""
    parts = dotted_name.split('.')
    parent_name = '.'.join(parts[:-1])
    leaf = parts[-1]
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parent_name, leaf


# ---------------------------------------------------------------------------
# Proximal pull scheduler
# ---------------------------------------------------------------------------

class ProximalPullScheduler:
    """Schedule rho for proximal pull across training steps.

    Prism pattern: start at 0 (warmup), ramp up gradually.
    """

    def __init__(self, schedule: List[Tuple[int, float]]):
        """Args:
            schedule: list of (step, rho) pairs, e.g. [(0, 0.0), (500, 0.01), (2000, 0.05), (5000, 0.10)]
        """
        self.schedule = sorted(schedule)

    def get_rho(self, step: int) -> float:
        """Linearly interpolate rho at given step."""
        if step <= self.schedule[0][0]:
            return self.schedule[0][1]
        if step >= self.schedule[-1][0]:
            return self.schedule[-1][1]
        for i in range(len(self.schedule) - 1):
            s0, r0 = self.schedule[i]
            s1, r1 = self.schedule[i + 1]
            if s0 <= step <= s1:
                frac = (step - s0) / max(s1 - s0, 1)
                return r0 + frac * (r1 - r0)
        return self.schedule[-1][1]


# ---------------------------------------------------------------------------
# Forward-blend (soft landing) for Phase 1
# ---------------------------------------------------------------------------

class ForwardBlendScheduler:
    """Blend continuous and quantized forward: W_fwd = (1-alpha)*u + alpha*Wq.

    Ramp alpha from 0 to 1 over training to ease the student onto the codebook.
    """

    def __init__(self, ramp_steps: int):
        self.ramp_steps = ramp_steps

    def get_alpha(self, step: int) -> float:
        if step >= self.ramp_steps:
            return 1.0
        return min(1.0, step / max(self.ramp_steps, 1))


# ---------------------------------------------------------------------------
# Evaluation and packing utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ternary_error(model: nn.Module) -> dict:
    """Measure ternary reconstruction error for all TernaryLinear layers."""
    errors = []
    zero_fracs = []
    for name, mod in model.named_modules():
        if isinstance(mod, TernaryLinear):
            Wq = mod.get_quantized_weight()
            err = ((Wq - mod.u.data).norm() / mod.u.data.norm().clamp_min(1e-9)).item()
            _, t = mod._quantize()
            zf = (t == 0).float().mean().item()
            errors.append(err)
            zero_fracs.append(zf)
    return {
        'mean_ternary_error': sum(errors) / max(len(errors), 1),
        'mean_zero_frac': sum(zero_fracs) / max(len(zero_fracs), 1),
        'n_layers': len(errors),
    }


@torch.no_grad()
def get_all_code_stats(model: nn.Module) -> dict:
    """Aggregate code transition stats across all TernaryLinear layers."""
    all_stats = {
        'same': [], 'sign_flip': [], 'zero_frac': [],
        'init_zero_frac': [], 'activated': [], 'deactivated': [],
    }
    for name, mod in model.named_modules():
        if isinstance(mod, TernaryLinear):
            s = mod.code_stats()
            for k in all_stats:
                all_stats[k].append(s[k])
    return {k: sum(v) / max(len(v), 1) for k, v in all_stats.items()}


@torch.no_grad()
def bake_codes(model: nn.Module):
    """Replace master weights with their quantized values (hard snap).

    After this, the model is fully quantized — no master weights remain.
    """
    for name, mod in list(model.named_modules()):
        if isinstance(mod, TernaryLinear):
            Wq = mod.get_quantized_weight()
            # Replace with a plain nn.Linear containing the quantized weight
            parent, _, leaf = _get_parent(model, name)
            new_linear = nn.Linear(mod.in_features, mod.out_features,
                                   bias=mod.has_bias, device=Wq.device, dtype=Wq.dtype)
            new_linear.weight.data.copy_(Wq)
            if mod.has_bias:
                new_linear.bias.data.copy_(mod.bias.data)
            setattr(parent, leaf, new_linear)


@torch.no_grad()
def count_params(model: nn.Module) -> dict:
    """Count trunk vs island parameters and compute effective bits/weight."""
    trunk_params = 0
    island_params = 0
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, TernaryLinear)):
            n = mod.weight.numel() if hasattr(mod, 'weight') else mod.u.numel()
            if isinstance(mod, TernaryLinear):
                trunk_params += n
            elif is_island(name):
                island_params += n
            else:
                trunk_params += n  # un-wrapped trunk
    total = trunk_params + island_params
    ternary_bpw = math.log2(3) + 16 / 128  # log2(3) + FP16 scale per group
    fp16_bpw = 16.0
    trunk_bits = trunk_params * ternary_bpw
    island_bits = island_params * fp16_bpw
    total_bits = trunk_bits + island_bits
    return {
        'trunk_params': trunk_params,
        'island_params': island_params,
        'total_params': total,
        'trunk_pct': trunk_params / max(total, 1) * 100,
        'island_pct': island_params / max(total, 1) * 100,
        'effective_bpw': total_bits / max(total, 1),
        'compression': total * 16 / max(total_bits, 1),
        'storage_gb': total_bits / 8 / 1e9,
    }
