"""Codebook-aware optimizer for Bonsai master-weight QAT.

BonsaiQAT8 keeps the successful invariant from the Sana runs: optimize a
continuous master weight and derive binary/ternary support only from the hard
forward quantizer. It adds a cheap threshold-aware preconditioned update, but
does not introduce explicit mask logits or target density losses.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _rms(x: Tensor, eps: float = 1e-12) -> Tensor:
    x = x.float()
    return torch.sqrt(torch.mean(x * x) + eps)


@torch.no_grad()
def _quantize_blockwise_int8(x: Tensor, block_size: int = 4096) -> Tuple[Tensor, Tensor, torch.Size]:
    shape = x.shape
    flat = x.detach().float().reshape(-1)
    n = flat.numel()
    blocks = _ceil_div(n, block_size)
    pad = blocks * block_size - n
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    view = flat.view(blocks, block_size)
    scale = view.abs().amax(dim=1).clamp_min(1e-12)
    q = torch.round(view / scale[:, None] * 127.0).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32), shape


@torch.no_grad()
def _dequantize_blockwise_int8(
    q: Optional[Tensor],
    scale: Optional[Tensor],
    shape: torch.Size,
    *,
    block_size: int = 4096,
    device=None,
) -> Tensor:
    if q is None or scale is None:
        return torch.zeros(shape, device=device, dtype=torch.float32)
    x = q.float() * (scale[:, None].float() / 127.0)
    x = x.reshape(-1)[: math.prod(shape)]
    return x.view(shape).to(device=device, dtype=torch.float32)


@torch.no_grad()
def _newton_schulz_orthogonalize(g: Tensor, *, steps: int = 2, eps: float = 1e-7) -> Tensor:
    if g.ndim != 2:
        return g
    orig_dtype = g.dtype
    x = g.float()
    norm = torch.linalg.norm(x)
    if not torch.isfinite(norm) or norm <= 0:
        return torch.zeros_like(g)
    x = x / (norm + eps)
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    for _ in range(int(steps)):
        xx_t = x @ x.T
        x = 1.5 * x - 0.5 * (xx_t @ x)
    if transposed:
        x = x.T
    return x.to(orig_dtype)


class BonsaiQAT8(torch.optim.Optimizer):
    """Hybrid optimizer for binary/ternary master-weight QAT.

    Matrix masters use block-int8 first-moment state plus factored row/column
    second-moment state. Scale and island parameters use AdamW.
    """

    def __init__(self, params: Iterable, defaults: Optional[Dict[str, Any]] = None):
        base = dict(
            role="matrix_qat8",
            lr=2e-5,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8,
            block_size=4096,
            prox_rho=0.0,
            prox_interval=1,
            scale_refresh_interval=0,
            threshold_boost=0.0,
            threshold_temp=0.20,
            support_only_boost=True,
            prevent_sign_flip=False,
            max_update_rms=0.0,
            use_muon=False,
            muon_every=16,
            muon_steps=2,
            muon_max_elements=8_000_000,
            muon_rms_match=True,
            adam_state_dtype=torch.float32,
        )
        if defaults:
            base.update(defaults)
        super().__init__(params, base)
        self.global_step = 0

    def set_prox_rho(self, rho: float) -> None:
        for group in self.param_groups:
            if group.get("role", "matrix_qat8") == "matrix_qat8":
                group["prox_rho"] = float(rho)

    def set_threshold_boost(self, value: float) -> None:
        for group in self.param_groups:
            if group.get("role", "matrix_qat8") == "matrix_qat8":
                group["threshold_boost"] = float(value)

    def set_use_muon(self, value: bool) -> None:
        for group in self.param_groups:
            if group.get("role", "matrix_qat8") == "matrix_qat8":
                group["use_muon"] = bool(value)

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["bonsai_global_step"] = int(self.global_step)
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        state = dict(state_dict)
        saved_global_step = state.pop("bonsai_global_step", None)
        if saved_global_step is None:
            saved_global_step = max(
                (
                    int(item.get("step", 0))
                    for item in state.get("state", {}).values()
                    if isinstance(item, dict)
                ),
                default=0,
            )
        super().load_state_dict(state)
        self.global_step = int(saved_global_step)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.global_step += 1
        for group in self.param_groups:
            role = group.get("role", "matrix_qat8")
            for p in group["params"]:
                if p.grad is None:
                    continue
                if role == "matrix_qat8" and p.ndim == 2:
                    self._matrix_qat8_step(p, group)
                elif role == "adamw":
                    self._adamw_step(p, group)
                else:
                    raise ValueError(f"Unknown BonsaiQAT8 optimizer role: {role}")
        return loss

    @torch.no_grad()
    def _adamw_step(self, p: Tensor, group: Dict[str, Any]) -> None:
        grad = p.grad
        if grad is None:
            return
        if grad.is_sparse:
            raise RuntimeError("BonsaiQAT8 does not support sparse gradients")
        lr = float(group.get("lr", 1e-3))
        beta1, beta2 = group.get("betas", (0.9, 0.95))
        eps = float(group.get("eps", 1e-8))
        wd = float(group.get("weight_decay", 0.0))
        dtype = group.get("adam_state_dtype", torch.float32)
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["m"] = torch.zeros_like(p, dtype=dtype, memory_format=torch.preserve_format)
            state["v"] = torch.zeros_like(p, dtype=dtype, memory_format=torch.preserve_format)
        state["step"] += 1
        step = state["step"]
        m = state["m"]
        v = state["v"]
        g = grad.detach().to(dtype=dtype)
        m.mul_(beta1).add_(g, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        if wd != 0.0:
            p.mul_(1.0 - lr * wd)
        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step
        denom = v.sqrt().div_(math.sqrt(bc2)).add_(eps)
        p.addcdiv_(m.to(dtype=p.dtype), denom.to(dtype=p.dtype), value=-(lr / bc1))

    @torch.no_grad()
    def _matrix_qat8_step(self, p: Tensor, group: Dict[str, Any]) -> None:
        grad = p.grad
        if grad is None:
            return
        if grad.is_sparse:
            raise RuntimeError("BonsaiQAT8 does not support sparse gradients")
        lr = float(group.get("lr", 2e-5))
        beta1, beta2 = group.get("betas", (0.9, 0.95))
        eps = float(group.get("eps", 1e-8))
        wd = float(group.get("weight_decay", 0.0))
        block_size = int(group.get("block_size", 4096))
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["m_q"] = None
            state["m_scale"] = None
            state["m_shape"] = p.shape
            state["v_row"] = torch.zeros(p.shape[0], device=p.device, dtype=torch.float32)
            state["v_col"] = torch.zeros(p.shape[1], device=p.device, dtype=torch.float32)
        state["step"] += 1
        step = state["step"]

        m = _dequantize_blockwise_int8(
            state["m_q"],
            state["m_scale"],
            state["m_shape"],
            block_size=block_size,
            device=p.device,
        )
        g = grad.detach().float()
        m.mul_(beta1).add_(g, alpha=1.0 - beta1)

        g2 = g * g
        v_row = state["v_row"]
        v_col = state["v_col"]
        v_row.mul_(beta2).add_(g2.mean(dim=1), alpha=1.0 - beta2)
        v_col.mul_(beta2).add_(g2.mean(dim=0), alpha=1.0 - beta2)

        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step
        m_hat = m / max(bc1, 1e-12)
        vr = (v_row / max(bc2, 1e-12)).clamp_min(eps)
        vc = (v_col / max(bc2, 1e-12)).clamp_min(eps)
        denom = (vr.sqrt().unsqueeze(1) * vc.sqrt().unsqueeze(0)) / vr.mean().sqrt().clamp_min(eps)
        update = m_hat / denom.clamp_min(eps)
        update = self._apply_threshold_boost(p, update, group)

        if self._should_use_muon(p, group):
            raw_rms = _rms(update)
            mu = _newton_schulz_orthogonalize(update, steps=int(group.get("muon_steps", 2)))
            if bool(group.get("muon_rms_match", True)):
                update = mu * (raw_rms / (_rms(mu) + 1e-12))
            else:
                update = mu

        max_rms = float(group.get("max_update_rms", 0.0))
        if max_rms > 0.0:
            urms = _rms(update)
            if torch.isfinite(urms) and urms > max_rms:
                update.mul_(max_rms / (urms + 1e-12))

        old = p.detach().float()
        proposal = old * (1.0 - lr * wd) if wd != 0.0 else old
        proposal = proposal - lr * update
        if bool(group.get("prevent_sign_flip", False)):
            proposal = self._prevent_sign_flip(old, proposal)
        p.copy_(proposal.to(dtype=p.dtype))

        self._maybe_prox(p, group)
        self._maybe_refresh_scales(p, group)

        q, scale, shape = _quantize_blockwise_int8(m, block_size=block_size)
        state["m_q"] = q
        state["m_scale"] = scale
        state["m_shape"] = shape

    def _should_use_muon(self, p: Tensor, group: Dict[str, Any]) -> bool:
        if not bool(group.get("use_muon", False)):
            return False
        if p.ndim != 2 or p.numel() > int(group.get("muon_max_elements", 8_000_000)):
            return False
        every = int(group.get("muon_every", 16))
        return every <= 1 or (self.global_step % every) == 0

    @torch.no_grad()
    def _apply_threshold_boost(self, p: Tensor, update: Tensor, group: Dict[str, Any]) -> Tensor:
        boost = float(group.get("threshold_boost", 0.0))
        if boost <= 0.0:
            return update
        get_scales = getattr(p, "_bonsai_get_scales", None)
        if not callable(get_scales):
            return update
        mode = getattr(p, "_bonsai_quant_mode", "ternary")
        if mode != "ternary" and bool(group.get("support_only_boost", True)):
            return update
        scales = get_scales()
        if scales is None:
            return update
        scales = scales.detach().float().to(device=p.device)
        group_size = int(getattr(p, "_bonsai_group_size", 128))
        expanded = self._expand_group_scales(scales, p.shape, group_size)
        temp = max(float(group.get("threshold_temp", 0.20)), 1e-6)
        if mode == "ternary":
            threshold_factor = float(getattr(p, "_bonsai_threshold_factor", 0.5))
            threshold = threshold_factor * expanded
            margin = (p.detach().float().abs() - threshold).abs() / expanded.clamp_min(1e-8)
        else:
            margin = p.detach().float().abs() / expanded.clamp_min(1e-8)
        gain = 1.0 + boost * torch.exp(-margin / temp)
        return update * gain

    @staticmethod
    @torch.no_grad()
    def _expand_group_scales(scales: Tensor, shape: torch.Size, group_size: int) -> Tensor:
        out, in_features = shape
        if scales.ndim == 1:
            scales = scales.view(out, 1)
        groups = _ceil_div(in_features, group_size)
        if scales.shape[1] < groups:
            pad = groups - scales.shape[1]
            scales = torch.cat([scales, scales[:, -1:].expand(out, pad)], dim=1)
        expanded = scales[:, :groups].unsqueeze(-1).expand(out, groups, group_size)
        return expanded.reshape(out, groups * group_size)[:, :in_features].contiguous()

    @torch.no_grad()
    def _prevent_sign_flip(self, old: Tensor, proposal: Tensor) -> Tensor:
        crossed = (old * proposal) < 0
        eps = torch.where(old >= 0, torch.full_like(proposal, 1e-8), torch.full_like(proposal, -1e-8))
        return torch.where(crossed, eps, proposal)

    @torch.no_grad()
    def _maybe_prox(self, p: Tensor, group: Dict[str, Any]) -> None:
        rho = float(group.get("prox_rho", 0.0))
        if rho <= 0.0:
            return
        interval = int(group.get("prox_interval", 1))
        if interval > 1 and (self.global_step % interval) != 0:
            return
        make_q = getattr(p, "_bonsai_make_quantized", None)
        if not callable(make_q):
            return
        q = make_q().detach().to(device=p.device, dtype=p.dtype)
        if q.shape != p.shape:
            raise RuntimeError(f"Quantized callback returned {tuple(q.shape)}, expected {tuple(p.shape)}")
        p.lerp_(q, max(0.0, min(1.0, rho)))

    @torch.no_grad()
    def _maybe_refresh_scales(self, p: Tensor, group: Dict[str, Any]) -> None:
        interval = int(group.get("scale_refresh_interval", 0))
        if interval <= 0 or (self.global_step % interval) != 0:
            return
        refresh = getattr(p, "_bonsai_refresh_scales", None)
        if callable(refresh):
            refresh()
