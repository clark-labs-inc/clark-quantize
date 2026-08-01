# bonsai_muon.py
# Complete PyTorch implementation of:
#   - Groupwise binary / ternary STE modules for nn.Linear and 1x1 nn.Conv2d
#   - BonsaiMuon optimizer:
#       Muon-style Newton-Schulz matrix updates for large 2D quantized masters
#       AdamW updates for scales, islands, biases, embeddings, norms
#       proximal pull toward the active codebook
#       optional flip-rate trust region to prevent violent code changes

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _rms(x: Tensor, eps: float = 1e-12) -> Tensor:
    return torch.sqrt(torch.mean(x.float() * x.float()) + eps)


def _as_tuple_2(x) -> Tuple[int, int]:
    if isinstance(x, tuple):
        return x
    return (x, x)


def _group_view_2d(w: Tensor, group_size: int) -> Tuple[Tensor, int, int]:
    if w.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {tuple(w.shape)}")
    out_features, in_features = w.shape
    pad = (-in_features) % group_size
    if pad:
        w_pad = F.pad(w, (0, pad))
    else:
        w_pad = w
    groups = w_pad.shape[1] // group_size
    return w_pad.reshape(out_features, groups, group_size), pad, in_features


def _ungroup_view_2d(wg: Tensor, in_features: int) -> Tensor:
    out_features = wg.shape[0]
    return wg.reshape(out_features, -1)[:, :in_features]


def _valid_group_mask(out_features, groups, group_size, in_features, device) -> Tensor:
    total = groups * group_size
    idx = torch.arange(total, device=device)
    valid = idx < in_features
    return valid.view(1, groups, group_size).expand(out_features, -1, -1)


@torch.no_grad()
def groupwise_binary_absmean_init(w: Tensor, group_size: int = 128, eps: float = 1e-8) -> Tuple[Tensor, Tensor]:
    w_float = w.detach().float()
    wg, pad, in_features = _group_view_2d(w_float, group_size)
    out, groups, g = wg.shape
    mask = _valid_group_mask(out, groups, g, in_features, wg.device)
    abs_sum = wg.abs().masked_fill(~mask, 0.0).sum(dim=-1)
    count = mask.sum(dim=-1).clamp_min(1)
    scales = (abs_sum / count).clamp_min(eps)
    codes = torch.where(wg >= 0, 1.0, -1.0).masked_fill(~mask, 0.0)
    qg = codes * scales.unsqueeze(-1)
    q = _ungroup_view_2d(qg, in_features)
    return q.to(w.dtype), scales.to(w.dtype)


@torch.no_grad()
def groupwise_mse_ternary_init(w: Tensor, group_size: int = 128, eps: float = 1e-8) -> Tuple[Tensor, Tensor]:
    w_float = w.detach().float()
    wg, pad, in_features = _group_view_2d(w_float, group_size)
    out, groups, g = wg.shape
    device = wg.device
    mask = _valid_group_mask(out, groups, g, in_features, device)
    absw = wg.abs().masked_fill(~mask, -1.0)
    sorted_abs, sorted_idx = torch.sort(absw, dim=-1, descending=True)
    sorted_abs = sorted_abs.clamp_min(0.0)
    prefix = torch.cumsum(sorted_abs, dim=-1)
    k = torch.arange(1, g + 1, device=device, dtype=prefix.dtype).view(1, 1, g)
    counts = mask.sum(dim=-1, keepdim=True)
    scores = prefix.square() / k
    scores = scores.masked_fill(k > counts, -float("inf"))
    best_idx = torch.argmax(scores, dim=-1)
    best_k = (best_idx + 1).to(prefix.dtype)
    best_prefix = torch.gather(prefix, -1, best_idx.unsqueeze(-1)).squeeze(-1)
    scales = (best_prefix / best_k).clamp_min(eps)
    rank = torch.arange(g, device=device).view(1, 1, g)
    active_sorted = rank <= best_idx.unsqueeze(-1)
    sorted_signs = torch.gather(torch.sign(wg), -1, sorted_idx)
    sorted_codes = sorted_signs * active_sorted.to(wg.dtype)
    codes = torch.zeros_like(wg)
    codes.scatter_(-1, sorted_idx, sorted_codes)
    codes = codes.masked_fill(~mask, 0.0)
    qg = codes * scales.unsqueeze(-1)
    q = _ungroup_view_2d(qg, in_features)
    return q.to(w.dtype), scales.to(w.dtype)


class _GroupwiseSTEBase(nn.Module):
    weight_master: nn.Parameter
    log_scales: nn.Parameter
    mode: str
    group_size: int
    threshold_factor: float
    forward_blend_alpha: float

    def _scales(self) -> Tensor:
        return torch.exp(self.log_scales).clamp(min=1e-8, max=1e4)

    @torch.no_grad()
    def _codes_from_matrix_detached(self, matrix: Tensor) -> Tensor:
        mat = matrix.detach().float()
        wg, pad, in_features = _group_view_2d(mat, self.group_size)
        out, groups, g = wg.shape
        scales = self._scales().detach().float().to(wg.device).unsqueeze(-1)
        if self.mode == "binary":
            codes = torch.where(wg >= 0, 1, -1).to(torch.int8)
        elif self.mode == "ternary":
            threshold = self.threshold_factor * scales
            codes = torch.zeros_like(wg, dtype=torch.int8)
            codes = torch.where(wg > threshold, torch.ones_like(codes), codes)
            codes = torch.where(wg < -threshold, -torch.ones_like(codes), codes)
        else:
            raise ValueError(f"Unknown quantization mode: {self.mode}")
        mask = _valid_group_mask(out, groups, g, in_features, wg.device)
        codes = codes.masked_fill(~mask, 0)
        return codes

    def _quantized_matrix_from_master(self) -> Tensor:
        master = self.weight_master
        wg, pad, in_features = _group_view_2d(master.float(), self.group_size)
        out, groups, g = wg.shape
        scales = self._scales().float().to(wg.device).unsqueeze(-1)
        with torch.no_grad():
            if self.mode == "binary":
                codes = torch.where(wg >= 0, 1.0, -1.0)
            elif self.mode == "ternary":
                threshold = self.threshold_factor * scales
                codes = torch.zeros_like(wg)
                codes = torch.where(wg > threshold, torch.ones_like(codes), codes)
                codes = torch.where(wg < -threshold, -torch.ones_like(codes), codes)
            else:
                raise ValueError(f"Unknown quantization mode: {self.mode}")
            mask = _valid_group_mask(out, groups, g, in_features, wg.device)
            codes = codes.masked_fill(~mask, 0.0)
        qg = codes.to(scales.dtype) * scales
        q = _ungroup_view_2d(qg, in_features).to(master.dtype)
        w_ste = q + (master - master.detach())
        alpha = float(self.forward_blend_alpha)
        if alpha <= 0.0:
            return master
        if alpha >= 1.0:
            return w_ste
        return (1.0 - alpha) * master + alpha * w_ste

    @torch.no_grad()
    def _quantized_matrix_detached(self) -> Tensor:
        master = self.weight_master.detach()
        wg, pad, in_features = _group_view_2d(master.float(), self.group_size)
        out, groups, g = wg.shape
        scales = self._scales().detach().float().to(wg.device).unsqueeze(-1)
        if self.mode == "binary":
            codes = torch.where(wg >= 0, 1.0, -1.0)
        elif self.mode == "ternary":
            threshold = self.threshold_factor * scales
            codes = torch.zeros_like(wg)
            codes = torch.where(wg > threshold, torch.ones_like(codes), codes)
            codes = torch.where(wg < -threshold, -torch.ones_like(codes), codes)
        else:
            raise ValueError(f"Unknown quantization mode: {self.mode}")
        mask = _valid_group_mask(out, groups, g, in_features, wg.device)
        codes = codes.masked_fill(~mask, 0.0)
        qg = codes.to(scales.dtype) * scales
        q = _ungroup_view_2d(qg, in_features)
        return q.to(self.weight_master.dtype)

    def _attach_bonsai_metadata(self) -> None:
        self.weight_master._bonsai_make_quantized = self._quantized_matrix_detached
        self.weight_master._bonsai_codes_from_weight = self._codes_from_matrix_detached
        self.weight_master._bonsai_get_scales = self._scales
        self.weight_master._bonsai_refresh_scales = self.reset_quant_scales
        self.weight_master._bonsai_quant_mode = self.mode
        self.weight_master._bonsai_group_size = self.group_size
        self.weight_master._bonsai_threshold_factor = self.threshold_factor

    def set_forward_blend_alpha(self, alpha: float) -> None:
        self.forward_blend_alpha = float(max(0.0, min(1.0, alpha)))


class QuantizedLinearSTE(_GroupwiseSTEBase):
    def __init__(self, in_features, out_features, bias=True, *, mode="ternary", group_size=128,
                 threshold_factor=0.5, init_weight=None, init_bias=None, init="mse_ternary",
                 forward_blend_alpha=1.0, dtype=torch.float32, device=None):
        super().__init__()
        if mode not in ("ternary", "binary"):
            raise ValueError("mode must be 'ternary' or 'binary'")
        self.in_features = int(in_features); self.out_features = int(out_features)
        self.mode = mode; self.group_size = int(group_size)
        self.threshold_factor = float(threshold_factor); self.forward_blend_alpha = float(forward_blend_alpha)
        if init_weight is None:
            weight = torch.empty(out_features, in_features, dtype=dtype, device=device)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        else:
            weight = init_weight.detach().to(device=device, dtype=dtype).clone()
        self.weight_master = nn.Parameter(weight.float())
        groups = _ceil_div(in_features, group_size)
        self.log_scales = nn.Parameter(torch.empty(out_features, groups, dtype=torch.float32, device=weight.device))
        if bias:
            if init_bias is None:
                self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=weight.device))
            else:
                self.bias = nn.Parameter(init_bias.detach().to(device=weight.device, dtype=dtype).clone())
        else:
            self.register_parameter("bias", None)
        self.reset_quant_scales(init=init)
        self._attach_bonsai_metadata()

    @classmethod
    def from_linear(cls, linear, *, mode="ternary", group_size=128, threshold_factor=0.5,
                    init="mse_ternary", forward_blend_alpha=1.0):
        return cls(linear.in_features, linear.out_features, linear.bias is not None, mode=mode,
                   group_size=group_size, threshold_factor=threshold_factor,
                   init_weight=linear.weight.detach(),
                   init_bias=None if linear.bias is None else linear.bias.detach(),
                   init=init, forward_blend_alpha=forward_blend_alpha,
                   dtype=linear.weight.dtype, device=linear.weight.device)

    @torch.no_grad()
    def reset_quant_scales(self, init="mse_ternary") -> None:
        w = self.weight_master.detach()
        if self.mode == "binary":
            _, scales = groupwise_binary_absmean_init(w, self.group_size)
        else:
            if init == "mse_ternary":
                _, scales = groupwise_mse_ternary_init(w, self.group_size)
            elif init == "absmean":
                _, scales = groupwise_binary_absmean_init(w, self.group_size)
            else:
                raise ValueError(f"Unknown init: {init}")
        self.log_scales.copy_(scales.float().clamp_min(1e-8).log())

    def forward(self, x: Tensor) -> Tensor:
        w = self._quantized_matrix_from_master().to(dtype=x.dtype)
        b = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, w, b)


class QuantizedConv1x1STE(_GroupwiseSTEBase):
    def __init__(self, in_channels, out_channels, bias=True, *, stride=1, padding=0, dilation=1,
                 mode="ternary", group_size=128, threshold_factor=0.5, init_weight=None, init_bias=None,
                 init="mse_ternary", forward_blend_alpha=1.0, dtype=torch.float32, device=None):
        super().__init__()
        if mode not in ("ternary", "binary"):
            raise ValueError("mode must be 'ternary' or 'binary'")
        self.in_channels = int(in_channels); self.out_channels = int(out_channels)
        self.kernel_size = (1, 1); self.stride = _as_tuple_2(stride); self.padding = _as_tuple_2(padding)
        self.dilation = _as_tuple_2(dilation); self.groups = 1
        self.mode = mode; self.group_size = int(group_size)
        self.threshold_factor = float(threshold_factor); self.forward_blend_alpha = float(forward_blend_alpha)
        if init_weight is None:
            w4 = torch.empty(out_channels, in_channels, 1, 1, dtype=dtype, device=device)
            nn.init.kaiming_uniform_(w4, a=math.sqrt(5))
        else:
            w4 = init_weight.detach().to(device=device, dtype=dtype).clone()
            if tuple(w4.shape[-2:]) != (1, 1):
                raise ValueError("QuantizedConv1x1STE only supports 1x1 conv weights")
        weight_2d = w4[:, :, 0, 0].contiguous()
        self.weight_master = nn.Parameter(weight_2d.float())
        n_groups = _ceil_div(in_channels, group_size)
        self.log_scales = nn.Parameter(torch.empty(out_channels, n_groups, dtype=torch.float32, device=weight_2d.device))
        if bias:
            if init_bias is None:
                self.bias = nn.Parameter(torch.zeros(out_channels, dtype=dtype, device=weight_2d.device))
            else:
                self.bias = nn.Parameter(init_bias.detach().to(device=weight_2d.device, dtype=dtype).clone())
        else:
            self.register_parameter("bias", None)
        self.reset_quant_scales(init=init)
        self._attach_bonsai_metadata()

    @classmethod
    def from_conv2d(cls, conv, *, mode="ternary", group_size=128, threshold_factor=0.5,
                    init="mse_ternary", forward_blend_alpha=1.0):
        if tuple(conv.kernel_size) != (1, 1):
            raise ValueError("Only 1x1 Conv2d can be replaced by QuantizedConv1x1STE")
        if conv.groups != 1:
            raise ValueError("Grouped/depthwise convs should be protected, not wrapped here")
        return cls(conv.in_channels, conv.out_channels, conv.bias is not None, stride=conv.stride,
                   padding=conv.padding, dilation=conv.dilation, mode=mode, group_size=group_size,
                   threshold_factor=threshold_factor, init_weight=conv.weight.detach(),
                   init_bias=None if conv.bias is None else conv.bias.detach(),
                   init=init, forward_blend_alpha=forward_blend_alpha,
                   dtype=conv.weight.dtype, device=conv.weight.device)

    @torch.no_grad()
    def reset_quant_scales(self, init="mse_ternary") -> None:
        w = self.weight_master.detach()
        if self.mode == "binary":
            _, scales = groupwise_binary_absmean_init(w, self.group_size)
        else:
            if init == "mse_ternary":
                _, scales = groupwise_mse_ternary_init(w, self.group_size)
            elif init == "absmean":
                _, scales = groupwise_binary_absmean_init(w, self.group_size)
            else:
                raise ValueError(f"Unknown init: {init}")
        self.log_scales.copy_(scales.float().clamp_min(1e-8).log())

    def forward(self, x: Tensor) -> Tensor:
        w2 = self._quantized_matrix_from_master().to(dtype=x.dtype)
        w4 = w2[:, :, None, None].contiguous()
        b = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.conv2d(x, w4, b, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)


def _get_parent_and_child(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def replace_modules_with_bonsai_quant(model, predicate, *, mode="ternary", group_size=128,
                                      threshold_factor=0.5, init="mse_ternary", forward_blend_alpha=1.0,
                                      include_linear=True, include_conv1x1=True) -> List[str]:
    replaced = []
    named = list(model.named_modules())
    for name, module in named:
        if name == "":
            continue
        if not predicate(name, module):
            continue
        new_module = None
        if include_linear and isinstance(module, nn.Linear):
            new_module = QuantizedLinearSTE.from_linear(module, mode=mode, group_size=group_size,
                threshold_factor=threshold_factor, init=init, forward_blend_alpha=forward_blend_alpha)
        elif include_conv1x1 and isinstance(module, nn.Conv2d):
            if tuple(module.kernel_size) == (1, 1) and module.groups == 1:
                new_module = QuantizedConv1x1STE.from_conv2d(module, mode=mode, group_size=group_size,
                    threshold_factor=threshold_factor, init=init, forward_blend_alpha=forward_blend_alpha)
        if new_module is not None:
            parent, child = _get_parent_and_child(model, name)
            setattr(parent, child, new_module)
            replaced.append(name)
    return replaced


def set_bonsai_forward_blend_alpha(model: nn.Module, alpha: float) -> None:
    for m in model.modules():
        if isinstance(m, _GroupwiseSTEBase):
            m.set_forward_blend_alpha(alpha)


def collect_bonsai_param_groups(model, *, muon_lr=2e-5, scale_lr=5e-5, island_lr=2e-6,
                                prox_rho=0.0, max_flip_rate=None, weight_decay=0.0) -> List[dict]:
    master_params, scale_params, other_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("weight_master"):
            master_params.append(p)
        elif name.endswith("log_scales"):
            scale_params.append(p)
        else:
            other_params.append(p)
    groups = []
    if master_params:
        groups.append(dict(params=master_params, role="muon", lr=muon_lr, weight_decay=weight_decay,
            prox_rho=prox_rho, max_flip_rate=max_flip_rate, momentum=0.95, nesterov=True,
            ns_steps=5, rms_match=True, state_dtype=torch.float32))
    if scale_params:
        groups.append(dict(params=scale_params, role="adamw", lr=scale_lr, weight_decay=0.0,
            betas=(0.9, 0.95), eps=1e-8))
    if other_params:
        groups.append(dict(params=other_params, role="adamw", lr=island_lr, weight_decay=0.0,
            betas=(0.9, 0.95), eps=1e-8))
    return groups


@torch.no_grad()
def newton_schulz_orthogonalize(g: Tensor, *, steps: int = 5, eps: float = 1e-7,
                                compute_dtype: Optional[torch.dtype] = None) -> Tensor:
    if g.ndim != 2:
        return g
    orig_dtype = g.dtype
    x = g.to(compute_dtype) if compute_dtype is not None else g.float()
    norm = torch.linalg.norm(x)
    if not torch.isfinite(norm) or norm.item() == 0.0:
        return torch.zeros_like(g)
    x = x / (norm + eps)
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    for _ in range(int(steps)):
        a = x @ x.T
        x = 1.5 * x - 0.5 * (a @ x)
    if transposed:
        x = x.T
    return x.to(orig_dtype)


class BonsaiMuon(torch.optim.Optimizer):
    def __init__(self, params, defaults=None):
        if defaults is None:
            defaults = dict(lr=2e-5, role="muon", weight_decay=0.0, momentum=0.95, nesterov=True,
                ns_steps=5, rms_match=True, prox_rho=0.0, max_flip_rate=None,
                state_dtype=torch.float32, betas=(0.9, 0.95), eps=1e-8)
        super().__init__(params, defaults)

    def set_prox_rho(self, rho: float) -> None:
        for group in self.param_groups:
            if group.get("role", "muon") == "muon":
                group["prox_rho"] = float(rho)

    def set_max_flip_rate(self, max_flip_rate) -> None:
        for group in self.param_groups:
            if group.get("role", "muon") == "muon":
                group["max_flip_rate"] = max_flip_rate

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            role = group.get("role", "muon")
            for p in group["params"]:
                if p.grad is None:
                    continue
                if role == "adamw" or p.ndim != 2:
                    self._adamw_step(p, group)
                elif role == "muon":
                    self._muon_step(p, group)
                else:
                    raise ValueError(f"Unknown optimizer role: {role}")
        return loss

    @torch.no_grad()
    def _adamw_step(self, p, group) -> None:
        grad = p.grad
        if grad is None:
            return
        if grad.is_sparse:
            raise RuntimeError("BonsaiMuon AdamW path does not support sparse gradients")
        lr = float(group.get("lr", 1e-3))
        beta1, beta2 = group.get("betas", (0.9, 0.95))
        eps = float(group.get("eps", 1e-8))
        wd = float(group.get("weight_decay", 0.0))
        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        exp_avg = state["exp_avg"]; exp_avg_sq = state["exp_avg_sq"]
        state["step"] += 1; step = state["step"]
        if wd != 0.0:
            p.mul_(1.0 - lr * wd)
        grad_f = grad.to(dtype=p.dtype)
        exp_avg.mul_(beta1).add_(grad_f, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad_f, grad_f, value=1.0 - beta2)
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step
        denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
        step_size = lr / bias_correction1
        p.addcdiv_(exp_avg, denom, value=-step_size)

    @torch.no_grad()
    def _muon_step(self, p, group) -> None:
        grad = p.grad
        if grad is None:
            return
        if grad.is_sparse:
            raise RuntimeError("BonsaiMuon Muon path does not support sparse gradients")
        lr = float(group.get("lr", 2e-5))
        wd = float(group.get("weight_decay", 0.0))
        momentum = float(group.get("momentum", 0.95))
        nesterov = bool(group.get("nesterov", True))
        ns_steps = int(group.get("ns_steps", 5))
        rms_match = bool(group.get("rms_match", True))
        prox_rho = float(group.get("prox_rho", 0.0))
        state_dtype = group.get("state_dtype", torch.float32)
        state = self.state[p]
        if len(state) == 0:
            state["momentum_buffer"] = torch.zeros_like(p, dtype=state_dtype, memory_format=torch.preserve_format)
        if wd != 0.0:
            p.mul_(1.0 - lr * wd)
        buf = state["momentum_buffer"]
        grad_buf_dtype = grad.detach().to(dtype=buf.dtype)
        buf.mul_(momentum).add_(grad_buf_dtype)
        if nesterov:
            update_raw = grad.detach().float() + momentum * buf.float()
        else:
            update_raw = buf.float()
        ns_dtype = group.get("ns_dtype", None)  # e.g. bf16: NS on tensor cores
        update = newton_schulz_orthogonalize(update_raw, steps=ns_steps, compute_dtype=ns_dtype)
        update = update.float()
        if rms_match:
            raw_rms = _rms(update_raw); upd_rms = _rms(update)
            update = update * (raw_rms / (upd_rms + 1e-12))
        old_p = p.detach().float()
        proposal = old_p - lr * update.float()
        proposal = self._apply_flip_trust_region(p=p, old_p=old_p, proposal=proposal, group=group)
        p.copy_(proposal.to(dtype=p.dtype))
        if prox_rho > 0.0:
            self._apply_proximal_pull(p, prox_rho)

    @torch.no_grad()
    def _apply_proximal_pull(self, p, rho) -> None:
        make_q = getattr(p, "_bonsai_make_quantized", None)
        if not callable(make_q):
            return
        q = make_q().to(device=p.device, dtype=p.dtype)
        if q.shape != p.shape:
            raise RuntimeError(f"Quantized callback returned {q.shape}, expected {p.shape}")
        rho = float(max(0.0, min(1.0, rho)))
        p.lerp_(q, rho)

    @torch.no_grad()
    def _apply_flip_trust_region(self, *, p, old_p, proposal, group) -> Tensor:
        max_flip_rate = group.get("max_flip_rate", None)
        if max_flip_rate is None:
            return proposal
        code_fn = getattr(p, "_bonsai_codes_from_weight", None)
        if not callable(code_fn):
            return proposal
        max_flip_rate = float(max_flip_rate)
        if max_flip_rate <= 0:
            return old_p
        old_codes = code_fn(old_p); new_codes = code_fn(proposal)
        flip = (old_codes != new_codes).float().mean().item()
        if flip <= max_flip_rate:
            return proposal
        lo, hi = 0.0, 1.0; best = old_p
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            cand = old_p + mid * (proposal - old_p)
            cand_codes = code_fn(cand)
            cand_flip = (old_codes != cand_codes).float().mean().item()
            if cand_flip <= max_flip_rate:
                best = cand; lo = mid
            else:
                hi = mid
        return best


@torch.no_grad()
def bonsai_code_stats(model: nn.Module) -> dict:
    total = zeros = pos = neg = 0
    for m in model.modules():
        if not isinstance(m, _GroupwiseSTEBase):
            continue
        codes = m._codes_from_matrix_detached(m.weight_master)
        total += codes.numel(); zeros += (codes == 0).sum().item()
        pos += (codes > 0).sum().item(); neg += (codes < 0).sum().item()
    if total == 0:
        return dict(total=0, zero_frac=0.0, pos_frac=0.0, neg_frac=0.0)
    return dict(total=total, zero_frac=zeros / total, pos_frac=pos / total, neg_frac=neg / total)
