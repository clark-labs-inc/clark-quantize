"""All-2D hard-forward binary/ternary QAT for Qwen AgentWorld.

This is the Prism/Bonsai-shaped experiment:
  * every Linear/Embedding 2D weight runs as binary/ternary g128 in the forward pass;
  * only a selected rotating window owns trainable master weights;
  * 1D norms can remain continuous and optionally train in the selected window.

It deliberately avoids the old patchwork pattern where only one operator family
is quantized while the rest of the model stays BF16.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import re
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clark_air.quant.master_weight_qat import ste_quantized_weight
from clark_air.quant.optimal_ternary import quantize_ternary_mse
from clark_quantize.bonsai_qat8 import BonsaiQAT8
from train_mlp_triplet_survival_qat import (
    anti_collapse_loss,
    hidden_loss,
    load_model,
    register_hidden_hooks,
    row_to_inputs,
    topk_teacher_kl,
)
from weight_patch import apply_weight_patch


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def get_parent_and_attr(model: torch.nn.Module, module_name: str) -> tuple[torch.nn.Module, str]:
    parent_name, _, attr = module_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, attr


def parse_int_csv(value: str | None) -> set[int]:
    if not value:
        return set()
    if value.strip().lower() == "all":
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def parse_task_caps(values: list[str] | None) -> dict[str, float]:
    caps: dict[str, float] = {}
    for value in values or []:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(f"task cap must be task:fraction, got {part!r}")
            task, raw_fraction = part.split(":", 1)
            fraction = float(raw_fraction)
            if fraction < 0.0 or fraction > 1.0:
                raise ValueError(f"task cap fraction must be in [0,1], got {part!r}")
            caps[task.strip()] = fraction
    return caps


def parse_torch_dtype(value: str) -> torch.dtype:
    if value == "float32":
        return torch.float32
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {value}")


def layer_id(module_name: str) -> int | None:
    match = LAYER_RE.search(module_name)
    return int(match.group(1)) if match else None


def canonical_weight_name(module_name: str) -> str:
    return f"{module_name}.weight"


def is_head_module(module_name: str) -> bool:
    return module_name == "lm_head" or module_name.endswith(".lm_head")


def is_embedding_module(module_name: str) -> bool:
    return module_name.endswith("embed_tokens")


def module_class(module_name: str) -> str:
    if is_embedding_module(module_name):
        return "embed"
    if is_head_module(module_name):
        return "head"
    if module_name.endswith(".gate_proj"):
        return "mlp_gate"
    if module_name.endswith(".up_proj"):
        return "mlp_up"
    if module_name.endswith(".down_proj"):
        return "mlp_down"
    if any(module_name.endswith(suffix) for suffix in (".q_proj", ".k_proj", ".v_proj", ".o_proj")):
        return "attn"
    if "self_attn" in module_name or "attention" in module_name:
        return "attn_other"
    if "mlp" in module_name:
        return "mlp_other"
    return "other_2d"


def should_skip_quant_module(module_name: str, args: argparse.Namespace) -> bool:
    if args.skip_quant_embed and is_embedding_module(module_name):
        return True
    if args.skip_quant_head and is_head_module(module_name):
        return True
    return False


def should_train_module(module_name: str, args: argparse.Namespace) -> bool:
    if args.train_regex and re.search(args.train_regex, module_name):
        return True
    if is_embedding_module(module_name):
        return args.train_embed
    if is_head_module(module_name):
        return args.train_head
    lid = layer_id(module_name)
    if lid is not None and args.train_all_layers:
        return True
    return lid is not None and lid in args.train_layer_set


def alpha_quantized_weight(master: torch.Tensor, *, mode: str, group_size: int, alpha: float) -> torch.Tensor:
    if mode == "ternary":
        q = dequantize_ternary_chunked(master, group_size=group_size)
    elif mode == "binary":
        q = dequantize_binary_chunked(master, group_size=group_size)
    else:
        raise ValueError(f"unsupported quantization mode: {mode}")
    if alpha >= 1.0:
        return q.detach() + (master - master.detach())
    with torch.no_grad():
        forward = (1.0 - alpha) * master.detach() + alpha * q
    return forward + (master - master.detach())


def quantized_slice(source: torch.Tensor, *, mode: str, group_size: int) -> torch.Tensor:
    if mode == "ternary":
        return quantize_ternary_mse(source.detach().float(), group_size=group_size).dequantize()
    if mode == "binary":
        chunk = source.detach().float()
        pad = (-chunk.numel()) % group_size
        if pad:
            chunk = F.pad(chunk, (0, pad))
        view = chunk.view(-1, group_size)
        scales = view.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.where(view >= 0, scales, -scales).reshape(-1)[: source.numel()]
    raise ValueError(f"unsupported quantization mode: {mode}")


def sampled_codebook_loss(
    master: torch.Tensor,
    *,
    mode: str,
    group_size: int,
    groups_per_sample: int,
    sample_index: int,
) -> torch.Tensor:
    flat = master.reshape(-1)
    total_groups = max(1, (flat.numel() + group_size - 1) // group_size)
    groups = min(max(1, groups_per_sample), total_groups)
    start_group = (sample_index * groups) % total_groups
    start = start_group * group_size
    end = min(start + groups * group_size, flat.numel())
    source = flat[start:end]
    q = quantized_slice(source, mode=mode, group_size=group_size).to(device=master.device, dtype=torch.float32)
    denom = source.detach().float().square().mean().clamp_min(1e-8)
    return (source.float() - q).square().mean() / denom


@torch.no_grad()
def apply_chunked_prox(param: torch.Tensor, *, mode: str, group_size: int, prox_rho: float, chunk_groups: int = 8192) -> None:
    flat = param.reshape(-1)
    chunk_elems = max(group_size, chunk_groups * group_size)
    for start in range(0, flat.numel(), chunk_elems):
        end = min(start + chunk_elems, flat.numel())
        source = flat[start:end]
        q = quantized_slice(source, mode=mode, group_size=group_size)
        source.lerp_(q.to(device=param.device, dtype=param.dtype), max(0.0, min(1.0, prox_rho)))


@torch.no_grad()
def dequantize_ternary_chunked(
    weight: torch.Tensor,
    *,
    group_size: int,
    chunk_groups: int = 32768,
) -> torch.Tensor:
    if weight.numel() % group_size:
        return quantize_ternary_mse(weight.detach().float(), group_size=group_size).dequantize().to(weight.device, weight.dtype)
    source = weight.detach().reshape(-1)
    out = torch.empty_like(source, dtype=weight.dtype, device=weight.device)
    chunk_elems = chunk_groups * group_size
    for start in range(0, source.numel(), chunk_elems):
        end = min(start + chunk_elems, source.numel())
        q = quantize_ternary_mse(source[start:end].float(), group_size=group_size).dequantize()
        out[start:end].copy_(q.to(device=weight.device, dtype=weight.dtype))
    return out.reshape_as(weight)


@torch.no_grad()
def dequantize_binary_chunked(
    weight: torch.Tensor,
    *,
    group_size: int,
    chunk_groups: int = 32768,
) -> torch.Tensor:
    source = weight.detach().reshape(-1)
    out = torch.empty_like(source, dtype=weight.dtype, device=weight.device)
    chunk_elems = chunk_groups * group_size
    for start in range(0, source.numel(), chunk_elems):
        end = min(start + chunk_elems, source.numel())
        chunk = source[start:end].float()
        pad = (-chunk.numel()) % group_size
        if pad:
            chunk = F.pad(chunk, (0, pad))
        view = chunk.view(-1, group_size)
        scales = view.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        q = torch.where(view >= 0, scales, -scales).reshape(-1)[: end - start]
        out[start:end].copy_(q.to(device=weight.device, dtype=weight.dtype))
    return out.reshape_as(weight)


def _quantize_chunk_for_forward(
    weight: torch.Tensor,
    *,
    mode: str,
    group_size: int,
    alpha: float,
) -> torch.Tensor:
    if mode == "none":
        return weight.detach().float()
    if mode == "ternary":
        q = dequantize_ternary_chunked(weight.detach().float(), group_size=group_size)
    elif mode == "binary":
        q = dequantize_binary_chunked(weight.detach().float(), group_size=group_size)
    else:
        raise ValueError(f"unsupported chunked head mode: {mode}")
    if alpha >= 1.0:
        return q.float()
    return (1.0 - alpha) * weight.detach().float() + alpha * q.float()


class ChunkedTopKOtherHeadKL(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        teacher_indices: torch.Tensor,
        format_ids: torch.Tensor,
        mode: str,
        group_size: int,
        alpha: float,
        chunk_rows: int,
        format_boost: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.ndim != 2:
            raise ValueError(f"chunked head hidden must be [tokens, dim], got {tuple(hidden.shape)}")
        if weight.ndim != 2:
            raise ValueError(f"chunked head weight must be [vocab, dim], got {tuple(weight.shape)}")
        if hidden.shape[-1] != weight.shape[-1]:
            raise ValueError(f"hidden dim {hidden.shape[-1]} != head dim {weight.shape[-1]}")
        if weight.shape[-1] % group_size:
            raise ValueError("chunked head requires row-local quant groups along hidden dim")
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        tokens = hidden.shape[0]
        vocab = weight.shape[0]
        topk = teacher_logprobs.shape[-1]
        hidden_f = hidden.detach().float()
        indices = teacher_indices.to(device=hidden.device, dtype=torch.long)
        teacher_logp = teacher_logprobs.to(device=hidden.device, dtype=torch.float32)
        teacher_prob = teacher_logp.exp()
        teacher_top_mass = teacher_prob.sum(dim=-1).clamp(max=1.0)
        teacher_other = (1.0 - teacher_top_mass).clamp_min(1e-8)
        lse = torch.full((tokens,), -float("inf"), device=hidden.device, dtype=torch.float32)
        gathered = torch.empty((tokens, topk), device=hidden.device, dtype=torch.float32)
        gathered.fill_(-float("inf"))
        best_values = torch.full((tokens,), -float("inf"), device=hidden.device, dtype=torch.float32)
        best_indices = torch.zeros((tokens,), device=hidden.device, dtype=torch.long)
        for start in range(0, vocab, chunk_rows):
            end = min(start + chunk_rows, vocab)
            w = _quantize_chunk_for_forward(
                weight[start:end],
                mode=mode,
                group_size=group_size,
                alpha=alpha,
            ).to(device=hidden.device)
            logits = hidden_f.matmul(w.t())
            chunk_lse = torch.logsumexp(logits, dim=-1)
            lse = torch.logaddexp(lse, chunk_lse)
            chunk_best, chunk_arg = logits.max(dim=-1)
            better = chunk_best > best_values
            best_values = torch.where(better, chunk_best, best_values)
            best_indices = torch.where(better, chunk_arg + start, best_indices)
            local = indices - start
            mask = (local >= 0) & (local < (end - start))
            safe_local = local.clamp(0, end - start - 1)
            chunk_gathered = logits.gather(dim=-1, index=safe_local)
            gathered = torch.where(mask, chunk_gathered, gathered)
            del w, logits
        student_logp = gathered - lse.reshape(tokens, 1)
        student_prob = student_logp.exp()
        student_top_mass = student_prob.sum(dim=-1).clamp(max=1.0)
        student_other = (1.0 - student_top_mass).clamp_min(1e-8)
        top_term = (teacher_prob * (teacher_logp - student_logp)).sum(dim=-1)
        other_term = teacher_other * (teacher_other.log() - student_other.log())
        per_pos = top_term + other_term
        weights = torch.ones_like(per_pos)
        format_mass = torch.zeros_like(per_pos)
        if format_ids.numel():
            matches = indices.unsqueeze(-1).eq(format_ids.reshape(1, 1, -1)).any(dim=-1)
            format_mass = (teacher_prob * matches.float()).sum(dim=-1)
            weights = weights + format_boost * format_mass
        loss = (per_pos * weights).mean()
        tail_scale = 1.0 - teacher_other / student_other
        ctx.save_for_backward(
            hidden.detach(),
            weight,
            indices,
            teacher_prob,
            lse.detach(),
            weights.detach(),
            tail_scale.detach(),
        )
        ctx.mode = mode
        ctx.group_size = int(group_size)
        ctx.alpha = float(alpha)
        ctx.chunk_rows = int(chunk_rows)
        stats = torch.stack(
            [
                per_pos.mean().detach(),
                loss.detach(),
                format_mass.mean().detach(),
                (best_indices == indices[:, 0]).float().mean().detach(),
                student_top_mass.mean().detach(),
                teacher_top_mass.mean().detach(),
                student_top_mass.mean().detach(),
                torch.ones((), device=hidden.device, dtype=torch.float32),
            ]
        )
        ctx.mark_non_differentiable(stats)
        return loss, stats

    @staticmethod
    def backward(ctx: Any, grad_loss: torch.Tensor, grad_stats: torch.Tensor | None = None) -> tuple[Any, ...]:
        del grad_stats
        hidden, weight, indices, teacher_prob, lse, weights, tail_scale = ctx.saved_tensors
        hidden_f = hidden.float()
        tokens = hidden.shape[0]
        vocab = weight.shape[0]
        scale = (weights * grad_loss.float() / max(1, tokens)).reshape(tokens, 1)
        grad_hidden = torch.zeros_like(hidden_f)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32) if ctx.needs_input_grad[1] else None
        for start in range(0, vocab, ctx.chunk_rows):
            end = min(start + ctx.chunk_rows, vocab)
            w = _quantize_chunk_for_forward(
                weight[start:end],
                mode=ctx.mode,
                group_size=ctx.group_size,
                alpha=ctx.alpha,
            ).to(device=hidden.device)
            logits = hidden_f.matmul(w.t())
            probs = (logits - lse.reshape(tokens, 1)).exp()
            grad_logits = probs * tail_scale.reshape(tokens, 1)
            local = indices - start
            mask = (local >= 0) & (local < (end - start))
            safe_local = local.clamp(0, end - start - 1)
            top_probs = probs.gather(dim=-1, index=safe_local)
            correction = top_probs * (1.0 - tail_scale.reshape(tokens, 1)) - teacher_prob
            grad_logits.scatter_add_(dim=-1, index=safe_local, src=torch.where(mask, correction, torch.zeros_like(correction)))
            grad_logits = grad_logits * scale
            grad_hidden.add_(grad_logits.matmul(w))
            if grad_weight is not None:
                grad_weight[start:end].copy_(grad_logits.t().matmul(hidden_f).to(dtype=grad_weight.dtype))
            del w, logits, probs, grad_logits
        return (
            grad_hidden.to(dtype=hidden.dtype),
            grad_weight.to(dtype=weight.dtype) if grad_weight is not None else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def chunked_head_topk_other_kl(
    hidden: torch.Tensor,
    head: torch.nn.Module,
    teacher_values: torch.Tensor,
    teacher_indices: torch.Tensor,
    *,
    teacher_logsumexp: torch.Tensor | None = None,
    teacher_logprobs: torch.Tensor | None = None,
    format_ids: torch.Tensor,
    format_boost: float,
    chunk_rows: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    if isinstance(head, TrainQuantLinear):
        if head.bias is not None:
            raise RuntimeError("chunked head KD does not support lm_head bias yet")
        weight = head.master
        mode = head.mode
        group_size = head.group_size
        alpha = head.alpha
    elif isinstance(head, FrozenQuantLinear):
        if head.bias is not None:
            raise RuntimeError("chunked head KD does not support lm_head bias yet")
        weight = head.weight
        mode = "none"
        group_size = 1
        alpha = 1.0
    elif isinstance(head, torch.nn.Linear):
        if head.bias is not None:
            raise RuntimeError("chunked head KD does not support lm_head bias yet")
        weight = head.weight
        mode = "none"
        group_size = 1
        alpha = 1.0
    else:
        raise RuntimeError(f"unsupported lm_head type for chunked KD: {type(head)}")
    if teacher_logprobs is None:
        if teacher_logsumexp is None:
            raise RuntimeError("chunked head KD requires teacher_logsumexp or teacher_top_logprobs in the cache row")
        teacher_logprobs = teacher_values.to(dtype=torch.float32) - teacher_logsumexp.reshape(-1, 1).to(dtype=torch.float32)
    loss, stats_tensor = ChunkedTopKOtherHeadKL.apply(
        hidden.squeeze(0).float(),
        weight,
        teacher_logprobs,
        teacher_indices,
        format_ids,
        mode,
        group_size,
        alpha,
        chunk_rows,
        format_boost,
    )
    stats_values = [float(value.detach().cpu()) for value in stats_tensor]
    return loss, {
        "kl": stats_values[0],
        "weighted_kl": stats_values[1],
        "format_mass": stats_values[2],
        "top1": stats_values[3],
        "mass": stats_values[4],
        "teacher_top_mass": stats_values[5],
        "student_top_mass": stats_values[6],
        "other_bucket": stats_values[7],
    }


class TrainQuantLinear(torch.nn.Module):
    def __init__(
        self,
        source: torch.nn.Linear,
        *,
        mode: str,
        group_size: int,
        alpha: float,
        master_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.master = torch.nn.Parameter(source.weight.detach().to(dtype=master_dtype).clone())
        self.mode = mode
        self.group_size = group_size
        self.alpha = alpha
        if source.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", source.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = alpha_quantized_weight(self.master, mode=self.mode, group_size=self.group_size, alpha=self.alpha)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight.to(dtype=x.dtype), bias)

    def codebook_loss(self, *, groups_per_sample: int = 4096, sample_index: int = 0) -> torch.Tensor:
        return sampled_codebook_loss(
            self.master,
            mode=self.mode,
            group_size=self.group_size,
            groups_per_sample=groups_per_sample,
            sample_index=sample_index,
        )

    @torch.no_grad()
    def hard_weight(self) -> torch.Tensor:
        return alpha_quantized_weight(self.master, mode=self.mode, group_size=self.group_size, alpha=1.0).cpu().to(torch.bfloat16)


class FrozenQuantLinear(torch.nn.Module):
    def __init__(self, source: torch.nn.Linear, *, mode: str, group_size: int) -> None:
        super().__init__()
        if mode == "ternary":
            q = dequantize_ternary_chunked(source.weight, group_size=group_size)
        elif mode == "binary":
            q = dequantize_binary_chunked(source.weight, group_size=group_size)
        else:
            raise ValueError(f"unsupported quantization mode: {mode}")
        self.register_buffer("weight", q.to(device=source.weight.device, dtype=source.weight.dtype))
        if source.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", source.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(dtype=x.dtype), bias)

    @torch.no_grad()
    def hard_weight(self) -> torch.Tensor:
        return self.weight.detach().cpu().to(torch.bfloat16)


class TrainQuantEmbedding(torch.nn.Module):
    def __init__(
        self,
        source: torch.nn.Embedding,
        *,
        mode: str,
        group_size: int,
        alpha: float,
        master_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.master = torch.nn.Parameter(source.weight.detach().to(dtype=master_dtype).clone())
        self.mode = mode
        self.group_size = group_size
        self.alpha = alpha
        self.padding_idx = source.padding_idx
        self.max_norm = source.max_norm
        self.norm_type = source.norm_type
        self.scale_grad_by_freq = source.scale_grad_by_freq
        self.sparse = source.sparse
        self.forward_dtype = source.weight.dtype

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None or self.scale_grad_by_freq or self.sparse:
            weight = alpha_quantized_weight(self.master, mode=self.mode, group_size=self.group_size, alpha=self.alpha)
            return F.embedding(
                input_ids,
                weight.to(dtype=self.forward_dtype),
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
        flat_ids = input_ids.reshape(-1)
        unique_ids, inverse = torch.unique(flat_ids, sorted=True, return_inverse=True)
        row_master = self.master.index_select(0, unique_ids)
        row_weight = alpha_quantized_weight(row_master, mode=self.mode, group_size=self.group_size, alpha=self.alpha)
        out = row_weight.to(dtype=self.forward_dtype).index_select(0, inverse)
        return out.reshape(*input_ids.shape, row_weight.shape[-1])

    def codebook_loss(self, *, groups_per_sample: int = 4096, sample_index: int = 0) -> torch.Tensor:
        return sampled_codebook_loss(
            self.master,
            mode=self.mode,
            group_size=self.group_size,
            groups_per_sample=groups_per_sample,
            sample_index=sample_index,
        )

    @torch.no_grad()
    def hard_weight(self) -> torch.Tensor:
        return alpha_quantized_weight(self.master, mode=self.mode, group_size=self.group_size, alpha=1.0).cpu().to(torch.bfloat16)


class FrozenQuantEmbedding(torch.nn.Module):
    def __init__(self, source: torch.nn.Embedding, *, mode: str, group_size: int) -> None:
        super().__init__()
        if mode == "ternary":
            q = dequantize_ternary_chunked(source.weight, group_size=group_size)
        elif mode == "binary":
            q = dequantize_binary_chunked(source.weight, group_size=group_size)
        else:
            raise ValueError(f"unsupported quantization mode: {mode}")
        self.register_buffer("weight", q.to(device=source.weight.device, dtype=source.weight.dtype))
        self.padding_idx = source.padding_idx
        self.max_norm = source.max_norm
        self.norm_type = source.norm_type
        self.scale_grad_by_freq = source.scale_grad_by_freq
        self.sparse = source.sparse

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            input_ids,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    @torch.no_grad()
    def hard_weight(self) -> torch.Tensor:
        return self.weight.detach().cpu().to(torch.bfloat16)


QuantModule = TrainQuantLinear | FrozenQuantLinear | TrainQuantEmbedding | FrozenQuantEmbedding


class NonReentrantCheckpointLayer(torch.nn.Module):
    """Checkpoint a transformer block without relying on HF model internals.

    The trainable quantized Linear path materializes a full fake-quantized
    weight during forward. Without a checkpoint boundary, autograd keeps those
    weights alive for every block until backward. Wrapping each block bounds
    the live fake-quantized weight set to the current block's recompute.
    """

    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        def run(*inner_args):
            return self.layer(*inner_args, **kwargs)

        return checkpoint(
            run,
            *args,
            use_reentrant=False,
            preserve_rng_state=False,
        )


def enable_explicit_layer_checkpointing(model: torch.nn.Module) -> int:
    """Wrap the Qwen language-model layer ModuleList with checkpoint modules."""

    candidate_paths = [
        "model.language_model.layers",
        "language_model.layers",
        "model.layers",
        "language_model.model.layers",
    ]
    layers = None
    for path in candidate_paths:
        try:
            candidate = model.get_submodule(path)
        except AttributeError:
            continue
        if isinstance(candidate, torch.nn.ModuleList):
            layers = candidate
            break
    if layers is None:
        raise RuntimeError(
            "could not find a language layer ModuleList for explicit checkpointing; "
            f"tried {candidate_paths}"
        )
    wrapped = 0
    for index, layer in enumerate(layers):
        if isinstance(layer, NonReentrantCheckpointLayer):
            continue
        layers[index] = NonReentrantCheckpointLayer(layer)
        wrapped += 1
    return wrapped


@torch.no_grad()
def _group_normalized_update(grad: torch.Tensor, *, group_size: int, eps: float) -> torch.Tensor:
    flat = grad.detach().float().reshape(-1)
    if group_size <= 0:
        rms = flat.square().mean().sqrt().clamp_min(eps)
        return (flat / rms).reshape_as(grad)
    pad = (-flat.numel()) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))
    view = flat.view(-1, group_size)
    rms = view.square().mean(dim=1, keepdim=True).sqrt().clamp_min(eps)
    update = (view / rms).reshape(-1)
    if pad:
        update = update[:-pad]
    return update.reshape_as(grad)


class GroupMirrorSGD(torch.optim.Optimizer):
    """No-moment mirror-style optimizer for hard-forward codebook QAT.

    Matrix updates are normalized by the same flat groups used by the current
    ternary/binary quantizer. This deliberately carries no Adam moments or Muon
    matrices, so global all-2D runs can spend memory on masters and gradients.
    """

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        group_size: int,
        eps: float,
        prox_rho: float,
        max_update_rms: float,
        group_adagrad: bool,
    ) -> None:
        super().__init__(
            [{"params": params}],
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "group_size": group_size,
                "eps": eps,
                "prox_rho": prox_rho,
                "max_update_rms": max_update_rms,
                "group_adagrad": group_adagrad,
            },
        )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        group = self.param_groups[0]
        lr = float(group["lr"])
        wd = float(group["weight_decay"])
        group_size = int(group["group_size"])
        eps = float(group["eps"])
        prox_rho = float(group["prox_rho"])
        max_update_rms = float(group["max_update_rms"])
        group_adagrad = bool(group["group_adagrad"])
        for param in group["params"]:
            if param.grad is None:
                continue
            if param.grad.is_sparse:
                raise RuntimeError("GroupMirrorSGD does not support sparse gradients")
            if group_adagrad and param.ndim >= 2:
                grad_flat = param.grad.detach().float().reshape(-1)
                pad = (-grad_flat.numel()) % group_size
                if pad:
                    grad_flat = F.pad(grad_flat, (0, pad))
                grad_view = grad_flat.view(-1, group_size)
                group_sq = grad_view.square().mean(dim=1)
                state = self.state[param]
                if "group_accum" not in state:
                    state["group_accum"] = torch.zeros_like(group_sq)
                accum = state["group_accum"]
                accum.add_(group_sq)
                update = (grad_view / accum.sqrt().clamp_min(eps).view(-1, 1)).reshape(-1)
                if pad:
                    update = update[:-pad]
                update = update.reshape_as(param.grad)
            else:
                update = _group_normalized_update(
                    param.grad,
                    group_size=group_size if param.ndim >= 2 else 0,
                    eps=eps,
                )
            if max_update_rms > 0.0:
                rms = torch.linalg.vector_norm(update.detach().float()) / math.sqrt(update.numel())
                if torch.isfinite(rms) and rms > max_update_rms:
                    update.mul_(max_update_rms / rms.clamp_min(eps))
            if wd:
                param.mul_(1.0 - lr * wd)
            param.add_(update.to(dtype=param.dtype), alpha=-lr)
            if prox_rho > 0.0:
                quant_mode = getattr(param, "_quant_mode", None)
                quant_group_size = getattr(param, "_quant_group_size", None)
                if quant_mode is not None and quant_group_size is not None:
                    apply_chunked_prox(
                        param,
                        mode=str(quant_mode),
                        group_size=int(quant_group_size),
                        prox_rho=prox_rho,
                    )
        return loss


class FactoredAdafactorQAT(torch.optim.Optimizer):
    """Low-state Adafactor-style optimizer for global hard-forward QAT.

    Large 2D masters keep row/column second-moment estimates instead of full
    Adam moments. Tiny 1D tensors, mostly norms, use an unfactored variance.
    """

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        beta2: float,
        eps: float,
        prox_rho: float,
        max_update_rms: float,
        update_chunk_rows: int,
    ) -> None:
        if beta2 < 0.0 or beta2 >= 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        super().__init__(
            [{"params": params}],
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "beta2": beta2,
                "eps": eps,
                "prox_rho": prox_rho,
                "max_update_rms": max_update_rms,
                "update_chunk_rows": update_chunk_rows,
            },
        )

    @torch.no_grad()
    def step_param(self, param: torch.nn.Parameter, group: dict[str, Any] | None = None) -> None:
        if group is None:
            group = self.param_groups[0]
        lr = float(group["lr"])
        wd = float(group["weight_decay"])
        beta2 = float(group["beta2"])
        eps = float(group["eps"])
        prox_rho = float(group["prox_rho"])
        max_update_rms = float(group["max_update_rms"])
        update_chunk_rows = max(1, int(group["update_chunk_rows"]))
        if param.grad is None:
            return
        if param.grad.is_sparse:
            raise RuntimeError("FactoredAdafactorQAT does not support sparse gradients")
        grad = param.grad.detach().float()
        state = self.state[param]
        if grad.ndim == 2:
            rows, cols = grad.shape
            row_mean = torch.empty(rows, device=grad.device, dtype=torch.float32)
            col_sum = torch.zeros(cols, device=grad.device, dtype=torch.float32)
            for start in range(0, rows, update_chunk_rows):
                end = min(start + update_chunk_rows, rows)
                grad_chunk = grad[start:end]
                sq = grad_chunk.square()
                row_mean[start:end] = sq.mean(dim=1)
                col_sum.add_(sq.sum(dim=0))
            col_mean = col_sum / max(rows, 1)
            if "row_accum" not in state:
                state["row_accum"] = torch.zeros_like(row_mean)
                state["col_accum"] = torch.zeros_like(col_mean)
            row_accum = state["row_accum"]
            col_accum = state["col_accum"]
            row_accum.mul_(beta2).add_(row_mean, alpha=1.0 - beta2)
            col_accum.mul_(beta2).add_(col_mean, alpha=1.0 - beta2)
            row_factor = row_accum / row_accum.mean().clamp_min(eps)
            update_scale = 1.0
            if max_update_rms > 0.0:
                update_sq_sum = 0.0
                for start in range(0, rows, update_chunk_rows):
                    end = min(start + update_chunk_rows, rows)
                    variance = row_factor[start:end, None] * col_accum[None, :]
                    update = grad[start:end] / variance.sqrt().clamp_min(eps)
                    update_sq_sum += float(update.detach().square().sum().item())
                rms = math.sqrt(update_sq_sum / max(grad.numel(), 1))
                if math.isfinite(rms) and rms > max_update_rms:
                    update_scale = max_update_rms / max(rms, eps)
            if wd:
                param.mul_(1.0 - lr * wd)
            for start in range(0, rows, update_chunk_rows):
                end = min(start + update_chunk_rows, rows)
                variance = row_factor[start:end, None] * col_accum[None, :]
                update = grad[start:end] / variance.sqrt().clamp_min(eps)
                if update_scale != 1.0:
                    update.mul_(update_scale)
                param[start:end].add_(update.to(dtype=param.dtype), alpha=-lr)
            if prox_rho > 0.0:
                quant_mode = getattr(param, "_quant_mode", None)
                quant_group_size = getattr(param, "_quant_group_size", None)
                if quant_mode is not None and quant_group_size is not None:
                    apply_chunked_prox(
                        param,
                        mode=str(quant_mode),
                        group_size=int(quant_group_size),
                        prox_rho=prox_rho,
                    )
            return
        else:
            grad_sq = grad.square()
            if "accum" not in state:
                state["accum"] = torch.zeros_like(grad_sq)
            accum = state["accum"]
            accum.mul_(beta2).add_(grad_sq, alpha=1.0 - beta2)
            update = grad / accum.sqrt().clamp_min(eps)
        if max_update_rms > 0.0:
            rms = torch.linalg.vector_norm(update.detach().float()) / math.sqrt(update.numel())
            if torch.isfinite(rms) and rms > max_update_rms:
                update.mul_(max_update_rms / rms.clamp_min(eps))
        if wd:
            param.mul_(1.0 - lr * wd)
        param.add_(update.to(dtype=param.dtype), alpha=-lr)
        if prox_rho > 0.0:
            quant_mode = getattr(param, "_quant_mode", None)
            quant_group_size = getattr(param, "_quant_group_size", None)
            if quant_mode is not None and quant_group_size is not None:
                apply_chunked_prox(
                    param,
                    mode=str(quant_mode),
                    group_size=int(quant_group_size),
                    prox_rho=prox_rho,
                )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        group = self.param_groups[0]
        for param in group["params"]:
            self.step_param(param, group)
        return loss


def build_master_delta_baseline(
    train_quant: dict[str, QuantModule],
    *,
    sample_size: int,
) -> dict[str, dict[str, torch.Tensor]]:
    if sample_size <= 0:
        return {}
    baseline: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in train_quant.items():
        master = getattr(module, "master", None)
        if not isinstance(master, torch.nn.Parameter):
            continue
        flat = master.detach().reshape(-1)
        if flat.numel() == 0:
            continue
        count = min(sample_size, flat.numel())
        if count == 1:
            indices = torch.zeros(1, dtype=torch.long)
        else:
            # Build on CPU with explicit integer arithmetic. CUDA linspace with
            # dtype long can round the endpoint out of range for huge tensors.
            positions = torch.arange(count, dtype=torch.float64)
            indices = torch.floor(positions * float(flat.numel() - 1) / float(count - 1)).to(torch.long)
        baseline[name] = {
            "indices": indices,
            "values": flat.index_select(0, indices.to(device=flat.device)).detach().float().cpu(),
        }
    return baseline


@torch.no_grad()
def summarize_master_delta(
    train_quant: dict[str, QuantModule],
    baseline: dict[str, dict[str, torch.Tensor]],
) -> dict[str, float]:
    if not baseline:
        return {}
    total = 0
    abs_sum = 0.0
    sq_sum = 0.0
    max_abs = 0.0
    init_abs_sum = 0.0
    moved = 0
    for name, module in train_quant.items():
        item = baseline.get(name)
        master = getattr(module, "master", None)
        if item is None or not isinstance(master, torch.nn.Parameter):
            continue
        indices = item["indices"].to(device=master.device)
        initial = item["values"].to(device=master.device)
        current = master.detach().reshape(-1).index_select(0, indices).float()
        delta = current - initial
        abs_delta = delta.abs()
        total += int(delta.numel())
        abs_sum += float(abs_delta.sum().item())
        sq_sum += float(delta.square().sum().item())
        max_abs = max(max_abs, float(abs_delta.max().item()))
        init_abs_sum += float(initial.abs().sum().item())
        moved += int((abs_delta > 0).sum().item())
    if total <= 0:
        return {}
    return {
        "master_delta_sample_count": float(total),
        "master_delta_mean_abs": abs_sum / total,
        "master_delta_rms": math.sqrt(sq_sum / total),
        "master_delta_max_abs": max_abs,
        "master_delta_moved_frac": moved / total,
        "master_delta_rel_mean_abs": abs_sum / max(init_abs_sum, 1e-20),
    }


@torch.no_grad()
def summarize_master_delta_by_class(
    train_quant: dict[str, QuantModule],
    baseline: dict[str, dict[str, torch.Tensor]],
    *,
    diffusion_expected_rms: float,
) -> dict[str, dict[str, float]]:
    if not baseline:
        return {}
    stats: dict[str, dict[str, float]] = {}
    for name, module in train_quant.items():
        item = baseline.get(name)
        master = getattr(module, "master", None)
        if item is None or not isinstance(master, torch.nn.Parameter):
            continue
        cls = module_class(name)
        bucket = stats.setdefault(
            cls,
            {
                "sample_count": 0.0,
                "mean_abs_sum": 0.0,
                "sq_sum": 0.0,
                "max_abs": 0.0,
                "moved_count": 0.0,
                "init_abs_sum": 0.0,
            },
        )
        indices = item["indices"].to(device=master.device)
        initial = item["values"].to(device=master.device)
        current = master.detach().reshape(-1).index_select(0, indices).float()
        delta = current - initial
        abs_delta = delta.abs()
        bucket["sample_count"] += float(delta.numel())
        bucket["mean_abs_sum"] += float(abs_delta.sum().item())
        bucket["sq_sum"] += float(delta.square().sum().item())
        bucket["max_abs"] = max(bucket["max_abs"], float(abs_delta.max().item()))
        bucket["moved_count"] += float((abs_delta > 0).sum().item())
        bucket["init_abs_sum"] += float(initial.abs().sum().item())
    out: dict[str, dict[str, float]] = {}
    for cls, bucket in stats.items():
        total = bucket["sample_count"]
        if total <= 0:
            continue
        rms = math.sqrt(bucket["sq_sum"] / total)
        out[cls] = {
            "master_delta_sample_count": total,
            "master_delta_mean_abs": bucket["mean_abs_sum"] / total,
            "master_delta_rms": rms,
            "master_delta_max_abs": bucket["max_abs"],
            "master_delta_moved_frac": bucket["moved_count"] / total,
            "master_delta_rel_mean_abs": bucket["mean_abs_sum"] / max(bucket["init_abs_sum"], 1e-20),
            "master_delta_diffusion_ratio": (
                rms / diffusion_expected_rms if diffusion_expected_rms > 0.0 else float("nan")
            ),
        }
    return out


@torch.no_grad()
def _quantize_group_values_forensics(
    values: torch.Tensor,
    *,
    mode: str,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.numel() % group_size:
        values = F.pad(values.reshape(-1), (0, (-values.numel()) % group_size))
    if mode == "ternary":
        quant = quantize_ternary_mse(values.reshape(-1).float(), group_size=group_size)
        return quant.codes.reshape(-1, group_size), quant.scales.float()
    if mode == "binary":
        groups = values.reshape(-1, group_size).float()
        scales = groups.abs().mean(dim=1).clamp_min(1e-8)
        codes = torch.where(groups >= 0, torch.ones_like(groups), -torch.ones_like(groups)).to(torch.int8)
        return codes, scales
    raise ValueError(f"unsupported quantization mode: {mode}")


@torch.no_grad()
def build_code_forensics_baseline(
    train_quant: dict[str, QuantModule],
    *,
    groups_per_module: int,
) -> dict[str, dict[str, torch.Tensor]]:
    if groups_per_module <= 0:
        return {}
    baseline: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in train_quant.items():
        master = getattr(module, "master", None)
        mode = getattr(module, "mode", None)
        group_size = int(getattr(module, "group_size", 0) or 0)
        if not isinstance(master, torch.nn.Parameter) or mode is None or group_size <= 0:
            continue
        flat = master.detach().reshape(-1)
        if flat.numel() == 0:
            continue
        total_groups = max(1, (flat.numel() + group_size - 1) // group_size)
        count = min(groups_per_module, total_groups)
        if count == 1:
            group_indices = torch.zeros(1, dtype=torch.long)
        else:
            positions = torch.arange(count, dtype=torch.float64)
            group_indices = torch.floor(positions * float(total_groups - 1) / float(count - 1)).to(torch.long)
        pad = (-flat.numel()) % group_size
        padded = F.pad(flat.detach().float(), (0, pad)) if pad else flat.detach().float()
        groups = padded.view(total_groups, group_size).index_select(0, group_indices.to(device=master.device))
        codes, scales = _quantize_group_values_forensics(groups, mode=str(mode), group_size=group_size)
        baseline[name] = {
            "group_indices": group_indices,
            "codes": codes.cpu(),
            "scales": scales.cpu(),
            "mode": torch.tensor(0 if str(mode) == "ternary" else 1, dtype=torch.int8),
            "group_size": torch.tensor(group_size, dtype=torch.long),
        }
    return baseline


@torch.no_grad()
def summarize_code_forensics(
    train_quant: dict[str, QuantModule],
    baseline: dict[str, dict[str, torch.Tensor]],
    *,
    changed_chunks: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    if not baseline:
        return {}
    total_codes = 0
    changed = 0
    support_added = 0
    support_dropped = 0
    active_sign_flips = 0
    scale_ratio_sum = 0.0
    scale_ratio_sq_sum = 0.0
    scale_groups = 0
    for name, module in train_quant.items():
        item = baseline.get(name)
        master = getattr(module, "master", None)
        mode = getattr(module, "mode", None)
        group_size = int(getattr(module, "group_size", 0) or 0)
        if item is None or not isinstance(master, torch.nn.Parameter) or mode is None or group_size <= 0:
            continue
        flat = master.detach().reshape(-1)
        total_groups = max(1, (flat.numel() + group_size - 1) // group_size)
        pad = (-flat.numel()) % group_size
        padded = F.pad(flat.detach().float(), (0, pad)) if pad else flat.detach().float()
        group_indices = item["group_indices"].to(device=master.device)
        groups = padded.view(total_groups, group_size).index_select(0, group_indices)
        current_codes, current_scales = _quantize_group_values_forensics(groups, mode=str(mode), group_size=group_size)
        initial_codes = item["codes"].to(device=current_codes.device)
        initial_scales = item["scales"].to(device=current_scales.device)
        changed_mask = current_codes != initial_codes
        if changed_chunks is not None:
            changed_chunks.append(changed_mask.reshape(-1).detach().cpu())
        total_codes += int(current_codes.numel())
        changed += int(changed_mask.sum().item())
        initial_active = initial_codes != 0
        current_active = current_codes != 0
        support_added += int((current_active & ~initial_active).sum().item())
        support_dropped += int((initial_active & ~current_active).sum().item())
        active_sign_flips += int((initial_active & current_active & (current_codes != initial_codes)).sum().item())
        ratio = current_scales.float() / initial_scales.float().clamp_min(1e-8)
        scale_ratio_sum += float(ratio.sum().item())
        scale_ratio_sq_sum += float(ratio.square().sum().item())
        scale_groups += int(ratio.numel())
    if total_codes <= 0 or scale_groups <= 0:
        return {}
    scale_mean = scale_ratio_sum / scale_groups
    scale_var = max(scale_ratio_sq_sum / scale_groups - scale_mean * scale_mean, 0.0)
    return {
        "code_forensics_sample_count": float(total_codes),
        "code_changed_frac": changed / total_codes,
        "support_added_frac": support_added / total_codes,
        "support_dropped_frac": support_dropped / total_codes,
        "support_net_frac": (support_added - support_dropped) / total_codes,
        "support_asymmetry_frac": (
            (support_added - support_dropped) / max(support_added + support_dropped, 1)
        ),
        "active_sign_flip_frac": active_sign_flips / total_codes,
        "scale_ratio_mean": scale_mean,
        "scale_ratio_std": math.sqrt(scale_var),
    }


@torch.no_grad()
def summarize_code_forensics_by_class(
    train_quant: dict[str, QuantModule],
    baseline: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, float]]:
    if not baseline:
        return {}
    stats: dict[str, dict[str, float]] = {}
    for name, module in train_quant.items():
        item = baseline.get(name)
        master = getattr(module, "master", None)
        mode = getattr(module, "mode", None)
        group_size = int(getattr(module, "group_size", 0) or 0)
        if item is None or not isinstance(master, torch.nn.Parameter) or mode is None or group_size <= 0:
            continue
        flat = master.detach().reshape(-1)
        total_groups = max(1, (flat.numel() + group_size - 1) // group_size)
        pad = (-flat.numel()) % group_size
        padded = F.pad(flat.detach().float(), (0, pad)) if pad else flat.detach().float()
        group_indices = item["group_indices"].to(device=master.device)
        groups = padded.view(total_groups, group_size).index_select(0, group_indices)
        current_codes, current_scales = _quantize_group_values_forensics(groups, mode=str(mode), group_size=group_size)
        initial_codes = item["codes"].to(device=current_codes.device)
        initial_scales = item["scales"].to(device=current_scales.device)
        changed_mask = current_codes != initial_codes
        initial_active = initial_codes != 0
        current_active = current_codes != 0
        ratio = current_scales.float() / initial_scales.float().clamp_min(1e-8)

        cls = module_class(name)
        bucket = stats.setdefault(
            cls,
            {
                "code_count": 0.0,
                "changed_count": 0.0,
                "support_added_count": 0.0,
                "support_dropped_count": 0.0,
                "active_sign_flip_count": 0.0,
                "scale_ratio_sum": 0.0,
                "scale_ratio_sq_sum": 0.0,
                "scale_group_count": 0.0,
            },
        )
        bucket["code_count"] += float(current_codes.numel())
        bucket["changed_count"] += float(changed_mask.sum().item())
        bucket["support_added_count"] += float((current_active & ~initial_active).sum().item())
        bucket["support_dropped_count"] += float((initial_active & ~current_active).sum().item())
        bucket["active_sign_flip_count"] += float(
            (initial_active & current_active & (current_codes != initial_codes)).sum().item()
        )
        bucket["scale_ratio_sum"] += float(ratio.sum().item())
        bucket["scale_ratio_sq_sum"] += float(ratio.square().sum().item())
        bucket["scale_group_count"] += float(ratio.numel())
    out: dict[str, dict[str, float]] = {}
    for cls, bucket in stats.items():
        total_codes = bucket["code_count"]
        scale_groups = bucket["scale_group_count"]
        if total_codes <= 0 or scale_groups <= 0:
            continue
        scale_mean = bucket["scale_ratio_sum"] / scale_groups
        scale_var = max(bucket["scale_ratio_sq_sum"] / scale_groups - scale_mean * scale_mean, 0.0)
        out[cls] = {
            "code_forensics_sample_count": total_codes,
            "code_changed_frac": bucket["changed_count"] / total_codes,
            "support_added_frac": bucket["support_added_count"] / total_codes,
            "support_dropped_frac": bucket["support_dropped_count"] / total_codes,
            "support_net_frac": (
                (bucket["support_added_count"] - bucket["support_dropped_count"]) / total_codes
            ),
            "support_asymmetry_frac": (
                (bucket["support_added_count"] - bucket["support_dropped_count"])
                / max(bucket["support_added_count"] + bucket["support_dropped_count"], 1.0)
            ),
            "active_sign_flip_frac": bucket["active_sign_flip_count"] / total_codes,
            "scale_ratio_mean": scale_mean,
            "scale_ratio_std": math.sqrt(scale_var),
        }
    return out


def replace_all_2d_modules(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[dict[str, QuantModule], dict[str, QuantModule]]:
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)) and tuple(module.weight.shape)
    ]
    all_quant: dict[str, QuantModule] = {}
    train_quant: dict[str, QuantModule] = {}
    for name, module in modules:
        if should_skip_quant_module(name, args):
            continue
        train = should_train_module(name, args)
        if isinstance(module, torch.nn.Linear):
            replacement: QuantModule = (
                TrainQuantLinear(
                    module,
                    mode=args.mode,
                    group_size=args.group_size,
                    alpha=args.alpha_start,
                    master_dtype=args.master_dtype,
                )
                if train
                else FrozenQuantLinear(module, mode=args.mode, group_size=args.group_size)
            )
        elif isinstance(module, torch.nn.Embedding):
            replacement = (
                TrainQuantEmbedding(
                    module,
                    mode=args.mode,
                    group_size=args.group_size,
                    alpha=args.alpha_start,
                    master_dtype=args.master_dtype,
                )
                if train
                else FrozenQuantEmbedding(module, mode=args.mode, group_size=args.group_size)
            )
        else:
            continue
        parent, attr = get_parent_and_attr(model, name)
        setattr(parent, attr, replacement)
        all_quant[name] = replacement
        if train:
            train_quant[name] = replacement
    return all_quant, train_quant


def select_norm_parameters(model: torch.nn.Module, args: argparse.Namespace) -> dict[str, torch.nn.Parameter]:
    if not args.train_norms:
        return {}
    selected: dict[str, torch.nn.Parameter] = {}
    for name, param in model.named_parameters():
        if param.ndim != 1 or not name.endswith(".weight"):
            continue
        module_name = name[: -len(".weight")]
        lid = layer_id(module_name)
        train = False
        if args.train_regex and re.search(args.train_regex, module_name):
            train = True
        elif lid is not None and (args.train_all_layers or lid in args.train_layer_set):
            train = True
        if train:
            param.data = param.data.float()
            param.requires_grad_(True)
            selected[name] = param
    return selected


def set_alpha(train_quant: dict[str, QuantModule], alpha: float) -> None:
    for module in train_quant.values():
        if hasattr(module, "alpha"):
            module.alpha = alpha


def attach_quant_callbacks(train_quant: dict[str, QuantModule]) -> None:
    for module in train_quant.values():
        master = getattr(module, "master", None)
        if not isinstance(master, torch.nn.Parameter):
            continue

        def make_quantized(module: QuantModule = module) -> torch.Tensor:
            return alpha_quantized_weight(module.master, mode=module.mode, group_size=module.group_size, alpha=1.0).detach()

        master._make_quantized = make_quantized  # type: ignore[attr-defined]
        master._quant_mode = module.mode  # type: ignore[attr-defined]
        master._quant_group_size = module.group_size  # type: ignore[attr-defined]


def cache_rows(cache: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [row for row in cache["rows"] if int(row["tokens"]) <= args.max_train_tokens]
    if args.task_max_frac:
        total = len(rows)
        task_counts: dict[str, int] = {}
        selected_rows = []
        for row in rows:
            task = str(row.get("task") or "")
            cap = args.task_max_frac.get(task)
            if cap is not None:
                limit = max(1, int(total * cap))
                if task_counts.get(task, 0) >= limit:
                    continue
            task_counts[task] = task_counts.get(task, 0) + 1
            selected_rows.append(row)
        rows = selected_rows
    if args.shuffle_rows:
        random.Random(args.row_seed).shuffle(rows)
    if args.max_train_rows is not None:
        rows = rows[: args.max_train_rows]
    if not rows:
        raise RuntimeError("no cache rows selected")
    for row in rows:
        teacher_values = row.get("top_values")
        if not isinstance(teacher_values, torch.Tensor):
            teacher_values = row.get("teacher_top_logprobs")
        if not isinstance(teacher_values, torch.Tensor):
            raise RuntimeError("cache row must contain top_values or teacher_top_logprobs")
        row["keep"] = int(row.get("keep", teacher_values.shape[0]))
    return rows


def teacher_values_for_row(row: dict[str, Any]) -> torch.Tensor:
    value = row.get("top_values")
    if isinstance(value, torch.Tensor):
        return value
    value = row.get("teacher_top_logprobs")
    if isinstance(value, torch.Tensor):
        return value
    raise RuntimeError("cache row must contain top_values or teacher_top_logprobs")


def teacher_logprobs_for_row(row: dict[str, Any]) -> torch.Tensor | None:
    value = row.get("teacher_top_logprobs")
    return value if isinstance(value, torch.Tensor) else None


def resolve_cache_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.cache is not None:
        paths.append(args.cache)
    if args.cache_glob:
        globs = args.cache_glob if isinstance(args.cache_glob, list) else [args.cache_glob]
        expanded = [[Path(path) for path in sorted(glob.glob(pattern))] for pattern in globs]
        if args.interleave_cache_globs:
            max_len = max((len(items) for items in expanded), default=0)
            for index in range(max_len):
                for items in expanded:
                    if index < len(items):
                        paths.append(items[index])
        else:
            for items in expanded:
                paths.extend(items)
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise RuntimeError("training requires --cache or --cache-glob")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing cache path(s): " + ", ".join(missing[:8]))
    return paths


def load_training_cache(path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache = torch.load(path, map_location="cpu")
    if cache.get("format") not in {"qwen_agentworld_teacher_behavior_v1", "qwen_onpolicy_survival_cache_v1"}:
        raise ValueError(f"unsupported cache format in {path}: {cache.get('format')!r}")
    rows = cache_rows(cache, args)
    return cache, rows


def hidden_projection(
    *,
    layer_id: int,
    hidden_dim: int,
    projected_dim: int,
    seed: int,
    device: torch.device,
    cache: dict[tuple[int, int, int], torch.Tensor],
) -> torch.Tensor:
    key = (layer_id, hidden_dim, projected_dim)
    projection = cache.get(key)
    if projection is not None and projection.device == device:
        return projection
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + layer_id * 1009 + hidden_dim * 9176 + projected_dim)
    projection_cpu = torch.empty(hidden_dim, projected_dim, dtype=torch.float32)
    projection_cpu.bernoulli_(0.5, generator=generator)
    projection_cpu.mul_(2.0).sub_(1.0).div_(projected_dim**0.5)
    projection = projection_cpu.to(device=device)
    cache[key] = projection
    return projection


def projected_or_full_hidden_loss(
    captured: dict[int, torch.Tensor],
    row: dict[str, Any],
    layers: list[int],
    device: torch.device,
    *,
    projected_dim: int,
    projection_seed: int,
    projection_cache: dict[tuple[int, int, int], torch.Tensor],
) -> torch.Tensor:
    losses = []
    teacher_hidden = row.get("teacher_hidden", {})
    teacher_hidden_proj = row.get("teacher_hidden_proj", {})
    for layer_id in layers:
        if layer_id not in captured:
            continue
        student = captured[layer_id].float()
        if layer_id in teacher_hidden_proj:
            projection = hidden_projection(
                layer_id=layer_id,
                hidden_dim=student.shape[-1],
                projected_dim=projected_dim,
                seed=projection_seed,
                device=device,
                cache=projection_cache,
            )
            student_norm = F.layer_norm(student, student.shape[-1:])
            student_value = student_norm @ projection
            teacher_value = teacher_hidden_proj[layer_id].to(device=device, dtype=torch.float32)
            losses.append(
                (student_value - teacher_value).square().mean()
                / teacher_value.square().mean().clamp_min(1e-8)
            )
        elif layer_id in teacher_hidden:
            teacher_value = teacher_hidden[layer_id].to(device=device, dtype=torch.float32)
            losses.append(
                (student - teacher_value).square().mean()
                / teacher_value.square().mean().clamp_min(1e-8)
            )
    if not losses:
        return next(iter(captured.values())).sum() * 0.0 if captured else torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def save_patch(
    path: Path,
    modules: dict[str, QuantModule],
    *,
    norms: dict[str, torch.nn.Parameter] | None = None,
) -> None:
    tensors: dict[str, torch.Tensor] = {}
    for name, module in modules.items():
        tensors[canonical_weight_name(name)] = module.hard_weight().contiguous()
    for name, param in (norms or {}).items():
        tensors[name] = param.detach().cpu().to(torch.bfloat16).contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))


def save_patch_atomic(
    path: Path,
    modules: dict[str, QuantModule],
    *,
    norms: dict[str, torch.nn.Parameter] | None = None,
) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    # The all-2D latest patch is large. Remove the previous latest before
    # writing the replacement so checkpoint refresh does not require two full
    # copies worth of free disk.
    if path.exists():
        path.unlink()
    save_patch(tmp, modules, norms=norms)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument(
        "--cache-glob",
        action="append",
        help="repeatable glob for sharded teacher caches; loaded one shard at a time",
    )
    parser.add_argument("--cache-shard-steps", type=int, default=0, help="steps per cache shard before rotating to the next shard")
    parser.add_argument("--interleave-cache-globs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out-patch", type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--out-full-patch", type=Path)
    parser.add_argument("--skip-final-save", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-latest-every", type=int, default=0)
    parser.add_argument("--latest-patch", type=Path)
    parser.add_argument("--latest-report", type=Path)
    parser.add_argument("--init-weight-patch", action="append", type=Path)
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--mode", choices=("ternary", "binary"), default="ternary")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--master-dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--train-layers", default="")
    parser.add_argument("--train-embed", action="store_true")
    parser.add_argument("--train-head", action="store_true")
    parser.add_argument("--skip-quant-embed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-quant-head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-regex")
    parser.add_argument("--train-norms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-tokens", type=int, default=3500)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--shuffle-rows", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--row-seed", type=int, default=17)
    parser.add_argument("--task-max-frac", action="append", help="comma-separated task:fraction caps applied per loaded shard")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--optimizer",
        choices=("bonsai_qat8", "adamw", "mirror_sgd", "factored_adafactor"),
        default="bonsai_qat8",
    )
    parser.add_argument("--use-muon", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--muon-every", type=int, default=16)
    parser.add_argument("--muon-steps", type=int, default=2)
    parser.add_argument("--muon-max-elements", type=int, default=80_000_000)
    parser.add_argument("--muon-rms-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prox-rho", type=float, default=0.0)
    parser.add_argument("--max-update-rms", type=float, default=0.0)
    parser.add_argument("--mirror-eps", type=float, default=1e-6)
    parser.add_argument("--adafactor-beta2", type=float, default=0.999)
    parser.add_argument("--adafactor-eps", type=float, default=1e-6)
    parser.add_argument("--adafactor-update-chunk-rows", type=int, default=1024)
    parser.add_argument("--group-adagrad", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loss-adaptive-lr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loss-adaptive-beta", type=float, default=0.99)
    parser.add_argument("--loss-adaptive-min-scale", type=float, default=0.25)
    parser.add_argument("--loss-adaptive-max-scale", type=float, default=1.0)
    parser.add_argument("--task-family-normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--task-family-beta", type=float, default=0.98)
    parser.add_argument("--task-family-min-scale", type=float, default=0.25)
    parser.add_argument("--task-family-max-scale", type=float, default=2.0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--explicit-layer-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--post-accumulate-step", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--format-boost", type=float, default=5.0)
    parser.add_argument("--topk-other-bucket", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--chunked-head-kd", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--head-chunk-rows", type=int, default=8192)
    parser.add_argument("--hidden-weight", type=float, default=0.0)
    parser.add_argument("--hard-shadow-weight-start", type=float, default=0.0)
    parser.add_argument("--hard-shadow-weight-end", type=float, default=0.0)
    parser.add_argument("--hard-shadow-every", type=int, default=1)
    parser.add_argument("--anti-weight", type=float, default=0.1)
    parser.add_argument("--anti-margin", type=float, default=0.05)
    parser.add_argument("--code-weight", type=float, default=0.001)
    parser.add_argument("--code-modules-per-step", type=int, default=8)
    parser.add_argument("--code-groups-per-module", type=int, default=4096)
    parser.add_argument("--alpha-start", type=float, default=1.0)
    parser.add_argument("--alpha-end", type=float, default=1.0)
    parser.add_argument("--stop-kl-above", type=float, default=0.0)
    parser.add_argument("--stop-kl-max-alpha", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--master-delta-sample-size", type=int, default=0)
    parser.add_argument("--code-forensics-groups-per-module", type=int, default=0)
    parser.add_argument("--code-persistence-short-step", type=int, default=0)
    parser.add_argument("--code-persistence-long-step", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--log-first-cycle", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.master_dtype_name = args.master_dtype
    args.master_dtype = parse_torch_dtype(args.master_dtype)
    args.train_all_layers = args.train_layers.strip().lower() == "all"
    args.train_layer_set = parse_int_csv(args.train_layers)
    args.task_max_frac = parse_task_caps(args.task_max_frac)

    if args.skip_quant_embed and args.train_embed:
        raise RuntimeError("--skip-quant-embed cannot be combined with --train-embed")
    if args.skip_quant_head and args.train_head:
        raise RuntimeError("--skip-quant-head cannot be combined with --train-head")
    if args.chunked_head_kd:
        if not args.topk_other_bucket:
            raise RuntimeError("--chunked-head-kd requires --topk-other-bucket")
        if abs(args.temperature - 1.0) > 1e-6:
            raise RuntimeError("--chunked-head-kd currently requires --temperature 1.0")
        if args.anti_weight != 0.0:
            raise RuntimeError("--chunked-head-kd currently requires --anti-weight 0")
    if args.post_accumulate_step:
        if args.optimizer != "factored_adafactor":
            raise RuntimeError("--post-accumulate-step currently requires --optimizer factored_adafactor")
        if args.grad_clip != 0.0:
            raise RuntimeError("--post-accumulate-step cannot use global --grad-clip; set --grad-clip 0")
        if args.hard_shadow_weight_start > 0.0 or args.hard_shadow_weight_end > 0.0:
            raise RuntimeError("--post-accumulate-step cannot be combined with hard-shadow second backward")

    if args.materialize_only and args.out_full_patch is None:
        raise RuntimeError("--materialize-only requires --out-full-patch")
    if not args.materialize_only and (args.cache is None and not args.cache_glob):
        raise RuntimeError("training requires --cache/--cache-glob and --out-patch")
    if not args.materialize_only and not args.skip_final_save and args.out_patch is None:
        raise RuntimeError("training requires --out-patch unless --skip-final-save is set")
    if args.save_latest_every > 0:
        if args.latest_patch is None:
            args.latest_patch = args.out_patch.with_name(args.out_patch.stem + "_latest" + args.out_patch.suffix)
        if args.latest_report is None:
            args.latest_report = args.out_report.with_name(args.out_report.stem + "_latest" + args.out_report.suffix)

    model = load_model(args.model_dir)
    init_patch_reports = []
    for patch in args.init_weight_patch or []:
        init_patch_reports.append(apply_weight_patch(model, patch))
    model.eval()
    if args.gradient_checkpointing and not args.explicit_layer_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    if args.explicit_layer_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)

    all_quant, train_quant = replace_all_2d_modules(model, args)
    attach_quant_callbacks(train_quant)
    explicit_checkpointed_layers = enable_explicit_layer_checkpointing(model) if args.explicit_layer_checkpointing else 0
    gc.collect()
    torch.cuda.empty_cache()
    if args.materialize_only:
        save_patch(args.out_full_patch, all_quant)
        report = {
            "materialize_only": True,
            "model_dir": str(args.model_dir),
            "init_weight_patches": [str(path) for path in args.init_weight_patch or []],
            "init_patch_reports": init_patch_reports,
            "out_full_patch": str(args.out_full_patch),
            "mode": args.mode,
            "group_size": args.group_size,
            "all_quantized_modules": len(all_quant),
            "all_quantized_weights": sorted(canonical_weight_name(name) for name in all_quant),
        }
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"out_full_patch": str(args.out_full_patch), "all_quantized_modules": len(all_quant)}, indent=2))
        return

    cache_paths = resolve_cache_paths(args)
    shard_steps = args.cache_shard_steps if args.cache_shard_steps > 0 else args.steps
    active_shard = 0
    cache, rows = load_training_cache(cache_paths[active_shard], args)

    norm_params = select_norm_parameters(model, args)
    train_params: list[torch.nn.Parameter] = []
    for module in train_quant.values():
        train_params.extend(list(module.parameters()))
    train_params.extend(norm_params.values())
    if not train_params:
        raise RuntimeError("no trainable modules selected")
    matrix_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []
    for param in train_params:
        if param.ndim == 2:
            matrix_params.append(param)
        else:
            other_params.append(param)
    if args.optimizer == "bonsai_qat8":
        groups: list[dict[str, Any]] = []
        if matrix_params:
            groups.append(
                {
                    "params": matrix_params,
                    "role": "matrix_qat8",
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "block_size": 4096,
                    "prox_rho": args.prox_rho,
                    "threshold_boost": 0.0,
                    "max_update_rms": args.max_update_rms,
                    "use_muon": args.use_muon,
                    "muon_every": args.muon_every,
                    "muon_steps": args.muon_steps,
                    "muon_max_elements": args.muon_max_elements,
                    "muon_rms_match": args.muon_rms_match,
                }
            )
        if other_params:
            groups.append(
                {
                    "params": other_params,
                    "role": "adamw",
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                }
            )
        optimizer = BonsaiQAT8(groups)
    elif args.optimizer == "mirror_sgd":
        optimizer = GroupMirrorSGD(
            train_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            group_size=args.group_size,
            eps=args.mirror_eps,
            prox_rho=args.prox_rho,
            max_update_rms=args.max_update_rms,
            group_adagrad=args.group_adagrad,
        )
    elif args.optimizer == "factored_adafactor":
        optimizer = FactoredAdafactorQAT(
            train_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            beta2=args.adafactor_beta2,
            eps=args.adafactor_eps,
            prox_rho=args.prox_rho,
            max_update_rms=args.max_update_rms,
            update_chunk_rows=args.adafactor_update_chunk_rows,
        )
    else:
        optimizer = torch.optim.AdamW(train_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    post_step_handles: list[Any] = []
    if args.post_accumulate_step:
        if not isinstance(optimizer, FactoredAdafactorQAT):
            raise RuntimeError("post-accumulate stepping requires FactoredAdafactorQAT")
        group_by_param_id: dict[int, dict[str, Any]] = {}
        for group in optimizer.param_groups:
            for param in group["params"]:
                group_by_param_id[id(param)] = group

        def make_post_step_hook():
            def hook(param: torch.Tensor) -> None:
                group = group_by_param_id[id(param)]
                optimizer.step_param(param, group)  # type: ignore[arg-type]
                param.grad = None

            return hook

        for param in train_params:
            post_step_handles.append(param.register_post_accumulate_grad_hook(make_post_step_hook()))

    device = next(model.parameters()).device
    format_ids = torch.tensor(cache.get("format_token_ids", []), device=device, dtype=torch.long)
    collapse_ids = torch.tensor(cache.get("collapse_token_ids", []), device=device, dtype=torch.long)
    hidden_layers = [layer for layer in cache.get("hidden_layers", []) if args.hidden_weight > 0]
    hidden_projection_dim = int(cache.get("hidden_projection_dim") or 0)
    hidden_projection_seed = int(cache.get("hidden_projection_seed") or 0)
    hidden_projection_cache: dict[tuple[int, int, int], torch.Tensor] = {}
    master_delta_baseline = build_master_delta_baseline(
        train_quant,
        sample_size=args.master_delta_sample_size,
    )
    code_forensics_baseline = build_code_forensics_baseline(
        train_quant,
        groups_per_module=args.code_forensics_groups_per_module,
    )
    keep_ref = {"keep": 1}
    handles, captured = register_hidden_hooks(model, hidden_layers, keep_ref)
    history = []
    early_stop: dict[str, Any] | None = None
    loss_ref_ema: float | None = None
    task_loss_ema: dict[str, float] = {}
    short_changed_mask: torch.Tensor | None = None
    diffusion_expected_sq = 0.0
    try:
        for step in range(1, args.steps + 1):
            if step > 1 and len(cache_paths) > 1 and (step - 1) % shard_steps == 0:
                active_shard = (active_shard + 1) % len(cache_paths)
                cache, rows = load_training_cache(cache_paths[active_shard], args)
                format_ids = torch.tensor(cache.get("format_token_ids", []), device=device, dtype=torch.long)
                collapse_ids = torch.tensor(cache.get("collapse_token_ids", []), device=device, dtype=torch.long)
                hidden_layers = [layer for layer in cache.get("hidden_layers", []) if args.hidden_weight > 0]
                hidden_projection_dim = int(cache.get("hidden_projection_dim") or 0)
                hidden_projection_seed = int(cache.get("hidden_projection_seed") or 0)
                print(
                    json.dumps(
                        {
                            "cache_shard_loaded": {
                                "step": step,
                                "shard_index": active_shard,
                                "cache": str(cache_paths[active_shard]),
                                "rows": len(rows),
                            }
                        }
                    ),
                    flush=True,
                )
                gc.collect()
            if args.steps <= 1:
                alpha = args.alpha_end
            else:
                frac = (step - 1) / (args.steps - 1)
                alpha = args.alpha_start + frac * (args.alpha_end - args.alpha_start)
            set_alpha(train_quant, alpha)
            row = rows[(step - 1) % len(rows)]
            teacher_values = teacher_values_for_row(row)
            teacher_logprobs = teacher_logprobs_for_row(row)
            keep_ref["keep"] = int(row["keep"])
            captured.clear()
            if args.chunked_head_kd:
                outputs = model.model(**row_to_inputs(row, device), use_cache=False)
                hidden_for_head = outputs[0][:, -int(row["keep"]) :, :]
                kl, stats = chunked_head_topk_other_kl(
                    hidden_for_head,
                    model.lm_head,
                    teacher_values,
                    row["top_indices"],
                    teacher_logsumexp=row.get("teacher_logsumexp"),
                    teacher_logprobs=teacher_logprobs,
                    format_ids=format_ids,
                    format_boost=args.format_boost,
                    chunk_rows=args.head_chunk_rows,
                )
                anti = kl.detach().new_zeros(())
            else:
                if row.get("top_values") is None:
                    raise RuntimeError("non-chunked KD requires raw teacher top_values; vLLM logprob caches require --chunked-head-kd")
                outputs = model(**row_to_inputs(row, device), use_cache=False, logits_to_keep=int(row["keep"]))
                kl, stats = topk_teacher_kl(
                    outputs.logits,
                    teacher_values,
                    row["top_indices"],
                    temperature=args.temperature,
                    format_ids=format_ids,
                    format_boost=args.format_boost,
                    teacher_logsumexp=row.get("teacher_logsumexp"),
                    use_other_bucket=args.topk_other_bucket,
                )
                anti = anti_collapse_loss(outputs.logits, teacher_values, row["top_indices"], collapse_ids, margin=args.anti_margin)
            h_loss = (
                projected_or_full_hidden_loss(
                    captured,
                    row,
                    hidden_layers,
                    device,
                    projected_dim=hidden_projection_dim,
                    projection_seed=hidden_projection_seed,
                    projection_cache=hidden_projection_cache,
                )
                if args.hidden_weight > 0
                else kl.detach().new_zeros(())
            )
            raw_kl_value = float(stats["kl"])
            task_name = str(row.get("task") or "?")
            family_scale = 1.0
            if args.task_family_normalize:
                family_ema = task_loss_ema.get(task_name)
                if family_ema is not None and loss_ref_ema is not None:
                    family_scale = loss_ref_ema / max(family_ema, 1e-8)
                    family_scale = max(args.task_family_min_scale, min(args.task_family_max_scale, family_scale))
            if args.code_weight > 0.0:
                code_modules = [module for module in train_quant.values() if hasattr(module, "codebook_loss")]
                if args.code_modules_per_step > 0 and len(code_modules) > args.code_modules_per_step:
                    start = ((step - 1) * args.code_modules_per_step) % len(code_modules)
                    selected_modules = [
                        code_modules[(start + offset) % len(code_modules)]
                        for offset in range(args.code_modules_per_step)
                    ]
                else:
                    selected_modules = code_modules
                code_terms = [
                    module.codebook_loss(
                        groups_per_sample=args.code_groups_per_module,
                        sample_index=step + module_index * 9973,
                    )
                    for module_index, module in enumerate(selected_modules)
                ]
                code = torch.stack(code_terms).mean() if code_terms else kl.detach().new_zeros(())
            else:
                code = kl.detach().new_zeros(())
            loss = family_scale * kl + args.hidden_weight * h_loss + args.anti_weight * anti + args.code_weight * code
            lr_scale = 1.0
            if args.loss_adaptive_lr and loss_ref_ema is not None:
                lr_scale = ((loss_ref_ema + 1e-8) / (raw_kl_value + 1e-8)) ** 0.5
                lr_scale = max(args.loss_adaptive_min_scale, min(args.loss_adaptive_max_scale, lr_scale))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lr_scale
            cap_step_rms = (
                float(args.lr * lr_scale * args.max_update_rms)
                if args.max_update_rms > 0.0
                else 0.0
            )
            diffusion_expected_sq += cap_step_rms * cap_step_rms
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            hard_shadow_weight = 0.0
            hard_stats: dict[str, float] = {}
            if (
                args.hard_shadow_weight_end > 0.0
                and alpha < 1.0
                and args.hard_shadow_every > 0
                and step % args.hard_shadow_every == 0
            ):
                if args.alpha_end == args.alpha_start:
                    alpha_frac = 1.0
                else:
                    alpha_frac = (alpha - args.alpha_start) / (args.alpha_end - args.alpha_start)
                alpha_frac = max(0.0, min(1.0, alpha_frac))
                hard_shadow_weight = args.hard_shadow_weight_start + alpha_frac * (
                    args.hard_shadow_weight_end - args.hard_shadow_weight_start
                )
                if hard_shadow_weight > 0.0:
                    del outputs
                    outputs = None
                    captured.clear()
                    set_alpha(train_quant, 1.0)
                    if args.chunked_head_kd:
                        hard_outputs = model.model(**row_to_inputs(row, device), use_cache=False)
                        hard_hidden_for_head = hard_outputs[0][:, -int(row["keep"]) :, :]
                        hard_kl, hard_stats = chunked_head_topk_other_kl(
                            hard_hidden_for_head,
                            model.lm_head,
                            teacher_values,
                            row["top_indices"],
                            teacher_logsumexp=row.get("teacher_logsumexp"),
                            teacher_logprobs=teacher_logprobs,
                            format_ids=format_ids,
                            format_boost=args.format_boost,
                            chunk_rows=args.head_chunk_rows,
                        )
                        hard_anti = hard_kl.detach().new_zeros(())
                    else:
                        if row.get("top_values") is None:
                            raise RuntimeError("non-chunked hard-shadow KD requires raw teacher top_values")
                        hard_outputs = model(**row_to_inputs(row, device), use_cache=False, logits_to_keep=int(row["keep"]))
                        hard_kl, hard_stats = topk_teacher_kl(
                            hard_outputs.logits,
                            teacher_values,
                            row["top_indices"],
                            temperature=args.temperature,
                            format_ids=format_ids,
                            format_boost=args.format_boost,
                            teacher_logsumexp=row.get("teacher_logsumexp"),
                            use_other_bucket=args.topk_other_bucket,
                        )
                        hard_anti = anti_collapse_loss(
                            hard_outputs.logits,
                            teacher_values,
                            row["top_indices"],
                            collapse_ids,
                            margin=args.anti_margin,
                        )
                    hard_h_loss = (
                        projected_or_full_hidden_loss(
                            captured,
                            row,
                            hidden_layers,
                            device,
                            projected_dim=hidden_projection_dim,
                            projection_seed=hidden_projection_seed,
                            projection_cache=hidden_projection_cache,
                        )
                        if args.hidden_weight > 0
                        else hard_kl.detach().new_zeros(())
                    )
                    hard_loss = family_scale * hard_kl + args.hidden_weight * hard_h_loss + args.anti_weight * hard_anti
                    (hard_shadow_weight * hard_loss).backward()
                    hard_stats = {
                        f"hard_{key}": float(value)
                        for key, value in hard_stats.items()
                    } | {
                        "hard_shadow_weight": float(hard_shadow_weight),
                        "hard_hidden": float(hard_h_loss.detach().cpu()),
                        "hard_anti": float(hard_anti.detach().cpu()),
                        "hard_loss": float(hard_loss.detach().cpu()),
                    }
                    del hard_outputs, hard_loss, hard_kl, hard_anti, hard_h_loss
                    set_alpha(train_quant, alpha)
            if args.post_accumulate_step:
                grad_global_norm = 0.0
                grad_clip_scale = 1.0
            else:
                if args.grad_clip > 0.0:
                    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(train_params, args.grad_clip)
                    grad_global_norm = float(grad_norm_tensor.detach().cpu())
                    grad_clip_scale = min(1.0, args.grad_clip / max(grad_global_norm, 1e-12))
                else:
                    grad_norm_sq = 0.0
                    for param in train_params:
                        if param.grad is not None:
                            grad_norm_sq += float(param.grad.detach().float().square().sum().item())
                    grad_global_norm = math.sqrt(grad_norm_sq)
                    grad_clip_scale = 1.0
                optimizer.step()
            should_log = (
                step == 1
                or step % args.log_every == 0
                or step == args.steps
                or (args.log_first_cycle and step <= len(rows))
            )
            if should_log:
                master_delta_stats = summarize_master_delta(train_quant, master_delta_baseline)
                diffusion_expected_rms = math.sqrt(diffusion_expected_sq) if diffusion_expected_sq > 0.0 else 0.0
                master_delta_rms = float(master_delta_stats.get("master_delta_rms", 0.0) or 0.0)
                master_delta_diffusion_ratio = (
                    master_delta_rms / diffusion_expected_rms
                    if diffusion_expected_rms > 0.0
                    else None
                )
                master_delta_by_class = summarize_master_delta_by_class(
                    train_quant,
                    master_delta_baseline,
                    diffusion_expected_rms=diffusion_expected_rms,
                )
                changed_chunks: list[torch.Tensor] | None = (
                    []
                    if args.code_persistence_short_step > 0
                    and args.code_persistence_long_step > args.code_persistence_short_step
                    and code_forensics_baseline
                    else None
                )
                code_forensics_stats = summarize_code_forensics(
                    train_quant,
                    code_forensics_baseline,
                    changed_chunks=changed_chunks,
                )
                code_forensics_by_class = summarize_code_forensics_by_class(
                    train_quant,
                    code_forensics_baseline,
                )
                module_class_forensics = {}
                for cls in sorted(set(master_delta_by_class) | set(code_forensics_by_class)):
                    module_class_forensics[cls] = {
                        **master_delta_by_class.get(cls, {}),
                        **code_forensics_by_class.get(cls, {}),
                    }
                if changed_chunks is not None:
                    current_changed_mask = torch.cat(changed_chunks) if changed_chunks else torch.empty(0, dtype=torch.bool)
                    if step == args.code_persistence_short_step:
                        short_changed_mask = current_changed_mask.clone()
                        code_forensics_stats["code_persistence_short_changed_count"] = float(
                            short_changed_mask.sum().item()
                        )
                    elif (
                        short_changed_mask is not None
                        and step >= args.code_persistence_long_step
                        and current_changed_mask.numel() == short_changed_mask.numel()
                    ):
                        denom = int(short_changed_mask.sum().item())
                        still = int((short_changed_mask & current_changed_mask).sum().item())
                        code_forensics_stats["code_persistence_short_changed_count"] = float(denom)
                        code_forensics_stats["code_persistence_still_changed_count"] = float(still)
                        code_forensics_stats["code_persistence_frac"] = still / max(denom, 1)
                record = {
                    "step": step,
                    "alpha": alpha,
                    "task": row.get("task"),
                    "id": row.get("id"),
                    "tokens": int(row["tokens"]),
                    "loss": float(loss.detach().cpu()),
                    "hidden": float(h_loss.detach().cpu()),
                    "anti": float(anti.detach().cpu()),
                    "code": float(code.detach().cpu()),
                    "lr_scale": float(lr_scale),
                    "family_scale": float(family_scale),
                    "effective_lr": float(args.lr * lr_scale),
                    "cap_step_rms": cap_step_rms,
                    "master_delta_diffusion_expected_rms": diffusion_expected_rms,
                    "master_delta_diffusion_ratio": master_delta_diffusion_ratio,
                    "grad_global_norm": grad_global_norm,
                    "grad_clip_scale": grad_clip_scale,
                    **stats,
                    **hard_stats,
                    **master_delta_stats,
                    **code_forensics_stats,
                    "module_class_forensics": module_class_forensics,
                }
                history.append(record)
                print(json.dumps(record), flush=True)
                if (
                    args.stop_kl_above > 0.0
                    and alpha <= args.stop_kl_max_alpha
                    and record["kl"] > args.stop_kl_above
                ):
                    early_stop = {
                        "reason": "kl_above_threshold",
                        "step": step,
                        "alpha": alpha,
                        "kl": record["kl"],
                        "threshold": args.stop_kl_above,
                        "max_alpha": args.stop_kl_max_alpha,
                    }
                    print(json.dumps({"early_stop": early_stop}), flush=True)
                    break
            if loss_ref_ema is None:
                loss_ref_ema = raw_kl_value
            else:
                loss_ref_ema = args.loss_adaptive_beta * loss_ref_ema + (1.0 - args.loss_adaptive_beta) * raw_kl_value
            family_prev = task_loss_ema.get(task_name)
            if family_prev is None:
                task_loss_ema[task_name] = raw_kl_value
            else:
                task_loss_ema[task_name] = args.task_family_beta * family_prev + (1.0 - args.task_family_beta) * raw_kl_value
            del outputs, loss, kl, anti, h_loss, code
            captured.clear()
            torch.cuda.empty_cache()
            if args.save_latest_every > 0 and (step % args.save_latest_every == 0):
                save_patch_atomic(args.latest_patch, train_quant, norms=norm_params)
                latest_payload = {
                    "partial": True,
                    "step": step,
                    "alpha": alpha,
                    "out_patch": str(args.out_patch),
                    "latest_patch": str(args.latest_patch),
                    "history": history,
                }
                args.latest_report.parent.mkdir(parents=True, exist_ok=True)
                args.latest_report.write_text(json.dumps(latest_payload, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "checkpoint_saved": {
                                "step": step,
                                "latest_patch": str(args.latest_patch),
                                "latest_report": str(args.latest_report),
                            }
                        }
                    ),
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()
        for handle in post_step_handles:
            handle.remove()

    if args.skip_final_save:
        print(json.dumps({"skip_final_save": True}), flush=True)
    else:
        save_patch(args.out_patch, train_quant, norms=norm_params)
        if args.out_full_patch is not None:
            save_patch(args.out_full_patch, all_quant, norms=norm_params)
    report = {
        "cache": str(args.cache),
        "cache_glob": args.cache_glob,
        "cache_paths": [str(path) for path in cache_paths],
        "cache_shard_steps": args.cache_shard_steps,
        "interleave_cache_globs": args.interleave_cache_globs,
        "task_max_frac": args.task_max_frac,
        "model_dir": str(args.model_dir),
        "init_weight_patches": [str(path) for path in args.init_weight_patch or []],
        "init_patch_reports": init_patch_reports,
        "out_patch": str(args.out_patch),
        "out_full_patch": str(args.out_full_patch) if args.out_full_patch else None,
        "skip_final_save": args.skip_final_save,
        "save_latest_every": args.save_latest_every,
        "latest_patch": str(args.latest_patch) if args.latest_patch else None,
        "latest_report": str(args.latest_report) if args.latest_report else None,
        "mode": args.mode,
        "group_size": args.group_size,
        "master_dtype": args.master_dtype_name,
        "train_layers": sorted(args.train_layer_set),
        "train_all_layers": args.train_all_layers,
        "train_embed": args.train_embed,
        "train_head": args.train_head,
        "skip_quant_embed": args.skip_quant_embed,
        "skip_quant_head": args.skip_quant_head,
        "train_regex": args.train_regex,
        "train_norms": args.train_norms,
        "optimizer": args.optimizer,
        "adafactor_update_chunk_rows": args.adafactor_update_chunk_rows,
        "group_adagrad": args.group_adagrad,
        "loss_adaptive_lr": args.loss_adaptive_lr,
        "loss_adaptive_beta": args.loss_adaptive_beta,
        "loss_adaptive_min_scale": args.loss_adaptive_min_scale,
        "loss_adaptive_max_scale": args.loss_adaptive_max_scale,
        "task_family_normalize": args.task_family_normalize,
        "task_family_beta": args.task_family_beta,
        "task_family_min_scale": args.task_family_min_scale,
        "task_family_max_scale": args.task_family_max_scale,
        "gradient_checkpointing": args.gradient_checkpointing,
        "explicit_layer_checkpointing": args.explicit_layer_checkpointing,
        "explicit_checkpointed_layers": explicit_checkpointed_layers,
        "post_accumulate_step": args.post_accumulate_step,
        "steps": args.steps,
        "lr": args.lr,
        "max_train_tokens": args.max_train_tokens,
        "max_train_rows": args.max_train_rows,
        "shuffle_rows": args.shuffle_rows,
        "row_seed": args.row_seed,
        "temperature": args.temperature,
        "format_boost": args.format_boost,
        "topk_other_bucket": args.topk_other_bucket,
        "chunked_head_kd": args.chunked_head_kd,
        "head_chunk_rows": args.head_chunk_rows,
        "hidden_weight": args.hidden_weight,
        "hard_shadow_weight_start": args.hard_shadow_weight_start,
        "hard_shadow_weight_end": args.hard_shadow_weight_end,
        "hard_shadow_every": args.hard_shadow_every,
        "anti_weight": args.anti_weight,
        "code_weight": args.code_weight,
        "use_muon": args.use_muon,
        "muon_every": args.muon_every,
        "muon_steps": args.muon_steps,
        "muon_max_elements": args.muon_max_elements,
        "muon_rms_match": args.muon_rms_match,
        "prox_rho": args.prox_rho,
        "max_update_rms": args.max_update_rms,
        "master_delta_sample_size": args.master_delta_sample_size,
        "code_forensics_groups_per_module": args.code_forensics_groups_per_module,
        "code_persistence_short_step": args.code_persistence_short_step,
        "code_persistence_long_step": args.code_persistence_long_step,
        "matrix_params": len(matrix_params),
        "other_params": len(other_params),
        "trainable_modules": sorted(train_quant),
        "all_quantized_modules": len(all_quant),
        "norm_params": sorted(norm_params),
        "rows": [
            {"index": row.get("index"), "task": row.get("task"), "id": row.get("id"), "tokens": int(row["tokens"]), "keep": int(row["keep"])}
            for row in rows
        ],
        "history": history,
        "early_stop": early_stop,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out_patch": str(args.out_patch), "out_full_patch": report["out_full_patch"], "trainable_modules": len(train_quant)}, indent=2))


if __name__ == "__main__":
    main()
