"""Lift-Project QAT for one-GPU Qwen compression.

The low-bit model stays frozen. We attach a global low-rank continuous tangent,
train it on on-policy survival loss, then optionally project the merged weights
back into binary/ternary group codebooks. The shipped artifact has no LoRA.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clark_quantize.bonsai_qat8 import BonsaiQAT8
from train_all2d_ternary_qat import (
    alpha_quantized_weight,
    canonical_weight_name,
    dequantize_binary_chunked,
    dequantize_ternary_chunked,
    get_parent_and_attr,
)
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


def layer_id(module_name: str) -> int | None:
    match = LAYER_RE.search(module_name)
    return int(match.group(1)) if match else None


def parse_int_csv(value: str | None) -> set[int]:
    if not value:
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def parameter_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def cache_rows(cache: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [row for row in cache["rows"] if int(row["tokens"]) <= args.max_train_tokens]
    if args.failures:
        wanted = {part.strip() for part in args.failures.split(",") if part.strip()}
        rows = [
            row
            for row in rows
            if row.get("failure_kind") in wanted or row.get("collapse", {}).get("failure_kind") in wanted
        ]
    if args.max_train_rows is not None:
        rows = rows[: args.max_train_rows]
    if not rows:
        raise RuntimeError("no cache rows selected")
    for row in rows:
        row["keep"] = int(row.get("keep", row["top_values"].shape[0]))
    return rows


def module_rank(name: str, module: torch.nn.Module, args: argparse.Namespace) -> int:
    if isinstance(module, torch.nn.Embedding):
        return args.rank_embed
    if name == "lm_head" or name.endswith(".lm_head"):
        return args.rank_head
    if ".mlp." in name:
        return args.rank_mlp
    return args.rank_attention


def should_attach_lora(name: str, module: torch.nn.Module, args: argparse.Namespace) -> bool:
    rank = module_rank(name, module, args)
    if rank <= 0:
        return False
    if args.train_regex:
        if re.search(args.train_regex, name):
            return True
        if not args.train_layer_set:
            if isinstance(module, torch.nn.Embedding):
                return args.train_embed
            if name == "lm_head" or name.endswith(".lm_head"):
                return args.train_head
            return False
    if isinstance(module, torch.nn.Embedding):
        return args.train_embed
    if name == "lm_head" or name.endswith(".lm_head"):
        return args.train_head
    lid = layer_id(name)
    return not args.train_layer_set or (lid is not None and lid in args.train_layer_set)


@torch.no_grad()
def group_rms_log(weight: torch.Tensor, *, group_size: int) -> torch.Tensor:
    source = weight.detach().float()
    rows, cols = source.shape
    pad = (-cols) % group_size
    if pad:
        source = F.pad(source, (0, pad))
    groups = source.view(rows, -1, group_size)
    rms = groups.square().mean(dim=-1).sqrt().clamp_min(1e-8)
    return rms.log()


def apply_group_magnitude(
    weight: torch.Tensor,
    log_mag: torch.Tensor,
    *,
    group_size: int,
    low_memory: bool,
) -> torch.Tensor:
    rows, cols = weight.shape
    pad = (-cols) % group_size
    if low_memory:
        source = F.pad(weight, (0, pad)) if pad else weight
        groups = source.view(rows, -1, group_size)
        rms = groups.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        normalized = groups / rms
        mag = log_mag.to(dtype=weight.dtype).exp().unsqueeze(-1)
        scaled = normalized * mag
        return scaled.reshape(rows, -1)[:, :cols]
    source = F.pad(weight.float(), (0, pad)) if pad else weight.float()
    groups = source.view(rows, -1, group_size)
    rms = groups.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    normalized = groups / rms
    scaled = normalized * log_mag.float().exp().unsqueeze(-1)
    out = scaled.reshape(rows, -1)[:, :cols]
    return out


def module_projection_mode(name: str, args: argparse.Namespace) -> str:
    if ".mlp." in name and args.project_mlp_mode is not None:
        return args.project_mlp_mode
    if ".mlp." not in name and args.project_attention_mode is not None:
        return args.project_attention_mode
    return args.project_mode


def module_group_size(name: str, args: argparse.Namespace) -> int:
    if ".mlp." in name and args.project_mlp_group_size is not None:
        return int(args.project_mlp_group_size)
    if ".mlp." not in name and args.project_attention_group_size is not None:
        return int(args.project_attention_group_size)
    return int(args.group_size)


class LiftProjectLinear(torch.nn.Module):
    def __init__(
        self,
        source: torch.nn.Linear,
        *,
        rank: int,
        alpha: float,
        mode: str,
        group_size: int,
        projection_aware: bool,
        use_dora: bool,
        lora_dtype: torch.dtype,
        low_memory_projection: bool,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.mode = mode
        self.group_size = group_size
        self.projection_aware = projection_aware
        self.projection_alpha = 1.0
        self.use_dora = use_dora
        self.low_memory_projection = low_memory_projection
        self.register_buffer("base_weight", source.weight.detach())
        if source.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", source.bias.detach())
        out_features, in_features = source.weight.shape
        self.lora_a = torch.nn.Parameter(torch.empty(rank, in_features, dtype=lora_dtype))
        self.lora_b = torch.nn.Parameter(torch.zeros(out_features, rank, dtype=lora_dtype))
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        init_log_mag = group_rms_log(source.weight.detach(), group_size=group_size)
        self.register_buffer("init_log_mag", init_log_mag)
        self.log_mag = torch.nn.Parameter(init_log_mag.to(dtype=lora_dtype).clone())

    @property
    def scaling(self) -> float:
        return self.alpha / max(self.rank, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        if self.projection_aware:
            weight = alpha_quantized_weight(
                self.effective_weight(),
                mode=self.mode,
                group_size=self.group_size,
                alpha=self.projection_alpha,
            )
            return F.linear(x, weight.to(dtype=x.dtype), bias)
        out = F.linear(x, self.base_weight.to(dtype=x.dtype), bias)
        lora_input = x.to(dtype=self.lora_a.dtype)
        lifted = F.linear(F.linear(lora_input, self.lora_a), self.lora_b) * self.scaling
        return out + lifted.to(dtype=out.dtype)

    def lora_penalty(self) -> torch.Tensor:
        penalty = self.lora_a.float().square().mean() + self.lora_b.float().square().mean()
        if self.use_dora:
            penalty = penalty + 0.01 * (self.log_mag.float() - self.init_log_mag.float()).square().mean()
        return penalty

    def effective_weight(self) -> torch.Tensor:
        if self.low_memory_projection:
            delta = self.lora_b @ self.lora_a
            lifted = self.base_weight + (self.scaling * delta).to(dtype=self.base_weight.dtype)
        else:
            delta = self.lora_b.float() @ self.lora_a.float()
            lifted = self.base_weight.float() + self.scaling * delta
        if not self.use_dora:
            return lifted
        return apply_group_magnitude(
            lifted,
            self.log_mag,
            group_size=self.group_size,
            low_memory=self.low_memory_projection,
        )


class LiftProjectEmbedding(torch.nn.Module):
    def __init__(
        self,
        source: torch.nn.Embedding,
        *,
        rank: int,
        alpha: float,
        mode: str,
        group_size: int,
        projection_aware: bool,
        use_dora: bool,
        lora_dtype: torch.dtype,
        low_memory_projection: bool,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.mode = mode
        self.group_size = group_size
        self.projection_aware = projection_aware
        self.projection_alpha = 1.0
        self.use_dora = use_dora
        self.low_memory_projection = low_memory_projection
        self.register_buffer("base_weight", source.weight.detach())
        self.padding_idx = source.padding_idx
        self.max_norm = source.max_norm
        self.norm_type = source.norm_type
        self.scale_grad_by_freq = source.scale_grad_by_freq
        self.sparse = source.sparse
        vocab_size, hidden = source.weight.shape
        self.lora_a = torch.nn.Parameter(torch.empty(rank, hidden, dtype=lora_dtype))
        self.lora_b = torch.nn.Parameter(torch.zeros(vocab_size, rank, dtype=lora_dtype))
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        init_log_mag = group_rms_log(source.weight.detach(), group_size=group_size)
        self.register_buffer("init_log_mag", init_log_mag)
        self.log_mag = torch.nn.Parameter(init_log_mag.to(dtype=lora_dtype).clone())

    @property
    def scaling(self) -> float:
        return self.alpha / max(self.rank, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.projection_aware:
            weight = alpha_quantized_weight(
                self.effective_weight(),
                mode=self.mode,
                group_size=self.group_size,
                alpha=self.projection_alpha,
            )
            return F.embedding(
                input_ids,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
        base = F.embedding(
            input_ids,
            self.base_weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        low = F.embedding(input_ids, self.lora_b, self.padding_idx, None, self.norm_type, False, False)
        delta = torch.matmul(low, self.lora_a) * self.scaling
        return base + delta.to(dtype=base.dtype)

    def lora_penalty(self) -> torch.Tensor:
        penalty = self.lora_a.float().square().mean() + self.lora_b.float().square().mean()
        if self.use_dora:
            penalty = penalty + 0.01 * (self.log_mag.float() - self.init_log_mag.float()).square().mean()
        return penalty

    def effective_weight(self) -> torch.Tensor:
        if self.low_memory_projection:
            delta = self.lora_b @ self.lora_a
            lifted = self.base_weight + (self.scaling * delta).to(dtype=self.base_weight.dtype)
        else:
            delta = self.lora_b.float() @ self.lora_a.float()
            lifted = self.base_weight.float() + self.scaling * delta
        if not self.use_dora:
            return lifted
        return apply_group_magnitude(
            lifted,
            self.log_mag,
            group_size=self.group_size,
            low_memory=self.low_memory_projection,
        )


LiftModule = LiftProjectLinear | LiftProjectEmbedding


def attach_lora(model: torch.nn.Module, args: argparse.Namespace) -> dict[str, LiftModule]:
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding))
    ]
    out: dict[str, LiftModule] = {}
    for name, module in modules:
        if not should_attach_lora(name, module, args):
            continue
        rank = module_rank(name, module, args)
        mode = module_projection_mode(name, args)
        group_size = module_group_size(name, args)
        replacement: LiftModule
        if isinstance(module, torch.nn.Linear):
            replacement = LiftProjectLinear(
                module,
                rank=rank,
                alpha=args.lora_alpha,
                mode=mode,
                group_size=group_size,
                projection_aware=args.projection_aware,
                use_dora=args.use_dora,
                lora_dtype=parameter_dtype(args.lora_dtype),
                low_memory_projection=args.low_memory_projection,
            )
        else:
            replacement = LiftProjectEmbedding(
                module,
                rank=rank,
                alpha=args.lora_alpha,
                mode=mode,
                group_size=group_size,
                projection_aware=args.projection_aware,
                use_dora=args.use_dora,
                lora_dtype=parameter_dtype(args.lora_dtype),
                low_memory_projection=args.low_memory_projection,
            )
        replacement.to(next(module.parameters()).device)
        parent, attr = get_parent_and_attr(model, name)
        setattr(parent, attr, replacement)
        out[name] = replacement
    return out


@torch.no_grad()
def projected_weight(weight: torch.Tensor, *, mode: str, group_size: int) -> torch.Tensor:
    if mode == "ternary":
        return dequantize_ternary_chunked(weight, group_size=group_size).cpu().to(torch.bfloat16)
    if mode == "binary":
        return dequantize_binary_chunked(weight, group_size=group_size).cpu().to(torch.bfloat16)
    raise ValueError(f"unsupported projection mode: {mode}")


@torch.no_grad()
def save_projected_patch(path: Path, modules: dict[str, LiftModule], args: argparse.Namespace) -> dict[str, str]:
    tensors: dict[str, torch.Tensor] = {}
    modes: dict[str, str] = {}
    for name, module in modules.items():
        mode = module_projection_mode(name, args)
        tensors[canonical_weight_name(name)] = projected_weight(
            module.effective_weight(),
            mode=mode,
            group_size=module_group_size(name, args),
        ).contiguous()
        modes[name] = mode
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    return modes


def save_lora_state(path: Path, modules: dict[str, LiftModule]) -> None:
    tensors: dict[str, torch.Tensor] = {}
    for name, module in modules.items():
        tensors[f"{name}.lora_a"] = module.lora_a.detach().cpu().contiguous()
        tensors[f"{name}.lora_b"] = module.lora_b.detach().cpu().contiguous()
        if hasattr(module, "log_mag"):
            tensors[f"{name}.log_mag"] = module.log_mag.detach().cpu().contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))


@torch.no_grad()
def load_lora_state(path: Path, modules: dict[str, LiftModule]) -> dict[str, Any]:
    module_names = set(modules)
    loaded = []
    missing = []
    unexpected = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for name, module in modules.items():
            for suffix in ("lora_a", "lora_b"):
                key = f"{name}.{suffix}"
                if key not in keys:
                    missing.append(key)
                    continue
                param = getattr(module, suffix)
                tensor = handle.get_tensor(key)
                param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
                loaded.append(key)
            key = f"{name}.log_mag"
            if hasattr(module, "log_mag") and key in keys:
                param = getattr(module, "log_mag")
                tensor = handle.get_tensor(key)
                param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
                loaded.append(key)
        for key in keys:
            root = key.rsplit(".", 1)[0]
            if root not in module_names:
                unexpected.append(key)
    return {
        "path": str(path),
        "loaded_tensors": len(loaded),
        "missing_tensors": missing,
        "unexpected_tensors": unexpected,
    }


@torch.no_grad()
def magnitude_stats(modules: dict[str, LiftModule]) -> dict[str, float]:
    shifts = []
    scale_ratios = []
    for module in modules.values():
        if not getattr(module, "use_dora", False):
            continue
        shift = module.log_mag.detach().float() - module.init_log_mag.detach().float()
        shifts.append(shift.abs().mean())
        scale_ratios.append(shift.exp().mean())
    if not shifts:
        return {"mag_abs_shift": 0.0, "mag_scale_ratio": 1.0}
    return {
        "mag_abs_shift": float(torch.stack(shifts).mean().cpu()),
        "mag_scale_ratio": float(torch.stack(scale_ratios).mean().cpu()),
    }


def projection_alpha_for_step(args: argparse.Namespace, step: int) -> float:
    if not args.projection_aware:
        return 0.0
    if args.projection_alpha_ramp_steps <= 1:
        return float(args.projection_alpha_end)
    denom = max(args.projection_alpha_ramp_steps - 1, 1)
    frac = min(max((step - 1) / denom, 0.0), 1.0)
    start = float(args.projection_alpha_start)
    end = float(args.projection_alpha_end)
    return start + frac * (end - start)


def set_projection_alpha(modules: dict[str, LiftModule], alpha: float) -> None:
    for module in modules.values():
        module.projection_alpha = float(alpha)


def target_ce_loss(logits: torch.Tensor, row: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Teacher-forced CE over cached input tokens.

    The cache stores full prompt+continuation `input_ids`, while Qwen's
    `logits_to_keep=K` returns logits for the last K positions. Logit position
    p predicts token p+1, so only the first K-1 returned logits have known
    targets inside the cached sequence.
    """
    if logits.shape[1] <= 1:
        return logits.sum() * 0.0
    usable = logits.shape[1] - 1
    targets = row["input_ids"][-usable:].to(device=device, dtype=torch.long)
    return F.cross_entropy(
        logits[:, :usable, :].float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--init-weight-patch", action="append", type=Path)
    parser.add_argument("--init-lora", type=Path)
    parser.add_argument("--out-lora", type=Path)
    parser.add_argument("--out-projected-patch", type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--rank-mlp", type=int, default=8)
    parser.add_argument("--rank-attention", type=int, default=4)
    parser.add_argument("--rank-head", type=int, default=0)
    parser.add_argument("--rank-embed", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--train-layers", default="")
    parser.add_argument("--train-regex")
    parser.add_argument("--train-head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-embed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-train-tokens", type=int, default=5000)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--failures")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--optimizer", choices=("bonsai_qat8", "adamw"), default="bonsai_qat8")
    parser.add_argument("--use-muon", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--muon-every", type=int, default=16)
    parser.add_argument("--muon-steps", type=int, default=1)
    parser.add_argument("--muon-max-elements", type=int, default=80_000_000)
    parser.add_argument("--muon-rms-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--format-boost", type=float, default=5.0)
    parser.add_argument("--hidden-weight", type=float, default=0.25)
    parser.add_argument("--anti-weight", type=float, default=0.1)
    parser.add_argument("--target-ce-weight", type=float, default=0.0)
    parser.add_argument("--anti-margin", type=float, default=0.05)
    parser.add_argument("--lora-penalty", type=float, default=1e-4)
    parser.add_argument("--project-mode", choices=("ternary", "binary"), default="ternary")
    parser.add_argument("--project-mlp-mode", choices=("ternary", "binary"))
    parser.add_argument("--project-attention-mode", choices=("ternary", "binary"))
    parser.add_argument("--project-mlp-group-size", type=int)
    parser.add_argument("--project-attention-group-size", type=int)
    parser.add_argument("--projection-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--projection-alpha-start", type=float, default=1.0)
    parser.add_argument("--projection-alpha-end", type=float, default=1.0)
    parser.add_argument("--projection-alpha-ramp-steps", type=int, default=0)
    parser.add_argument("--use-dora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low-memory-projection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.train_layer_set = parse_int_csv(args.train_layers)
    if args.out_lora is None and args.out_projected_patch is None:
        raise RuntimeError("provide --out-lora, --out-projected-patch, or both")

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("format") != "qwen_onpolicy_survival_cache_v1":
        raise ValueError(f"unsupported cache format: {cache.get('format')!r}")
    rows = cache_rows(cache, args)

    model = load_model(args.model_dir)
    init_patch_reports = [apply_weight_patch(model, patch) for patch in args.init_weight_patch or []]
    model.eval()
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)

    lora_modules = attach_lora(model, args)
    if not lora_modules:
        raise RuntimeError("no LoRA modules attached")
    init_lora_report = load_lora_state(args.init_lora, lora_modules) if args.init_lora else None
    train_params = [param for module in lora_modules.values() for param in module.parameters()]
    for param in train_params:
        param.requires_grad_(True)

    if args.optimizer == "bonsai_qat8":
        optimizer = BonsaiQAT8(
            [
                {
                    "params": train_params,
                    "role": "matrix_qat8",
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "betas": (0.9, 0.95),
                    "use_muon": args.use_muon,
                    "muon_every": args.muon_every,
                    "muon_steps": args.muon_steps,
                    "muon_max_elements": args.muon_max_elements,
                    "muon_rms_match": args.muon_rms_match,
                    "prox_rho": 0.0,
                    "max_update_rms": 0.0,
                }
            ]
        )
    else:
        optimizer = torch.optim.AdamW(train_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    device = next(model.parameters()).device
    format_ids = torch.tensor(cache.get("format_token_ids", []), device=device, dtype=torch.long)
    collapse_ids = torch.tensor(cache.get("collapse_token_ids", []), device=device, dtype=torch.long)
    hidden_layers = [layer for layer in cache.get("hidden_layers", []) if args.hidden_weight > 0]
    keep_ref = {"keep": 1}
    handles, captured = register_hidden_hooks(model, hidden_layers, keep_ref)
    history = []
    try:
        for step in range(1, args.steps + 1):
            projection_alpha = projection_alpha_for_step(args, step)
            set_projection_alpha(lora_modules, projection_alpha)
            row = rows[(step - 1) % len(rows)]
            keep_ref["keep"] = int(row["keep"])
            captured.clear()
            outputs = model(**row_to_inputs(row, device), use_cache=False, logits_to_keep=int(row["keep"]))
            kl, stats = topk_teacher_kl(
                outputs.logits,
                row["top_values"],
                row["top_indices"],
                temperature=args.temperature,
                format_ids=format_ids,
                format_boost=args.format_boost,
            )
            anti = anti_collapse_loss(outputs.logits, row["top_values"], row["top_indices"], collapse_ids, margin=args.anti_margin)
            ce = target_ce_loss(outputs.logits, row, device) if args.target_ce_weight > 0 else kl.detach().new_zeros(())
            h_loss = hidden_loss(captured, row, hidden_layers, device) if args.hidden_weight > 0 else kl.detach().new_zeros(())
            penalty_terms = [module.lora_penalty() for module in lora_modules.values()]
            penalty = torch.stack(penalty_terms).mean()
            loss = (
                kl
                + args.target_ce_weight * ce
                + args.hidden_weight * h_loss
                + args.anti_weight * anti
                + args.lora_penalty * penalty
            )
            if not torch.isfinite(loss):
                record = {
                    "step": step,
                    "task": row.get("task"),
                    "id": row.get("id"),
                    "tokens": int(row["tokens"]),
                    "loss": float(loss.detach().float().cpu()),
                    "target_ce": float(ce.detach().float().cpu()),
                    "hidden": float(h_loss.detach().float().cpu()),
                    "anti": float(anti.detach().float().cpu()),
                    "lora_penalty": float(penalty.detach().float().cpu()),
                    "projection_alpha": float(projection_alpha),
                    **magnitude_stats(lora_modules),
                    **stats,
                    "error": "nonfinite_loss",
                }
                print(json.dumps(record), flush=True)
                raise RuntimeError("non-finite Lift-Project QAT loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, args.grad_clip)
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                record = {
                    "step": step,
                    "task": row.get("task"),
                    "id": row.get("id"),
                    "tokens": int(row["tokens"]),
                    "loss": float(loss.detach().cpu()),
                    "target_ce": float(ce.detach().cpu()),
                    "hidden": float(h_loss.detach().cpu()),
                    "anti": float(anti.detach().cpu()),
                    "lora_penalty": float(penalty.detach().cpu()),
                    "projection_alpha": float(projection_alpha),
                    **magnitude_stats(lora_modules),
                    **stats,
                }
                history.append(record)
                print(json.dumps(record), flush=True)
            del outputs, loss, kl, ce, anti, h_loss, penalty
            captured.clear()
            torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()

    projected_modes = None
    if args.out_lora is not None:
        save_lora_state(args.out_lora, lora_modules)
    if args.out_projected_patch is not None:
        projected_modes = save_projected_patch(args.out_projected_patch, lora_modules, args)
    report = {
        "cache": str(args.cache),
        "model_dir": str(args.model_dir),
        "init_weight_patches": [str(path) for path in args.init_weight_patch or []],
        "init_patch_reports": init_patch_reports,
        "init_lora": str(args.init_lora) if args.init_lora else None,
        "init_lora_report": init_lora_report,
        "out_lora": str(args.out_lora) if args.out_lora else None,
        "out_projected_patch": str(args.out_projected_patch) if args.out_projected_patch else None,
        "attached_modules": len(lora_modules),
        "rank_mlp": args.rank_mlp,
        "rank_attention": args.rank_attention,
        "rank_head": args.rank_head,
        "rank_embed": args.rank_embed,
        "lora_dtype": args.lora_dtype,
        "projection_aware": args.projection_aware,
        "projection_alpha_start": args.projection_alpha_start,
        "projection_alpha_end": args.projection_alpha_end,
        "projection_alpha_ramp_steps": args.projection_alpha_ramp_steps,
        "use_dora": args.use_dora,
        "low_memory_projection": args.low_memory_projection,
        "project_mode": args.project_mode,
        "project_mlp_mode": args.project_mlp_mode,
        "project_attention_mode": args.project_attention_mode,
        "project_mlp_group_size": args.project_mlp_group_size,
        "project_attention_group_size": args.project_attention_group_size,
        "group_size": args.group_size,
        "train_layers": sorted(args.train_layer_set),
        "train_regex": args.train_regex,
        "train_head": args.train_head,
        "train_embed": args.train_embed,
        "failures": args.failures,
        "selected_rows": [
            {
                "index": row.get("index"),
                "task": row.get("task"),
                "id": row.get("id"),
                "failure_kind": row.get("failure_kind") or row.get("collapse", {}).get("failure_kind"),
                "tokens": int(row["tokens"]),
            }
            for row in rows
        ],
        "optimizer": args.optimizer,
        "use_muon": args.use_muon,
        "steps": args.steps,
        "lr": args.lr,
        "target_ce_weight": args.target_ce_weight,
        "history": history,
        "projected_modes": projected_modes,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out_report": str(args.out_report), "attached_modules": len(lora_modules)}, indent=2))


if __name__ == "__main__":
    main()
