"""Activation-local Projection-Aware DoRA QAT for Qwen linear tensors.

This is a memory-feasible probe for the Bonsai/Prism-style hypothesis:
train a low-rank direction plus group magnitude scaffold while the forward
weight is always projected to the deployable binary/ternary codebook, then
bake the scaffold away into a plain weight patch.

Unlike full-model lift-project training, this script trains one tensor at a
time on cached module activations. It is intended for large MLP projections
that do not fit through the full generation graph on one A100.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clark_quantize.bonsai_qat8 import BonsaiQAT8
from fake_quant_policy import resolve_parameter_name
from train_all2d_ternary_qat import alpha_quantized_weight
from train_mlp_triplet_survival_qat import load_model, row_to_inputs
from weight_patch import apply_weight_patch


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def load_tensor_list(args: argparse.Namespace) -> list[str]:
    out = parse_csv(args.tensors)
    if args.tensors_file is not None:
        data = json.loads(args.tensors_file.read_text())
        if isinstance(data, list):
            out.extend(str(item) for item in data)
        elif isinstance(data, dict):
            values = data.get("tensors") or data.get("parameters") or data.get("weights")
            if isinstance(values, list):
                out.extend(str(item) for item in values)
            else:
                raise ValueError(f"unsupported tensor list dict shape: {args.tensors_file}")
        else:
            raise ValueError(f"unsupported tensor list shape: {args.tensors_file}")
    if not out:
        raise RuntimeError("provide --tensors or --tensors-file")
    return out


def parameter_to_module(parameter_name: str) -> str:
    if not parameter_name.endswith(".weight"):
        raise ValueError(f"expected a .weight parameter, got {parameter_name!r}")
    return parameter_name[: -len(".weight")]


def cache_rows(cache: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [row for row in cache["rows"] if int(row["tokens"]) <= args.max_train_tokens]
    if args.failures:
        wanted = {part.strip() for part in args.failures.split(",") if part.strip()}
        rows = [
            row
            for row in rows
            if row.get("failure_kind") in wanted or row.get("collapse", {}).get("failure_kind") in wanted
        ]
    rng = random.Random(args.seed)
    if args.shuffle_rows:
        rng.shuffle(rows)
    if args.max_train_rows is not None:
        rows = rows[: args.max_train_rows]
    if not rows:
        raise RuntimeError("no cache rows selected")
    return rows


def token_sample(flat: torch.Tensor, need: int) -> torch.Tensor:
    if flat.shape[0] <= need:
        return flat
    index = torch.linspace(0, flat.shape[0] - 1, need, device=flat.device).round().long()
    return flat.index_select(0, index)


@torch.no_grad()
def collect_activation_banks(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    modules: dict[str, torch.nn.Module],
    *,
    tokens_per_module: int,
) -> dict[str, torch.Tensor]:
    banks: dict[str, list[torch.Tensor]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    handles = []

    def make_hook(name: str):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if counts[name] >= tokens_per_module or not inputs:
                return
            x = inputs[0].detach()
            if x.ndim < 2:
                return
            flat = x.reshape(-1, x.shape[-1])
            need = tokens_per_module - counts[name]
            chunk = token_sample(flat, need)
            banks[name].append(chunk.to(device="cpu", dtype=torch.bfloat16))
            counts[name] += int(chunk.shape[0])

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    try:
        device = next(model.parameters()).device
        for row_index, row in enumerate(rows, start=1):
            model(**row_to_inputs(row, device), use_cache=False, logits_to_keep=1)
            ready = sum(1 for name in modules if counts[name] >= tokens_per_module)
            print(f"activation row {row_index}/{len(rows)} ready={ready}/{len(modules)}", flush=True)
            if ready == len(modules):
                break
    finally:
        for handle in handles:
            handle.remove()

    return {
        name: torch.cat(chunks, dim=0)[:tokens_per_module].contiguous()
        for name, chunks in banks.items()
        if chunks
    }


@torch.no_grad()
def group_rms_log(weight: torch.Tensor, *, group_size: int) -> torch.Tensor:
    source = weight.detach().float()
    rows, cols = source.shape
    pad = (-cols) % group_size
    if pad:
        source = F.pad(source, (0, pad))
    groups = source.view(rows, -1, group_size)
    return groups.square().mean(dim=-1).sqrt().clamp_min(1e-8).log()


def apply_group_magnitude(weight: torch.Tensor, log_mag: torch.Tensor, *, group_size: int) -> torch.Tensor:
    rows, cols = weight.shape
    pad = (-cols) % group_size
    source = F.pad(weight.float(), (0, pad)) if pad else weight.float()
    groups = source.view(rows, -1, group_size)
    rms = groups.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    scaled = (groups / rms) * log_mag.float().exp().unsqueeze(-1)
    return scaled.reshape(rows, -1)[:, :cols]


class ActivationLiftProject(torch.nn.Module):
    def __init__(
        self,
        base_weight: torch.Tensor,
        *,
        rank: int,
        alpha: float,
        mode: str,
        group_size: int,
        use_dora: bool,
        parameter_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.mode = mode
        self.group_size = group_size
        self.use_dora = use_dora
        self.register_buffer("base_weight", base_weight.detach().to(dtype=torch.bfloat16))
        out_features, in_features = base_weight.shape
        self.lora_a = torch.nn.Parameter(torch.empty(rank, in_features, dtype=parameter_dtype))
        self.lora_b = torch.nn.Parameter(torch.zeros(out_features, rank, dtype=parameter_dtype))
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        init_log_mag = group_rms_log(base_weight, group_size=group_size)
        self.register_buffer("init_log_mag", init_log_mag)
        self.log_mag = torch.nn.Parameter(init_log_mag.to(dtype=parameter_dtype).clone())

    @property
    def scaling(self) -> float:
        return self.alpha / max(self.rank, 1)

    def effective_weight(self) -> torch.Tensor:
        delta = self.lora_b.float() @ self.lora_a.float()
        lifted = self.base_weight.float() + self.scaling * delta
        if not self.use_dora:
            return lifted
        return apply_group_magnitude(lifted, self.log_mag, group_size=self.group_size)

    def projected_weight(self) -> torch.Tensor:
        return alpha_quantized_weight(
            self.effective_weight(),
            mode=self.mode,
            group_size=self.group_size,
            alpha=1.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.projected_weight()
        return F.linear(x.float(), weight.float())

    def penalty(self) -> torch.Tensor:
        penalty = self.lora_a.float().square().mean() + self.lora_b.float().square().mean()
        if self.use_dora:
            penalty = penalty + 0.01 * (self.log_mag.float() - self.init_log_mag.float()).square().mean()
        return penalty

    @torch.no_grad()
    def baked_weight(self) -> torch.Tensor:
        return self.projected_weight().detach().cpu().to(torch.bfloat16).contiguous()


def relative_rmse(target: torch.Tensor, pred: torch.Tensor) -> float:
    err = (pred.float() - target.float()).square().mean().sqrt()
    denom = target.float().square().mean().sqrt().clamp_min(1e-8)
    return float((err / denom).item())


@torch.no_grad()
def eval_local(
    model: ActivationLiftProject,
    inputs: torch.Tensor,
    target_weight: torch.Tensor,
    *,
    batch_tokens: int,
) -> dict[str, float]:
    batch = inputs[: min(inputs.shape[0], batch_tokens)].to(device=target_weight.device, dtype=torch.bfloat16)
    target = F.linear(batch.float(), target_weight.float())
    pred = model(batch)
    baked = model.baked_weight().to(device=target_weight.device, dtype=target_weight.dtype)
    base_q = alpha_quantized_weight(
        model.base_weight.float(),
        mode=model.mode,
        group_size=model.group_size,
        alpha=1.0,
    ).to(device=target_weight.device, dtype=target_weight.dtype)
    return {
        "output_relative_rmse": relative_rmse(target, pred),
        "weight_relative_rmse": relative_rmse(target_weight, baked),
        "changed_fraction_vs_initial_projected": float((baked.cpu() != base_q.cpu()).float().mean().item()),
        "mean_abs_delta_vs_initial_projected": float((baked.cpu().float() - base_q.cpu().float()).abs().mean().item()),
    }


def train_one_tensor(
    *,
    inputs_cpu: torch.Tensor,
    base_weight_cpu: torch.Tensor,
    target_weight_cpu: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    tensor_index: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    inputs = inputs_cpu.to(device=device, dtype=torch.bfloat16)
    target_weight = target_weight_cpu.to(device=device, dtype=torch.float32)
    module = ActivationLiftProject(
        base_weight_cpu.to(device=device),
        rank=args.rank,
        alpha=args.lora_alpha,
        mode=args.mode,
        group_size=args.group_size,
        use_dora=args.use_dora,
        parameter_dtype=torch.bfloat16 if args.lora_dtype == "bfloat16" else torch.float32,
    ).to(device)
    params = list(module.parameters())
    if args.optimizer == "bonsai_qat8":
        optimizer = BonsaiQAT8(
            [
                {
                    "params": params,
                    "role": "matrix_qat8",
                    "lr": args.lr,
                    "weight_decay": 0.0,
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
        optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + tensor_index)
    initial = eval_local(module, inputs, target_weight, batch_tokens=args.batch_tokens)
    history = []
    for step in range(1, args.steps + 1):
        if inputs.shape[0] <= args.batch_tokens:
            batch = inputs
        else:
            idx = torch.randint(inputs.shape[0], (args.batch_tokens,), generator=generator)
            batch = inputs.index_select(0, idx.to(inputs.device))
        with torch.no_grad():
            target = F.linear(batch.float(), target_weight.float())
        pred = module(batch)
        fit = (pred - target).square().mean()
        penalty = module.penalty()
        loss = fit + args.lora_penalty * penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": step,
                "fit": float(fit.detach().cpu()),
                "penalty": float(penalty.detach().cpu()),
                "loss": float(loss.detach().cpu()),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    final = eval_local(module, inputs, target_weight, batch_tokens=args.batch_tokens)
    baked = module.baked_weight()
    report = {
        "initial": initial,
        "final": final,
        "history": history,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "use_dora": args.use_dora,
        "mode": args.mode,
    }
    del module, optimizer, inputs, target_weight
    torch.cuda.empty_cache()
    return baked, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--tensors")
    parser.add_argument("--tensors-file", type=Path)
    parser.add_argument("--context-weight-patch", action="append", type=Path)
    parser.add_argument("--out-patch", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--mode", choices=("ternary", "binary"), default="ternary")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--use-dora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tokens-per-module", type=int, default=256)
    parser.add_argument("--batch-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=("bonsai_qat8", "adamw"), default="bonsai_qat8")
    parser.add_argument("--use-muon", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--muon-every", type=int, default=4)
    parser.add_argument("--muon-steps", type=int, default=1)
    parser.add_argument("--muon-max-elements", type=int, default=80_000_000)
    parser.add_argument("--muon-rms-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-penalty", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--max-train-tokens", type=int, default=9000)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--failures")
    parser.add_argument("--shuffle-rows", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("format") not in {"qwen_onpolicy_survival_cache_v1", "qwen_agentworld_teacher_behavior_v1"}:
        raise ValueError(f"unsupported cache format: {cache.get('format')!r}")
    rows = cache_rows(cache, args)
    requested_tensors = load_tensor_list(args)

    model = load_model(args.model_dir)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    params_before_patch = dict(model.named_parameters())
    resolved: list[tuple[str, str]] = []
    target_weights: dict[str, torch.Tensor] = {}
    for tensor in requested_tensors:
        parameter_name = resolve_parameter_name(tensor, params_before_patch)
        if parameter_name is None:
            raise RuntimeError(f"could not resolve tensor: {tensor}")
        resolved.append((tensor, parameter_name))
        target_weights[parameter_name] = params_before_patch[parameter_name].detach().cpu().to(torch.bfloat16).contiguous()

    context_patch_reports = []
    for patch in args.context_weight_patch or []:
        context_patch_reports.append(apply_weight_patch(model, patch))

    params_after_patch = dict(model.named_parameters())
    modules: dict[str, torch.nn.Module] = {}
    base_weights: dict[str, torch.Tensor] = {}
    for _tensor, parameter_name in resolved:
        module_name = parameter_to_module(parameter_name)
        modules[module_name] = model.get_submodule(module_name)
        base_weights[parameter_name] = params_after_patch[parameter_name].detach().cpu().to(torch.bfloat16).contiguous()

    banks = collect_activation_banks(model, rows, modules, tokens_per_module=args.tokens_per_module)
    device = next(model.parameters()).device
    patch_tensors: dict[str, torch.Tensor] = {}
    tensor_reports = []
    for index, (requested, parameter_name) in enumerate(resolved, start=1):
        module_name = parameter_to_module(parameter_name)
        if module_name not in banks:
            tensor_reports.append({"tensor": requested, "parameter": parameter_name, "missing_bank": True})
            continue
        print(f"tensor {index}/{len(resolved)} {parameter_name}", flush=True)
        baked, report = train_one_tensor(
            inputs_cpu=banks[module_name],
            base_weight_cpu=base_weights[parameter_name],
            target_weight_cpu=target_weights[parameter_name],
            args=args,
            device=device,
            tensor_index=index,
        )
        patch_tensors[parameter_name] = baked
        tensor_reports.append(
            {
                "tensor": requested,
                "parameter": parameter_name,
                "module": module_name,
                "activation_tokens": int(banks[module_name].shape[0]),
                **report,
            }
        )
        args.out_patch.parent.mkdir(parents=True, exist_ok=True)
        save_file(patch_tensors, str(args.out_patch))

    result = {
        "model_dir": str(args.model_dir),
        "cache": str(args.cache),
        "context_weight_patches": [str(path) for path in args.context_weight_patch or []],
        "context_patch_reports": [
            {
                "patch": row.get("patch"),
                "applied_count": row.get("applied_count"),
                "missing_count": row.get("missing_count"),
            }
            for row in context_patch_reports
        ],
        "out_patch": str(args.out_patch),
        "mode": args.mode,
        "group_size": args.group_size,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "use_dora": args.use_dora,
        "tokens_per_module": args.tokens_per_module,
        "batch_tokens": args.batch_tokens,
        "steps": args.steps,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "use_muon": args.use_muon,
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
        "reports": tensor_reports,
    }
    args.out_patch.parent.mkdir(parents=True, exist_ok=True)
    save_file(patch_tensors, str(args.out_patch))
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out_patch": str(args.out_patch), "out_report": str(args.out_report), "tensors": len(patch_tensors)}, indent=2))


if __name__ == "__main__":
    main()
