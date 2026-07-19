"""Train one low-bit tensor against cached teacher logits.

This is a bounded teacher-behavior QAT probe. It replaces one linear module
with a hard-forward master-weight quantized module and optimizes the cached
teacher top-k KL on rows that fit the requested training token budget.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clark_air.quant.master_weight_qat import MasterWeightQuantEmbedding, MasterWeightQuantLinear
from fake_quant_policy import resolve_parameter_name
from weight_patch import apply_weight_patch


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def module_parent_and_attr(model: torch.nn.Module, module_name: str) -> tuple[torch.nn.Module, str]:
    parent_name, _, attr = module_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, attr


def module_for_parameter(model: torch.nn.Module, parameter_name: str) -> torch.nn.Module:
    if not parameter_name.endswith(".weight"):
        raise ValueError(f"expected a .weight parameter, got {parameter_name!r}")
    module_name = parameter_name[: -len(".weight")]
    module = model.get_submodule(module_name)
    if isinstance(module, torch.nn.Embedding):
        return module
    if not isinstance(module, torch.nn.Linear):
        # Qwen uses Linear subclasses in some installs; require Linear-like fields.
        if not all(hasattr(module, name) for name in ("weight", "in_features", "out_features")):
            raise TypeError(f"module is not linear/embedding-like: {module_name} -> {type(module)}")
    return module


def replace_linear_with_quant(
    model: torch.nn.Module,
    parameter_name: str,
    *,
    mode: str,
    group_size: int,
) -> MasterWeightQuantLinear | MasterWeightQuantEmbedding:
    module_name = parameter_name[: -len(".weight")]
    module = module_for_parameter(model, parameter_name)
    if isinstance(module, torch.nn.Embedding):
        quant = MasterWeightQuantEmbedding.from_embedding(module, mode=mode, group_size=group_size)
    else:
        quant = MasterWeightQuantLinear.from_linear(
            module,
            mode=mode,
            group_size=group_size,
            train_bias=False,
        )
    parent, attr = module_parent_and_attr(model, module_name)
    setattr(parent, attr, quant)
    return quant


def load_rows(path: Path, max_tokens: int, max_rows: int | None, seed: int) -> list[dict[str, Any]]:
    cache = torch.load(path, map_location="cpu")
    if cache.get("format") != "qwen_agentworld_teacher_behavior_v1":
        raise ValueError(f"unsupported cache format: {cache.get('format')!r}")
    rows = [row for row in cache["rows"] if int(row["tokens"]) <= max_tokens]
    if not rows:
        raise RuntimeError(f"no cache rows fit --max-train-tokens={max_tokens}")
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:max_rows] if max_rows is not None else rows


def teacher_topk_kl_loss(
    student_logits: torch.Tensor,
    teacher_values: torch.Tensor,
    teacher_indices: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    positions = int(teacher_values.shape[0])
    student = student_logits[0, -positions:, :].float() / temperature
    values = teacher_values.to(student.device, dtype=torch.float32) / temperature
    indices = teacher_indices.to(student.device, dtype=torch.long)
    gathered = student.gather(dim=-1, index=indices)
    student_logp_full = gathered - torch.logsumexp(student, dim=-1, keepdim=True)
    teacher_logp = F.log_softmax(values, dim=-1)
    teacher_prob = teacher_logp.exp()
    return (teacher_prob * (teacher_logp - student_logp_full)).sum(dim=-1).mean() * (temperature**2)


def flatten_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple, set)):
        out: list[int] = []
        for item in value:
            out.extend(flatten_token_ids(item))
        return out
    return []


def config_token_ids(config: Any, extra_special: list[int]) -> tuple[list[int], list[int]]:
    eos = flatten_token_ids(getattr(config, "eos_token_id", None))
    special = [
        *eos,
        *flatten_token_ids(getattr(config, "bos_token_id", None)),
        *flatten_token_ids(getattr(config, "pad_token_id", None)),
        *extra_special,
    ]
    return sorted(set(special)), sorted(set(eos))


def lexical_risk_metrics(
    logits: torch.Tensor,
    teacher_indices: torch.Tensor,
    *,
    special_token_ids: list[int],
    eos_token_ids: list[int],
) -> dict[str, float]:
    """First-token collapse proxies for lexical/head QAT."""

    vocab = int(logits.shape[-1])
    last_logits = logits[0, -1, :].float()
    last_argmax = int(last_logits.argmax().item())
    teacher_top1 = teacher_indices.to(logits.device, dtype=torch.long)[:, 0]
    result: dict[str, float] = {
        "last_token_matches_teacher_top1": float(last_argmax == int(teacher_top1[-1].item())),
    }

    def add_token_set(prefix: str, token_ids: list[int]) -> None:
        ids = sorted({int(token_id) for token_id in token_ids if 0 <= int(token_id) < vocab})
        if not ids:
            return
        id_tensor = torch.tensor(ids, device=logits.device, dtype=torch.long)
        log_probs = F.log_softmax(last_logits, dim=-1)
        last_prob = log_probs.index_select(0, id_tensor).exp().sum()
        student_top1 = logits[0].float().argmax(dim=-1)
        student_is_token = torch.isin(student_top1, id_tensor)
        teacher_is_token = torch.isin(teacher_top1, id_tensor)
        result[f"last_{prefix}_prob"] = float(last_prob.item())
        result[f"last_{prefix}_top1"] = float(last_argmax in ids)
        result[f"positions_{prefix}_top1_rate"] = float(student_is_token.float().mean().item())
        result[f"teacher_positions_{prefix}_top1_rate"] = float(teacher_is_token.float().mean().item())

    add_token_set("special", special_token_ids)
    add_token_set("eos", eos_token_ids)
    return result


def codebook_loss(module: torch.nn.Module) -> torch.Tensor:
    master = module.weight_master
    quantized = module.quantized_weight().detach().to(dtype=master.dtype)
    denom = master.detach().square().mean().clamp_min(1e-8)
    return (master - quantized).square().mean() / denom


def row_to_inputs(row: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "input_ids": row["input_ids"].to(device=device, dtype=torch.long).unsqueeze(0),
        "attention_mask": row["attention_mask"].to(device=device, dtype=torch.long).unsqueeze(0),
    }


@torch.no_grad()
def eval_rows(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    temperature: float,
    special_token_ids: list[int] | None = None,
    eos_token_ids: list[int] | None = None,
) -> dict[str, float]:
    losses = []
    top1 = []
    mass = []
    lexical_rows: list[dict[str, float]] = []
    for row in rows:
        keep = int(row["top_values"].shape[0])
        logits = model(**row_to_inputs(row, device), use_cache=False, logits_to_keep=keep).logits
        loss = teacher_topk_kl_loss(
            logits,
            row["top_values"],
            row["top_indices"],
            temperature=temperature,
        )
        student = logits[0, -keep:, :].float() / temperature
        values = row["top_values"].to(student.device, dtype=torch.float32) / temperature
        indices = row["top_indices"].to(student.device, dtype=torch.long)
        gathered = student.gather(dim=-1, index=indices)
        logp_full = gathered - torch.logsumexp(student, dim=-1, keepdim=True)
        mass.append(float(logp_full.exp().sum(dim=-1).mean().item()))
        top1.append(float((student.argmax(dim=-1) == indices[:, 0]).float().mean().item()))
        losses.append(float(loss.item()))
        lexical_rows.append(
            lexical_risk_metrics(
                logits,
                row["top_indices"],
                special_token_ids=special_token_ids or [],
                eos_token_ids=eos_token_ids or [],
            )
        )
    result = {
        "teacher_topk_kl_full_mean": sum(losses) / max(len(losses), 1),
        "student_mass_on_teacher_topk_mean": sum(mass) / max(len(mass), 1),
        "top1_agreement": sum(top1) / max(len(top1), 1),
    }
    for key in sorted({key for row in lexical_rows for key in row}):
        values = [row[key] for row in lexical_rows if key in row]
        result[f"{key}_mean"] = sum(values) / max(len(values), 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--out-patch", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--context-weight-patch", action="append", type=Path)
    parser.add_argument("--mode", choices=["ternary", "binary"], default="ternary")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-train-tokens", type=int, default=9000)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--codebook-weight", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--special-token-id",
        action="append",
        type=int,
        default=[],
        help="extra special/stop token id to track as a collapse proxy; repeatable",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    rows = load_rows(args.cache, args.max_train_tokens, args.max_train_rows, args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    special_token_ids, eos_token_ids = config_token_ids(model.config, args.special_token_id)

    context_patch_results = []
    for patch_path in args.context_weight_patch or []:
        context_patch_results.append(apply_weight_patch(model, patch_path))

    params = dict(model.named_parameters())
    parameter_name = resolve_parameter_name(args.tensor, params)
    if parameter_name is None:
        raise RuntimeError(f"tensor did not resolve to a model parameter: {args.tensor}")

    quant_module = replace_linear_with_quant(
        model,
        parameter_name,
        mode=args.mode,
        group_size=args.group_size,
    )
    quant_module.weight_master.requires_grad_(True)
    if getattr(quant_module, "bias", None) is not None:
        quant_module.bias.requires_grad_(False)

    device = first_parameter_device(model)
    optimizer = torch.optim.AdamW([quant_module.weight_master], lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    initial = eval_rows(
        model,
        rows,
        device=device,
        temperature=args.temperature,
        special_token_ids=special_token_ids,
        eos_token_ids=eos_token_ids,
    )
    train_log = []
    for step in range(1, args.steps + 1):
        row = rows[(step - 1) % len(rows)]
        keep = int(row["top_values"].shape[0])
        outputs = model(**row_to_inputs(row, device), use_cache=False, logits_to_keep=keep)
        kl = teacher_topk_kl_loss(
            outputs.logits,
            row["top_values"],
            row["top_indices"],
            temperature=args.temperature,
        )
        if args.codebook_weight > 0:
            c_loss = codebook_loss(quant_module)
        else:
            c_loss = kl.detach().new_zeros(())
        loss = kl + args.codebook_weight * c_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([quant_module.weight_master], 1.0)
        optimizer.step()
        record = {
            "step": step,
            "task": row.get("task"),
            "tokens": int(row["tokens"]),
            "kl": float(kl.detach().item()),
            "codebook": float(c_loss.detach().item()),
            "loss": float(loss.detach().item()),
        }
        train_log.append(record)
        print(json.dumps(record), flush=True)

    final = eval_rows(
        model,
        rows,
        device=device,
        temperature=args.temperature,
        special_token_ids=special_token_ids,
        eos_token_ids=eos_token_ids,
    )
    quantized = quant_module.quantized_weight().detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    args.out_patch.parent.mkdir(parents=True, exist_ok=True)
    save_file({parameter_name: quantized}, args.out_patch)

    result = {
        "model_dir": str(args.model_dir),
        "cache": str(args.cache),
        "tensor": args.tensor,
        "parameter": parameter_name,
        "patch": str(args.out_patch),
        "context_weight_patches": [str(path) for path in args.context_weight_patch or []],
        "context_patch_results": [
            {
                "patch": row.get("patch"),
                "applied_count": row.get("applied_count"),
                "missing_count": row.get("missing_count"),
            }
            for row in context_patch_results
        ],
        "mode": args.mode,
        "group_size": args.group_size,
        "max_train_tokens": args.max_train_tokens,
        "train_rows": [
            {"task": row.get("task"), "id": row.get("id"), "tokens": int(row["tokens"])}
            for row in rows
        ],
        "steps": args.steps,
        "lr": args.lr,
        "codebook_weight": args.codebook_weight,
        "special_token_ids": special_token_ids,
        "eos_token_ids": eos_token_ids,
        "initial": initial,
        "final": final,
        "train_log": train_log,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"patch": str(args.out_patch), "initial": initial, "final": final}, indent=2))


if __name__ == "__main__":
    main()
