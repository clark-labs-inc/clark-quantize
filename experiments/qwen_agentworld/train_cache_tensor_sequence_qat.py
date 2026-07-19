"""Sequential cached-teacher QAT for a small list of tensors.

This keeps one model loaded, trains one hard-forward tensor at a time against
cached teacher top-k logits, freezes that tensor as a plain low-bit Linear, and
moves to the next tensor. It is meant for bounded probes before attempting a
larger global QAT run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_cache_tensor_qat import (
    codebook_loss,
    eval_rows,
    first_parameter_device,
    load_rows,
    replace_linear_with_quant,
    row_to_inputs,
    teacher_topk_kl_loss,
)
from fake_quant_policy import resolve_parameter_name
from weight_patch import apply_weight_patch


def load_tensor_list(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [str(item) for item in data]
    if isinstance(data, dict):
        values = data.get("tensors") or data.get("parameters") or data.get("excluded")
        if isinstance(values, list):
            return [str(item) for item in values]
    raise ValueError(f"unsupported tensor list shape: {path}")


def module_parent_and_attr(model: torch.nn.Module, parameter_name: str) -> tuple[torch.nn.Module, str]:
    module_name = parameter_name[: -len(".weight")]
    parent_name, _, attr = module_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, attr


@torch.no_grad()
def freeze_quant_as_linear(
    model: torch.nn.Module,
    parameter_name: str,
    quant_module: torch.nn.Module,
) -> torch.Tensor:
    weight = quant_module.quantized_weight().detach()
    if hasattr(quant_module, "num_embeddings") and hasattr(quant_module, "embedding_dim"):
        embedding = torch.nn.Embedding(
            quant_module.num_embeddings,
            quant_module.embedding_dim,
            padding_idx=quant_module.padding_idx,
            max_norm=quant_module.max_norm,
            norm_type=quant_module.norm_type,
            scale_grad_by_freq=quant_module.scale_grad_by_freq,
            sparse=quant_module.sparse,
            dtype=weight.dtype,
            device=weight.device,
        )
        embedding.weight.data.copy_(weight)
        embedding.weight.requires_grad_(False)
        parent, attr = module_parent_and_attr(model, parameter_name)
        setattr(parent, attr, embedding)
        return weight.to(device="cpu", dtype=torch.bfloat16).contiguous()

    bias = quant_module.bias.detach() if getattr(quant_module, "bias", None) is not None else None
    linear = torch.nn.Linear(
        quant_module.in_features,
        quant_module.out_features,
        bias=bias is not None,
        dtype=weight.dtype,
        device=weight.device,
    )
    linear.weight.data.copy_(weight)
    linear.weight.requires_grad_(False)
    if bias is not None and linear.bias is not None:
        linear.bias.data.copy_(bias)
        linear.bias.requires_grad_(False)
    parent, attr = module_parent_and_attr(model, parameter_name)
    setattr(parent, attr, linear)
    return weight.to(device="cpu", dtype=torch.bfloat16).contiguous()


def train_current_tensor(
    model: torch.nn.Module,
    quant_module: torch.nn.Module,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    steps: int,
    lr: float,
    codebook_weight: float,
    temperature: float,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW([quant_module.weight_master], lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    train_log = []
    for step in range(1, steps + 1):
        row = rows[(step - 1) % len(rows)]
        keep = int(row["top_values"].shape[0])
        outputs = model(**row_to_inputs(row, device), use_cache=False, logits_to_keep=keep)
        kl = teacher_topk_kl_loss(
            outputs.logits,
            row["top_values"],
            row["top_indices"],
            temperature=temperature,
        )
        if codebook_weight > 0:
            c_loss = codebook_loss(quant_module)
        else:
            c_loss = kl.detach().new_zeros(())
        loss = kl + codebook_weight * c_loss
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
    return train_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--tensors-file", required=True, type=Path)
    parser.add_argument("--out-patch", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--context-weight-patch", action="append", type=Path)
    parser.add_argument("--mode", choices=["ternary", "binary"], default="ternary")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-train-tokens", type=int, default=3500)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--steps-per-tensor", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--codebook-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rows = load_rows(args.cache, args.max_train_tokens, args.max_train_rows, args.seed)
    tensors = load_tensor_list(args.tensors_file)

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

    context_patch_results = []
    for patch_path in args.context_weight_patch or []:
        context_patch_results.append(apply_weight_patch(model, patch_path))

    device = first_parameter_device(model)
    patch = {}
    tensor_reports = []
    for tensor_index, tensor_name in enumerate(tensors, start=1):
        params = dict(model.named_parameters())
        parameter_name = resolve_parameter_name(tensor_name, params)
        if parameter_name is None:
            tensor_reports.append({"tensor": tensor_name, "missing": True})
            continue
        print(f"tensor {tensor_index}/{len(tensors)} {parameter_name}", flush=True)
        quant_module = replace_linear_with_quant(
            model,
            parameter_name,
            mode=args.mode,
            group_size=args.group_size,
        )
        quant_module.weight_master.requires_grad_(True)
        if getattr(quant_module, "bias", None) is not None:
            quant_module.bias.requires_grad_(False)

        initial = eval_rows(model, rows, device=device, temperature=args.temperature)
        train_log = train_current_tensor(
            model,
            quant_module,
            rows,
            device=device,
            steps=args.steps_per_tensor,
            lr=args.lr,
            codebook_weight=args.codebook_weight,
            temperature=args.temperature,
        )
        final = eval_rows(model, rows, device=device, temperature=args.temperature)
        patch[parameter_name] = freeze_quant_as_linear(model, parameter_name, quant_module)
        tensor_reports.append(
            {
                "tensor": tensor_name,
                "parameter": parameter_name,
                "initial": initial,
                "final": final,
                "train_log": train_log,
            }
        )
        torch.cuda.empty_cache()
        args.out_patch.parent.mkdir(parents=True, exist_ok=True)
        save_file(patch, args.out_patch)
        partial = {
            "model_dir": str(args.model_dir),
            "cache": str(args.cache),
            "tensors_file": str(args.tensors_file),
            "patch": str(args.out_patch),
            "partial": True,
            "patched_tensors": len(patch),
            "reports": tensor_reports,
        }
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(json.dumps(partial, indent=2) + "\n")

    args.out_patch.parent.mkdir(parents=True, exist_ok=True)
    save_file(patch, args.out_patch)
    result = {
        "model_dir": str(args.model_dir),
        "cache": str(args.cache),
        "tensors_file": str(args.tensors_file),
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
        "steps_per_tensor": args.steps_per_tensor,
        "lr": args.lr,
        "codebook_weight": args.codebook_weight,
        "patched_tensors": len(patch),
        "reports": tensor_reports,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"patch": str(args.out_patch), "patched_tensors": len(patch)}, indent=2))


if __name__ == "__main__":
    main()
