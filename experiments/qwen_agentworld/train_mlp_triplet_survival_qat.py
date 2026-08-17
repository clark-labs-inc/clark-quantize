"""Sequential on-policy survival QAT for Qwen MLP triplets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clark_air.quant.master_weight_qat import ste_quantized_weight
from clark_quantize.qwen_qat import hard_ternary_ste
from weight_patch import apply_weight_patch


MODEL_LOADER_CANDIDATES = (
    "AutoModelForImageTextToText",
    "AutoModelForMultimodalLM",
    "AutoModelForCausalLM",
    "AutoModelForVision2Seq",
)
BLOCK_RE = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)$")


def load_model(model_dir: Path, *, device: str = "auto") -> torch.nn.Module:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported device request: {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device_map: str | dict[str, str] = "auto" if device == "auto" else {"": device}
    errors = []
    for loader_name in MODEL_LOADER_CANDIDATES:
        loader = getattr(transformers, loader_name, None)
        if loader is None:
            continue
        try:
            model = loader.from_pretrained(
                model_dir,
                dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            print(f"loaded_model loader={loader_name} class={model.__class__.__name__}", flush=True)
            model.eval()
            return model
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{loader_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("could not load model:\n" + "\n".join(errors))


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def parse_layers(value: str, available: set[int]) -> list[int]:
    if value == "all":
        return sorted(available)
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def layer_tensor_name(layer_id: int, proj: str) -> str:
    return f"model.language_model.layers.{layer_id}.mlp.{proj}_proj.weight"


def mlp_module_name(layer_id: int) -> str:
    return f"model.language_model.layers.{layer_id}.mlp"


def get_parent_and_attr(model: torch.nn.Module, module_name: str) -> tuple[torch.nn.Module, str]:
    parent_name, _, attr = module_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, attr


class QuantSwiGLUMLP(torch.nn.Module):
    def __init__(
        self,
        source: torch.nn.Module,
        *,
        mode: str,
        group_size: int,
        artifact_exact_ternary: bool = False,
    ) -> None:
        super().__init__()
        self.gate = torch.nn.Parameter(source.gate_proj.weight.detach().float().clone())
        self.up = torch.nn.Parameter(source.up_proj.weight.detach().float().clone())
        self.down = torch.nn.Parameter(source.down_proj.weight.detach().float().clone())
        self.mode = mode
        self.group_size = group_size
        self.artifact_exact_ternary = artifact_exact_ternary

    def quantized(self, master: torch.Tensor) -> torch.Tensor:
        if self.mode == "ternary" and self.artifact_exact_ternary:
            return hard_ternary_ste(master, group_size=self.group_size, row_batch=128)
        return ste_quantized_weight(master, mode=self.mode, group_size=self.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.linear(x, self.quantized(self.gate).to(dtype=x.dtype))
        up = F.linear(x, self.quantized(self.up).to(dtype=x.dtype))
        return F.linear(F.silu(gate) * up, self.quantized(self.down).to(dtype=x.dtype))

    def codebook_loss(self) -> torch.Tensor:
        losses = []
        for master in (self.gate, self.up, self.down):
            with torch.no_grad():
                quantized = ste_quantized_weight(master, mode=self.mode, group_size=self.group_size).detach()
            denom = master.detach().float().square().mean().clamp_min(1e-8)
            losses.append((master.float() - quantized.float()).square().mean() / denom)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def hard_weights(self) -> dict[str, torch.Tensor]:
        return {
            "gate": ste_quantized_weight(self.gate, mode=self.mode, group_size=self.group_size).detach().cpu().to(torch.bfloat16),
            "up": ste_quantized_weight(self.up, mode=self.mode, group_size=self.group_size).detach().cpu().to(torch.bfloat16),
            "down": ste_quantized_weight(self.down, mode=self.mode, group_size=self.group_size).detach().cpu().to(torch.bfloat16),
        }


class FrozenSwiGLUMLP(torch.nn.Module):
    def __init__(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(gate.shape[1], gate.shape[0], bias=False, device=device, dtype=dtype)
        self.up_proj = torch.nn.Linear(up.shape[1], up.shape[0], bias=False, device=device, dtype=dtype)
        self.down_proj = torch.nn.Linear(down.shape[1], down.shape[0], bias=False, device=device, dtype=dtype)
        self.gate_proj.weight.data.copy_(gate.to(device=device, dtype=dtype))
        self.up_proj.weight.data.copy_(up.to(device=device, dtype=dtype))
        self.down_proj.weight.data.copy_(down.to(device=device, dtype=dtype))
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def replace_mlp_with_quant(
    model: torch.nn.Module,
    layer_id: int,
    *,
    mode: str,
    group_size: int,
    artifact_exact_ternary: bool = False,
) -> QuantSwiGLUMLP:
    module_name = mlp_module_name(layer_id)
    source = model.get_submodule(module_name)
    quant = QuantSwiGLUMLP(
        source,
        mode=mode,
        group_size=group_size,
        artifact_exact_ternary=artifact_exact_ternary,
    ).to(next(source.parameters()).device)
    parent, attr = get_parent_and_attr(model, module_name)
    setattr(parent, attr, quant)
    return quant


def freeze_quant_mlp(model: torch.nn.Module, layer_id: int, quant: QuantSwiGLUMLP) -> dict[str, torch.Tensor]:
    hard = quant.hard_weights()
    device = next(quant.parameters()).device
    dtype = torch.bfloat16
    frozen = FrozenSwiGLUMLP(hard["gate"], hard["up"], hard["down"], device=device, dtype=dtype)
    parent, attr = get_parent_and_attr(model, mlp_module_name(layer_id))
    setattr(parent, attr, frozen)
    return hard


def row_to_inputs(row: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    inputs: dict[str, torch.Tensor] = {
        "input_ids": row["input_ids"].to(device=device, dtype=torch.long).unsqueeze(0),
        "attention_mask": row["attention_mask"].to(device=device, dtype=torch.long).unsqueeze(0),
    }
    optional_tensor_keys = (
        "mm_token_type_ids",
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
    )
    batched_optional = {"mm_token_type_ids"}
    for key in optional_tensor_keys:
        value = row.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.to(device=device)
        if key in batched_optional and tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        inputs[key] = tensor
    return inputs


def topk_teacher_kl(
    student_logits: torch.Tensor,
    teacher_values: torch.Tensor,
    teacher_indices: torch.Tensor,
    *,
    temperature: float,
    format_ids: torch.Tensor | None,
    format_boost: float,
    teacher_logsumexp: torch.Tensor | None = None,
    use_other_bucket: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    keep = int(teacher_values.shape[0])
    student = student_logits[0, -keep:, :].float() / temperature
    values = teacher_values.to(student.device, dtype=torch.float32) / temperature
    indices = teacher_indices.to(student.device, dtype=torch.long)
    gathered = student.gather(dim=-1, index=indices)
    student_logz = torch.logsumexp(student, dim=-1, keepdim=True)
    student_logp = gathered - student_logz
    # Historical caches only contain the teacher top-k logits. They are treated
    # as a conditional distribution over top-k. Dense regime caches also store
    # the teacher full-vocab logsumexp, allowing a proper coarsened KL with a
    # single "other" bucket for the omitted tail mass.
    if use_other_bucket and teacher_logsumexp is not None:
        if abs(float(temperature) - 1.0) > 1e-6:
            raise ValueError("top-k other bucket currently requires temperature=1")
        teacher_logz = teacher_logsumexp.to(student.device, dtype=torch.float32).reshape(-1, 1)
        teacher_logp = teacher_values.to(student.device, dtype=torch.float32) - teacher_logz
        teacher_prob = teacher_logp.exp()
        teacher_top_mass = teacher_prob.sum(dim=-1).clamp(max=1.0)
        student_top_mass = student_logp.exp().sum(dim=-1).clamp(max=1.0)
        teacher_other = (1.0 - teacher_top_mass).clamp_min(1e-8)
        student_other = (1.0 - student_top_mass).clamp_min(1e-8)
        top_term = (teacher_prob * (teacher_logp - student_logp)).sum(dim=-1)
        other_term = teacher_other * (teacher_other.log() - student_other.log())
        per_pos = (top_term + other_term) * (temperature**2)
    else:
        teacher_logp = F.log_softmax(values, dim=-1)
        teacher_prob = teacher_logp.exp()
        teacher_top_mass = torch.ones_like(teacher_prob.sum(dim=-1))
        student_top_mass = student_logp.exp().sum(dim=-1)
        per_pos = (teacher_prob * (teacher_logp - student_logp)).sum(dim=-1) * (temperature**2)
    weights = torch.ones_like(per_pos)
    format_mass = torch.zeros_like(per_pos)
    if format_ids is not None and format_ids.numel():
        matches = indices.unsqueeze(-1).eq(format_ids.reshape(1, 1, -1)).any(dim=-1)
        format_mass = (teacher_prob * matches.float()).sum(dim=-1)
        weights = weights + format_boost * format_mass
    loss = (per_pos * weights).mean()
    with torch.no_grad():
        top1 = (student.argmax(dim=-1) == indices[:, 0]).float().mean()
        mass = student_logp.exp().sum(dim=-1).mean()
    return loss, {
        "kl": float(per_pos.mean().detach().cpu()),
        "weighted_kl": float(loss.detach().cpu()),
        "format_mass": float(format_mass.mean().detach().cpu()),
        "top1": float(top1.detach().cpu()),
        "mass": float(mass.detach().cpu()),
        "teacher_top_mass": float(teacher_top_mass.mean().detach().cpu()),
        "student_top_mass": float(student_top_mass.mean().detach().cpu()),
        "other_bucket": float(bool(use_other_bucket and teacher_logsumexp is not None)),
    }


def anti_collapse_loss(
    student_logits: torch.Tensor,
    teacher_values: torch.Tensor,
    teacher_indices: torch.Tensor,
    collapse_ids: torch.Tensor | None,
    *,
    margin: float,
) -> torch.Tensor:
    if collapse_ids is None or not collapse_ids.numel():
        return student_logits.sum() * 0.0
    keep = int(teacher_values.shape[0])
    student = student_logits[0, -keep:, :].float()
    ps = torch.softmax(student, dim=-1).index_select(-1, collapse_ids).sum(dim=-1)
    values = teacher_values.to(student.device, dtype=torch.float32)
    indices = teacher_indices.to(student.device, dtype=torch.long)
    teacher_prob = F.softmax(values, dim=-1)
    matches = indices.unsqueeze(-1).eq(collapse_ids.reshape(1, 1, -1)).any(dim=-1)
    pt = (teacher_prob * matches.float()).sum(dim=-1)
    return F.relu(ps - pt - margin).square().mean()


def register_hidden_hooks(model: torch.nn.Module, layers: list[int], keep_ref: dict[str, int]) -> tuple[list[Any], dict[int, torch.Tensor]]:
    captured: dict[int, torch.Tensor] = {}
    wanted = set(layers)
    handles = []

    def make_hook(layer_id: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            keep = int(keep_ref["keep"])
            captured[layer_id] = hidden[:, -keep:, :].squeeze(0)

        return hook

    for name, module in model.named_modules():
        match = BLOCK_RE.match(name)
        if not match:
            continue
        layer_id = int(match.group(1))
        if layer_id in wanted:
            handles.append(module.register_forward_hook(make_hook(layer_id)))
    if wanted and len(handles) != len(wanted):
        raise RuntimeError(f"registered {len(handles)} hidden hooks, wanted {len(wanted)}")
    return handles, captured


def hidden_loss(
    captured: dict[int, torch.Tensor],
    row: dict[str, Any],
    layers: list[int],
    device: torch.device,
) -> torch.Tensor:
    losses = []
    teacher_hidden = row.get("teacher_hidden", {})
    for layer_id in layers:
        if layer_id not in captured or layer_id not in teacher_hidden:
            continue
        student = captured[layer_id].float()
        teacher = teacher_hidden[layer_id].to(device=device, dtype=torch.float32)
        losses.append((student - teacher).square().mean() / teacher.square().mean().clamp_min(1e-8))
    if not losses:
        return next(iter(captured.values())).sum() * 0.0 if captured else torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def train_layer(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    layer_id: int,
    args: argparse.Namespace,
    device: torch.device,
    hidden_layers: list[int],
    format_ids: torch.Tensor | None,
    collapse_ids: torch.Tensor | None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    quant = replace_mlp_with_quant(model, layer_id, mode=args.mode, group_size=args.group_size)
    for param in quant.parameters():
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(quant.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    keep_ref = {"keep": 1}
    handles, captured = register_hidden_hooks(model, hidden_layers, keep_ref)
    history = []
    try:
        for step in range(1, args.steps_per_layer + 1):
            row = rows[(step - 1) % len(rows)]
            keep_ref["keep"] = int(row["keep"])
            captured.clear()
            outputs = model(
                **row_to_inputs(row, device),
                use_cache=False,
                logits_to_keep=int(row["keep"]),
            )
            kl, stats = topk_teacher_kl(
                outputs.logits,
                row["top_values"],
                row["top_indices"],
                temperature=args.temperature,
                format_ids=format_ids,
                format_boost=args.format_boost,
            )
            anti = anti_collapse_loss(
                outputs.logits,
                row["top_values"],
                row["top_indices"],
                collapse_ids,
                margin=args.anti_margin,
            )
            h_loss = hidden_loss(captured, row, hidden_layers, device) if args.hidden_weight > 0 else kl.detach().new_zeros(())
            code = quant.codebook_loss() if args.code_weight > 0 else kl.detach().new_zeros(())
            loss = kl + args.hidden_weight * h_loss + args.anti_weight * anti + args.code_weight * code
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(quant.parameters(), args.grad_clip)
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps_per_layer:
                record = {
                    "layer": layer_id,
                    "step": step,
                    "task": row.get("task"),
                    "id": row.get("id"),
                    "tokens": int(row["tokens"]),
                    "loss": float(loss.detach().cpu()),
                    "hidden": float(h_loss.detach().cpu()),
                    "anti": float(anti.detach().cpu()),
                    "code": float(code.detach().cpu()),
                    **stats,
                }
                history.append(record)
                print(json.dumps(record), flush=True)
            del outputs, loss, kl, anti, h_loss, code
            captured.clear()
            torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()
    hard = freeze_quant_mlp(model, layer_id, quant)
    return hard, {"layer": layer_id, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--base-patch", action="append", type=Path)
    parser.add_argument("--out-patch", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--layers", default="0,1,10,12,27,28,30,31")
    parser.add_argument("--max-train-tokens", type=int, default=8192)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--mode", choices=("ternary", "binary"), default="ternary")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--steps-per-layer", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--format-boost", type=float, default=5.0)
    parser.add_argument("--hidden-weight", type=float, default=0.05)
    parser.add_argument("--anti-weight", type=float, default=0.1)
    parser.add_argument("--anti-margin", type=float, default=0.05)
    parser.add_argument("--code-weight", type=float, default=0.002)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")

    cache = torch.load(args.cache, map_location="cpu")
    if cache.get("format") != "qwen_onpolicy_survival_cache_v1":
        raise ValueError(f"unsupported cache format: {cache.get('format')!r}")
    rows = [row for row in cache["rows"] if int(row["tokens"]) <= args.max_train_tokens]
    if args.max_train_rows is not None:
        rows = rows[: args.max_train_rows]
    if not rows:
        raise RuntimeError("no cache rows selected")
    available = set(range(32))
    selected_layers = parse_layers(args.layers, available)
    hidden_layers = [layer for layer in cache.get("hidden_layers", []) if layer in selected_layers]
    if args.hidden_weight <= 0:
        hidden_layers = []

    model = load_model(args.model_dir)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    patch_reports = []
    for patch in args.base_patch or []:
        patch_reports.append(apply_weight_patch(model, patch))
    device = first_parameter_device(model)
    format_ids = torch.tensor(cache.get("format_token_ids", []), device=device, dtype=torch.long)
    collapse_ids = torch.tensor(cache.get("collapse_token_ids", []), device=device, dtype=torch.long)

    output: dict[str, torch.Tensor] = {}
    reports = []
    for layer_id in selected_layers:
        hard, report = train_layer(
            model,
            rows,
            layer_id,
            args,
            device,
            hidden_layers,
            format_ids,
            collapse_ids,
        )
        output[layer_tensor_name(layer_id, "gate")] = hard["gate"]
        output[layer_tensor_name(layer_id, "up")] = hard["up"]
        output[layer_tensor_name(layer_id, "down")] = hard["down"]
        reports.append(report)
        args.out_patch.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        save_file(output, str(args.out_patch))
        args.out_report.write_text(
            json.dumps(
                {
                    "partial": True,
                    "cache": str(args.cache),
                    "out_patch": str(args.out_patch),
                    "layers": selected_layers,
                    "completed_layers": [row["layer"] for row in reports],
                    "patch_reports": patch_reports,
                    "reports": reports,
                },
                indent=2,
            )
            + "\n"
        )
        torch.cuda.empty_cache()

    args.out_patch.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, str(args.out_patch))
    result = {
        "cache": str(args.cache),
        "out_patch": str(args.out_patch),
        "base_patches": [str(path) for path in args.base_patch or []],
        "patch_reports": patch_reports,
        "layers": selected_layers,
        "hidden_layers": hidden_layers,
        "mode": args.mode,
        "group_size": args.group_size,
        "steps_per_layer": args.steps_per_layer,
        "lr": args.lr,
        "format_boost": args.format_boost,
        "hidden_weight": args.hidden_weight,
        "anti_weight": args.anti_weight,
        "code_weight": args.code_weight,
        "train_rows": [
            {"index": row.get("index"), "task": row.get("task"), "id": row.get("id"), "tokens": int(row["tokens"]), "keep": int(row["keep"])}
            for row in rows
        ],
        "reports": reports,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out_patch": str(args.out_patch), "out_report": str(args.out_report), "layers": selected_layers}, indent=2))


if __name__ == "__main__":
    main()
