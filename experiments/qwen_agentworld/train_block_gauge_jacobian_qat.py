"""Gauge-Jacobian block QAT for one-GPU Qwen compression.

This tests the conjugacy hypothesis:

    G_{l+1} S_l(h_l^Q) ~= T_l(G_l h_l^Q)

with a Jacobian-vector loss:

    G_{l+1} J_S v ~= J_T G_l v

The gauges are diagonal affine maps fitted from cached student prefixes and a
fresh BF16 teacher pass. Only one active block owns trainable low-bit masters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clark_quantize.bonsai_qat8 import BonsaiQAT8
from collect_onpolicy_survival_cache import load_model, parse_layers
from train_all2d_ternary_qat import alpha_quantized_weight, get_parent_and_attr
from weight_patch import apply_weight_patch


def layer_name(layer_id: int) -> str:
    return f"model.language_model.layers.{layer_id}"


def layer_module(model: torch.nn.Module, layer_id: int) -> torch.nn.Module:
    return model.get_submodule(layer_name(layer_id))


class TrainQuantLinear(torch.nn.Module):
    def __init__(
        self,
        source: torch.nn.Linear,
        *,
        mode: str,
        group_size: int,
        init_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        weight = source.weight if init_weight is None else init_weight
        self.master = torch.nn.Parameter(weight.detach().float().clone())
        self.mode = mode
        self.group_size = group_size
        if source.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", source.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = alpha_quantized_weight(self.master, mode=self.mode, group_size=self.group_size, alpha=1.0)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight.to(dtype=x.dtype), bias)

    def codebook_loss(self) -> torch.Tensor:
        with torch.no_grad():
            quantized = alpha_quantized_weight(self.master, mode=self.mode, group_size=self.group_size, alpha=1.0).detach()
        denom = self.master.detach().float().square().mean().clamp_min(1e-8)
        return (self.master.float() - quantized.float()).square().mean() / denom

    @torch.no_grad()
    def hard_weight(self) -> torch.Tensor:
        return alpha_quantized_weight(self.master, mode=self.mode, group_size=self.group_size, alpha=1.0).cpu().to(torch.bfloat16)


def module_quant_mode(name: str, args: argparse.Namespace) -> str:
    if ".mlp." in name and args.mlp_mode is not None:
        return args.mlp_mode
    if ".mlp." not in name and args.attention_mode is not None:
        return args.attention_mode
    return args.mode


def replace_layer_linears(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    layer_id: int,
    *,
    args: argparse.Namespace,
) -> dict[str, TrainQuantLinear]:
    prefix = layer_name(layer_id)
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if name.startswith(prefix + ".") and isinstance(module, torch.nn.Linear)
    ]
    out: dict[str, TrainQuantLinear] = {}
    for name, module in targets:
        mode = module_quant_mode(name, args)
        init_weight = None
        if args.init_mlp_from_teacher and ".mlp." in name:
            init_weight = teacher.get_submodule(name).weight
        quant = TrainQuantLinear(
            module,
            mode=mode,
            group_size=args.group_size,
            init_weight=init_weight,
        ).to(next(module.parameters()).device)
        parent, attr = get_parent_and_attr(model, name)
        setattr(parent, attr, quant)
        out[name] = quant
    return out


def select_layer_norms(model: torch.nn.Module, layer_id: int) -> dict[str, torch.nn.Parameter]:
    prefix = layer_name(layer_id)
    selected: dict[str, torch.nn.Parameter] = {}
    for name, param in model.named_parameters():
        if name.startswith(prefix + ".") and name.endswith(".weight") and param.ndim == 1:
            param.data = param.data.float()
            param.requires_grad_(True)
            selected[name] = param
    return selected


@torch.no_grad()
def fit_diag_affine_gauge(student: torch.Tensor, teacher: torch.Tensor, *, min_scale: float, max_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    xs = student.float().reshape(-1, student.shape[-1])
    yt = teacher.float().reshape(-1, teacher.shape[-1])
    mean_x = xs.mean(dim=0)
    mean_y = yt.mean(dim=0)
    xc = xs - mean_x
    yc = yt - mean_y
    var_x = (xc * xc).mean(dim=0).clamp_min(1e-6)
    cov_xy = (xc * yc).mean(dim=0)
    scale = (cov_xy / var_x).clamp(min_scale, max_scale)
    bias = mean_y - scale * mean_x
    return scale.detach(), bias.detach()


def apply_gauge(hidden: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return hidden.float() * scale.view(1, 1, -1) + bias.view(1, 1, -1)


def call_block(block: torch.nn.Module, hidden: torch.Tensor, item: dict[str, Any]) -> torch.Tensor:
    position_ids = item.get("position_ids")
    if position_ids is not None and position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0).to(hidden.device)
    cos = item["cos"].unsqueeze(0).to(hidden.device)
    sin = item["sin"].unsqueeze(0).to(hidden.device)
    out = block(
        hidden,
        position_embeddings=(cos, sin),
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
    )
    return out[0] if isinstance(out, tuple) else out


def capture_teacher_layer_io(
    teacher: torch.nn.Module,
    input_ids: torch.Tensor,
    layer_id: int,
    *,
    keep: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    layer = layer_module(teacher, layer_id)

    def pre_hook(_module: torch.nn.Module, inputs: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        captured["in"] = inputs[0].detach()[:, -keep:, :].cpu().to(torch.bfloat16).squeeze(0).contiguous()

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], _kwargs: dict[str, Any], output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured["out"] = hidden.detach()[:, -keep:, :].cpu().to(torch.bfloat16).squeeze(0).contiguous()

    h1 = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = layer.register_forward_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            ids = input_ids.unsqueeze(0).to(device=device, dtype=torch.long)
            teacher(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
    finally:
        h1.remove()
        h2.remove()
    if "in" not in captured or "out" not in captured:
        raise RuntimeError(f"failed to capture teacher layer {layer_id}")
    return captured["in"], captured["out"]


def fit_gauges_for_layer(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    rows: list[dict[str, Any]],
    blocks: dict[int, dict[int, dict[str, Any]]],
    layer_id: int,
    args: argparse.Namespace,
    device: torch.device,
    teacher_device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], dict[str, float]]:
    student_inputs = []
    student_outputs = []
    teacher_inputs = []
    teacher_outputs = []
    block = layer_module(student, layer_id)
    for row_index, row in enumerate(rows):
        keep = int(row["keep"])
        item = blocks[row_index][layer_id]
        h_q = item["h_in"].unsqueeze(0).to(device)
        with torch.no_grad():
            y_q = call_block(block, h_q, item)
        t_in, t_out = capture_teacher_layer_io(
            teacher,
            row["input_ids"].to(torch.long),
            layer_id,
            keep=keep,
            device=teacher_device,
        )
        student_inputs.append(h_q[:, -keep:, :].detach().cpu().to(torch.bfloat16).squeeze(0))
        student_outputs.append(y_q[:, -keep:, :].detach().cpu().to(torch.bfloat16).squeeze(0))
        teacher_inputs.append(t_in)
        teacher_outputs.append(t_out)
    gauge_in = fit_diag_affine_gauge(
        torch.cat(student_inputs, dim=0),
        torch.cat(teacher_inputs, dim=0),
        min_scale=args.gauge_min,
        max_scale=args.gauge_max,
    )
    gauge_out = fit_diag_affine_gauge(
        torch.cat(student_outputs, dim=0),
        torch.cat(teacher_outputs, dim=0),
        min_scale=args.gauge_min,
        max_scale=args.gauge_max,
    )
    stats = {
        "gauge_in_scale_mean": float(gauge_in[0].mean().cpu()),
        "gauge_in_scale_std": float(gauge_in[0].std().cpu()),
        "gauge_out_scale_mean": float(gauge_out[0].mean().cpu()),
        "gauge_out_scale_std": float(gauge_out[0].std().cpu()),
    }
    return gauge_in, gauge_out, stats


def jvp_block(block: torch.nn.Module, hidden: torch.Tensor, tangent: torch.Tensor, item: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return call_block(block, x, item)

    return torch.autograd.functional.jvp(fn, (hidden,), (tangent,), create_graph=True, strict=False)


def train_one_layer(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    cache: dict[str, Any],
    layer_id: int,
    args: argparse.Namespace,
    device: torch.device,
    teacher_device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    rows = [
        row
        for row in cache["rows"]
        if int(row["tokens"]) <= args.max_train_tokens
    ]
    if args.max_train_rows is not None:
        rows = rows[: args.max_train_rows]
    if not rows:
        raise RuntimeError("no rows selected")
    blocks = cache["blocks"]
    gauge_in, gauge_out, gauge_stats = fit_gauges_for_layer(student, teacher, rows, blocks, layer_id, args, device, teacher_device)
    gauge_in = (gauge_in[0].to(device), gauge_in[1].to(device))
    gauge_out = (gauge_out[0].to(device), gauge_out[1].to(device))

    quant_linears = replace_layer_linears(student, teacher, layer_id, args=args)
    norm_params = select_layer_norms(student, layer_id) if args.train_norms else {}
    params: list[torch.nn.Parameter] = [quant.master for quant in quant_linears.values()]
    params.extend(norm_params.values())
    matrix_params = [param for param in params if param.ndim == 2]
    other_params = [param for param in params if param.ndim != 2]
    groups: list[dict[str, Any]] = []
    if matrix_params:
        groups.append(
            {
                "params": matrix_params,
                "role": "matrix_qat8",
                "lr": args.lr,
                "betas": (0.9, 0.95),
                "use_muon": args.use_muon,
                "muon_every": args.muon_every,
                "muon_steps": args.muon_steps,
                "muon_max_elements": args.muon_max_elements,
                "muon_rms_match": args.muon_rms_match,
                "prox_rho": args.prox_rho,
                "max_update_rms": args.max_update_rms,
            }
        )
    if other_params:
        groups.append({"params": other_params, "role": "adamw", "lr": args.lr, "betas": (0.9, 0.95)})
    optimizer = BonsaiQAT8(groups)
    active = layer_module(student, layer_id)
    teacher_block = layer_module(teacher, layer_id)
    history = []
    for step in range(1, args.steps_per_layer + 1):
        row_index = (step - 1) % len(rows)
        row = rows[row_index]
        item = blocks[row_index][layer_id]
        keep = int(row["keep"])
        h_q = item["h_in"].unsqueeze(0).to(device)
        h_q = h_q.detach()
        h_t = apply_gauge(h_q, *gauge_in).to(device=device, dtype=h_q.dtype)
        with torch.no_grad():
            y_t = call_block(teacher_block, h_t.to(teacher_device), item).to(device)
        if args.jvp_weight > 0:
            v = torch.empty_like(h_q).bernoulli_(0.5).mul_(2).sub_(1)
            v = v / v.float().square().mean().sqrt().clamp_min(1e-8)
            y_q, j_q = jvp_block(active, h_q.detach().requires_grad_(True), v, item)
            v_t = (v.float() * gauge_in[0].view(1, 1, -1)).to(device=teacher_device, dtype=h_t.dtype)
            y_t_jvp, j_t = torch.autograd.functional.jvp(
                lambda x: call_block(teacher_block, x, item),
                (h_t.detach().to(teacher_device).requires_grad_(True),),
                (v_t,),
                create_graph=False,
                strict=False,
            )
            y_t = y_t_jvp.to(device)
            j_t = j_t.to(device)
            j_q_g = j_q.float() * gauge_out[0].view(1, 1, -1)
            jvp_loss = (
                (j_q_g[:, -keep:, :] - j_t[:, -keep:, :].float()).square().mean()
                / j_t[:, -keep:, :].float().square().mean().clamp_min(1e-8)
            )
        else:
            y_q = call_block(active, h_q, item)
            jvp_loss = y_q.sum() * 0.0
        y_q_g = apply_gauge(y_q, *gauge_out)
        value_loss = (
            (y_q_g[:, -keep:, :] - y_t[:, -keep:, :].float()).square().mean()
            / y_t[:, -keep:, :].float().square().mean().clamp_min(1e-8)
        )
        code_terms = [quant.codebook_loss() for quant in quant_linears.values()]
        code_loss = torch.stack(code_terms).mean() if code_terms else value_loss.detach().new_zeros(())
        loss = value_loss + args.jvp_weight * jvp_loss + args.code_weight * code_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps_per_layer:
            record = {
                "layer": layer_id,
                "step": step,
                "task": row.get("task"),
                "id": row.get("id"),
                "tokens": int(row["tokens"]),
                "loss": float(loss.detach().cpu()),
                "value": float(value_loss.detach().cpu()),
                "jvp": float(jvp_loss.detach().cpu()),
                "code": float(code_loss.detach().cpu()),
                **gauge_stats,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
        del y_q, y_t, y_q_g, loss, value_loss, jvp_loss, code_loss
        torch.cuda.empty_cache()

    patch: dict[str, torch.Tensor] = {}
    for name, quant in quant_linears.items():
        patch[f"{name}.weight"] = quant.hard_weight().contiguous()
    for name, param in norm_params.items():
        patch[name] = param.detach().cpu().to(torch.bfloat16).contiguous()
    quant_modes = {name: quant.mode for name, quant in quant_linears.items()}
    return patch, {
        "layer": layer_id,
        "linears": sorted(quant_linears),
        "quant_modes": quant_modes,
        "norms": sorted(norm_params),
        "gauge": gauge_stats,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models/Qwen3.5-9B", type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--base-patch", action="append", type=Path)
    parser.add_argument("--out-patch", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--mode", choices=("ternary", "binary"), default="ternary")
    parser.add_argument("--mlp-mode", choices=("ternary", "binary"))
    parser.add_argument("--attention-mode", choices=("ternary", "binary"))
    parser.add_argument("--init-mlp-from-teacher", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--steps-per-layer", type=int, default=100)
    parser.add_argument("--max-train-tokens", type=int, default=5000)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--jvp-weight", type=float, default=0.1)
    parser.add_argument("--code-weight", type=float, default=0.002)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gauge-min", type=float, default=0.1)
    parser.add_argument("--gauge-max", type=float, default=10.0)
    parser.add_argument("--use-muon", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--muon-every", type=int, default=16)
    parser.add_argument("--muon-steps", type=int, default=2)
    parser.add_argument("--muon-max-elements", type=int, default=80_000_000)
    parser.add_argument("--muon-rms-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prox-rho", type=float, default=0.0)
    parser.add_argument("--max-update-rms", type=float, default=0.0)
    parser.add_argument("--train-norms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("format") != "qwen_onpolicy_block_cache_v1":
        raise ValueError(f"unsupported cache format: {cache.get('format')!r}")
    layers = parse_layers(args.layers)
    cache["blocks"] = {int(row_index): {int(layer): item for layer, item in layer_items.items()} for row_index, layer_items in cache["blocks"].items()}

    student = load_model(args.model_dir)
    student.eval()
    for param in student.parameters():
        param.requires_grad_(False)
    patch_reports = [apply_weight_patch(student, patch) for patch in args.base_patch or []]
    teacher = load_model(args.model_dir)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    device = next(student.parameters()).device
    teacher_device = next(teacher.parameters()).device

    output: dict[str, torch.Tensor] = {}
    reports = []
    args.out_patch.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    for layer_id in layers:
        patch, report = train_one_layer(student, teacher, cache, layer_id, args, device, teacher_device)
        output.update(patch)
        reports.append(report)
        save_file(output, str(args.out_patch))
        args.out_report.write_text(json.dumps({"partial": True, "completed_layers": [r["layer"] for r in reports], "reports": reports}, indent=2) + "\n")
        torch.cuda.empty_cache()

    result = {
        "cache": str(args.cache),
        "base_patches": [str(path) for path in args.base_patch or []],
        "patch_reports": patch_reports,
        "out_patch": str(args.out_patch),
        "layers": layers,
        "mode": args.mode,
        "mlp_mode": args.mlp_mode,
        "attention_mode": args.attention_mode,
        "init_mlp_from_teacher": args.init_mlp_from_teacher,
        "group_size": args.group_size,
        "steps_per_layer": args.steps_per_layer,
        "jvp_weight": args.jvp_weight,
        "code_weight": args.code_weight,
        "use_muon": args.use_muon,
        "reports": reports,
    }
    save_file(output, str(args.out_patch))
    args.out_report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out_patch": str(args.out_patch), "layers": layers}, indent=2))


if __name__ == "__main__":
    main()
