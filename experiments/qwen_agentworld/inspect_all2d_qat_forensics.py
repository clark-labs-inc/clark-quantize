"""Inspect codebook movement in all-2D QAT checkpoints.

This is intentionally offline and read-only: it compares a saved BF16 master
weight patch against the original BF16 model, then derives binary/ternary codes
with the same groupwise quantizer used by training.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clark_air.quant.optimal_ternary import quantize_ternary_mse


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def load_weight_map(model_dir: Path) -> dict[str, Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        body = json.loads(index.read_text())
        return {name: model_dir / shard for name, shard in body["weight_map"].items()}

    result: dict[str, Path] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                result[key] = shard
    if not result:
        raise FileNotFoundError(f"no safetensors found under {model_dir}")
    return result


def tensor_family(name: str) -> str:
    if name == "lm_head.weight" or name.endswith(".lm_head.weight"):
        return "lm_head"
    if name.endswith("embed_tokens.weight"):
        return "embedding"
    if ".mlp.gate_proj." in name:
        return "mlp.gate"
    if ".mlp.up_proj." in name:
        return "mlp.up"
    if ".mlp.down_proj." in name:
        return "mlp.down"
    if ".self_attn.q_proj." in name:
        return "attn.q"
    if ".self_attn.k_proj." in name:
        return "attn.k"
    if ".self_attn.v_proj." in name:
        return "attn.v"
    if ".self_attn.o_proj." in name:
        return "attn.o"
    if ".linear_attn.in_proj_qkv." in name:
        return "linear_attn.qkv"
    if ".linear_attn.in_proj_z." in name:
        return "linear_attn.z"
    if ".linear_attn.out_proj." in name:
        return "linear_attn.out"
    if ".linear_attn.in_proj_a." in name:
        return "linear_attn.a"
    if ".linear_attn.in_proj_b." in name:
        return "linear_attn.b"
    if ".norm." in name or name.endswith("layernorm.weight"):
        return "norm"
    return "other"


def module_class(name: str) -> str:
    if name == "lm_head.weight" or name.endswith(".lm_head.weight"):
        return "head"
    if name.endswith("embed_tokens.weight"):
        return "embed"
    if ".mlp.gate_proj." in name:
        return "mlp_gate"
    if ".mlp.up_proj." in name:
        return "mlp_up"
    if ".mlp.down_proj." in name:
        return "mlp_down"
    if any(part in name for part in (".self_attn.q_proj.", ".self_attn.k_proj.", ".self_attn.v_proj.", ".self_attn.o_proj.")):
        return "attn"
    if ".self_attn." in name or ".attention." in name:
        return "attn_other"
    if ".mlp." in name:
        return "mlp_other"
    return "other_2d"


def layer_id(name: str) -> int | None:
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def empty_acc() -> dict[str, Any]:
    return {
        "tensors": 0,
        "numel": 0,
        "base_zero": 0,
        "current_zero": 0,
        "support_added": 0,
        "support_dropped": 0,
        "active_sign_flips": 0,
        "code_changed": 0,
        "master_sign_flips": 0,
        "delta_sq": 0.0,
        "base_sq": 0.0,
        "scale_ratio_sum": 0.0,
        "scale_ratio_sq_sum": 0.0,
        "scale_groups": 0,
    }


def add_acc(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if key == "tensors":
            dst[key] += value
        elif isinstance(value, int):
            dst[key] += value
        elif isinstance(value, float):
            dst[key] += value


def finalize_acc(acc: dict[str, Any]) -> dict[str, Any]:
    numel = max(int(acc["numel"]), 1)
    groups = max(int(acc["scale_groups"]), 1)
    ratio_mean = float(acc["scale_ratio_sum"]) / groups
    ratio_var = max(float(acc["scale_ratio_sq_sum"]) / groups - ratio_mean * ratio_mean, 0.0)
    support_added_frac = float(acc["support_added"]) / numel
    support_dropped_frac = float(acc["support_dropped"]) / numel
    return {
        "tensors": int(acc["tensors"]),
        "numel": int(acc["numel"]),
        "base_zero_frac": float(acc["base_zero"]) / numel,
        "current_zero_frac": float(acc["current_zero"]) / numel,
        "support_added_frac": support_added_frac,
        "support_dropped_frac": support_dropped_frac,
        "support_net_frac": support_added_frac - support_dropped_frac,
        "support_asymmetry_frac": (
            (support_added_frac - support_dropped_frac)
            / max(support_added_frac + support_dropped_frac, 1e-20)
        ),
        "active_sign_flip_frac": float(acc["active_sign_flips"]) / numel,
        "code_changed_frac": float(acc["code_changed"]) / numel,
        "master_sign_flip_frac": float(acc["master_sign_flips"]) / numel,
        "master_delta_rms": math.sqrt(float(acc["delta_sq"]) / numel),
        "relative_master_delta_rms": math.sqrt(float(acc["delta_sq"]) / max(float(acc["base_sq"]), 1e-20)),
        "scale_ratio_mean": ratio_mean,
        "scale_ratio_std": math.sqrt(ratio_var),
        "scale_groups": int(acc["scale_groups"]),
    }


def add_diffusion_ratios(report: dict[str, Any], expected_rms: float) -> None:
    if expected_rms <= 0:
        return
    for section in ("global",):
        if section in report:
            report[section]["master_delta_diffusion_ratio"] = report[section]["master_delta_rms"] / expected_rms
    for section in ("by_family", "by_module_class", "by_layer"):
        for item in report.get(section, {}).values():
            item["master_delta_diffusion_ratio"] = item["master_delta_rms"] / expected_rms
    for item in report.get("tensors", []):
        item["master_delta_diffusion_ratio"] = item["master_delta_rms"] / expected_rms


def expected_diffusion_rms(
    *,
    steps: int,
    lr: float,
    max_update_rms: float,
    warmup_steps: int,
) -> float:
    if steps <= 0 or lr <= 0 or max_update_rms <= 0:
        return 0.0
    total_sq = 0.0
    for step in range(1, steps + 1):
        if warmup_steps > 0:
            scale = min(1.0, step / warmup_steps)
        else:
            scale = 1.0
        step_rms = lr * max_update_rms * scale
        total_sq += step_rms * step_rms
    return math.sqrt(total_sq)


@torch.no_grad()
def compare_tensor(
    *,
    name: str,
    base: torch.Tensor,
    current: torch.Tensor,
    group_size: int,
    chunk_groups: int,
    sample_per_tensor: int,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    if base.shape != current.shape:
        raise ValueError(f"shape mismatch for {name}: {tuple(base.shape)} vs {tuple(current.shape)}")
    if base.ndim < 2:
        return empty_acc(), []

    flat_base = base.detach().reshape(-1)
    flat_current = current.detach().reshape(-1)
    chunk_elems = max(chunk_groups, 1) * group_size
    acc = empty_acc()
    acc["tensors"] = 1
    samples: list[dict[str, float]] = []

    for start in range(0, flat_base.numel(), chunk_elems):
        end = min(start + chunk_elems, flat_base.numel())
        b = flat_base[start:end].float()
        c = flat_current[start:end].float()
        qb = quantize_ternary_mse(b, group_size=group_size)
        qc = quantize_ternary_mse(c, group_size=group_size)
        codes_b = qb.codes
        codes_c = qc.codes
        if qb.pad:
            codes_b = codes_b[:-qb.pad]
        if qc.pad:
            codes_c = codes_c[:-qc.pad]

        active_b = codes_b != 0
        active_c = codes_c != 0
        both_active = active_b & active_c
        acc["numel"] += int(codes_b.numel())
        acc["base_zero"] += int((~active_b).sum().item())
        acc["current_zero"] += int((~active_c).sum().item())
        acc["support_added"] += int((~active_b & active_c).sum().item())
        acc["support_dropped"] += int((active_b & ~active_c).sum().item())
        acc["active_sign_flips"] += int((both_active & (codes_b != codes_c)).sum().item())
        acc["code_changed"] += int((codes_b != codes_c).sum().item())
        acc["master_sign_flips"] += int(((b.sign() != c.sign()) & (b != 0) & (c != 0)).sum().item())
        acc["delta_sq"] += float((c - b).square().sum().item())
        acc["base_sq"] += float(b.square().sum().item())

        ratios = (qc.scales.float() / qb.scales.float().clamp_min(1e-12)).reshape(-1)
        acc["scale_ratio_sum"] += float(ratios.sum().item())
        acc["scale_ratio_sq_sum"] += float(ratios.square().sum().item())
        acc["scale_groups"] += int(ratios.numel())

    if sample_per_tensor > 0 and flat_base.numel() > 0:
        count = min(sample_per_tensor, flat_base.numel())
        idx = torch.linspace(0, flat_base.numel() - 1, count, dtype=torch.long)
        b = flat_base[idx].float()
        c = flat_current[idx].float()
        for abs_value, flipped in zip(b.abs().tolist(), ((b.sign() != c.sign()) & (b != 0) & (c != 0)).tolist(), strict=False):
            samples.append({"abs_base": float(abs_value), "master_sign_flip": float(flipped)})

    return acc, samples


def decile_report(samples: list[dict[str, float]]) -> list[dict[str, float]]:
    if not samples:
        return []
    ordered = sorted(samples, key=lambda item: item["abs_base"])
    result = []
    total = len(ordered)
    for decile in range(10):
        start = total * decile // 10
        end = total * (decile + 1) // 10
        bucket = ordered[start:end]
        if not bucket:
            continue
        result.append(
            {
                "decile": decile,
                "count": len(bucket),
                "abs_min": bucket[0]["abs_base"],
                "abs_max": bucket[-1]["abs_base"],
                "master_sign_flip_frac": sum(item["master_sign_flip"] for item in bucket) / len(bucket),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--patch", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--chunk-groups", type=int, default=8192)
    parser.add_argument("--sample-per-tensor", type=int, default=2048)
    parser.add_argument("--include-regex")
    parser.add_argument("--max-tensors", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--max-update-rms", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--expected-diffusion-rms",
        type=float,
        default=0.0,
        help="override diffusion RMS denominator; otherwise computed from steps/lr/max-update-rms/warmup-steps",
    )
    args = parser.parse_args()

    include_re = re.compile(args.include_regex) if args.include_regex else None
    weight_map = load_weight_map(args.model_dir)

    by_family: dict[str, dict[str, Any]] = defaultdict(empty_acc)
    by_module_class: dict[str, dict[str, Any]] = defaultdict(empty_acc)
    by_layer: dict[str, dict[str, Any]] = defaultdict(empty_acc)
    global_acc = empty_acc()
    tensor_rows = []
    samples: list[dict[str, float]] = []
    processed = 0

    with safe_open(args.patch, framework="pt", device="cpu") as patch_handle:
        keys = sorted(key for key in patch_handle.keys() if key.endswith(".weight"))
        for name in keys:
            if include_re and not include_re.search(name):
                continue
            if name not in weight_map:
                continue
            if args.max_tensors and processed >= args.max_tensors:
                break

            current = patch_handle.get_tensor(name)
            if current.ndim < 2:
                continue
            with safe_open(weight_map[name], framework="pt", device="cpu") as base_handle:
                base = base_handle.get_tensor(name)

            acc, tensor_samples = compare_tensor(
                name=name,
                base=base,
                current=current,
                group_size=args.group_size,
                chunk_groups=args.chunk_groups,
                sample_per_tensor=args.sample_per_tensor,
            )
            processed += 1
            family = tensor_family(name)
            cls = module_class(name)
            layer = layer_id(name)
            add_acc(by_family[family], acc)
            add_acc(by_module_class[cls], acc)
            add_acc(global_acc, acc)
            if layer is not None:
                add_acc(by_layer[str(layer)], acc)
            samples.extend(tensor_samples)
            row = finalize_acc(acc)
            row["name"] = name
            row["family"] = family
            row["module_class"] = cls
            row["layer"] = layer
            tensor_rows.append(row)
            print(json.dumps({"tensor": name, "family": family, "module_class": cls, "layer": layer, "stats": row}), flush=True)

    expected_rms = args.expected_diffusion_rms or expected_diffusion_rms(
        steps=args.steps,
        lr=args.lr,
        max_update_rms=args.max_update_rms,
        warmup_steps=args.warmup_steps,
    )
    report = {
        "model_dir": str(args.model_dir),
        "patch": str(args.patch),
        "group_size": args.group_size,
        "expected_diffusion_rms": expected_rms,
        "expected_diffusion_rms_config": {
            "steps": args.steps,
            "lr": args.lr,
            "max_update_rms": args.max_update_rms,
            "warmup_steps": args.warmup_steps,
            "override": args.expected_diffusion_rms,
        },
        "processed_tensors": processed,
        "global": finalize_acc(global_acc),
        "by_family": {key: finalize_acc(value) for key, value in sorted(by_family.items())},
        "by_module_class": {key: finalize_acc(value) for key, value in sorted(by_module_class.items())},
        "by_layer": {key: finalize_acc(value) for key, value in sorted(by_layer.items(), key=lambda item: int(item[0]))},
        "sampled_master_sign_flip_by_abs_decile": decile_report(samples),
        "tensors": tensor_rows,
    }
    add_diffusion_ratios(report, expected_rms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"wrote": str(args.out), "processed_tensors": processed}), flush=True)


if __name__ == "__main__":
    main()
