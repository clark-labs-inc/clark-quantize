#!/usr/bin/env python3
"""Measure Clark Air binary/ternary PTQ on browser-sized Swin2SR.

The experiment intentionally uses real Clark product screenshots rather than a
generic image benchmark. It reports reconstruction quality against both the
high-resolution source crop and the unquantized Swin2SR output, then writes a
real packed Clark artifact for every policy.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

from clark_air.quant.binary import quantize_binary
from clark_air.quant.optimal_ternary import quantize_ternary_mse
from clark_air.quant.packing import pack_bits, pack_trits_40


MODEL_ID = "caidas/swin2SR-lightweight-x2-64"
MODEL_REVISION = "07e8e61492ce6daf4f34c651622960faf110c17f"
PROTECTED_IMAGE_ISLANDS = {
    "swin2sr.first_convolution.weight",
    "swin2sr.conv_after_body.weight",
    "upsample.conv.weight",
}
DEFAULT_IMAGES = (
    ("/Users/stan/Documents/git/clark-desktop/promos/source-ui-composer-current.png", 0.54, 0.52),
    ("/Users/stan/Documents/git/clark-desktop/promos/source-ui-chat-current.png", 0.62, 0.56),
    ("/Users/stan/Documents/git/clark-desktop/promos/source-ui-research-current.png", 0.43, 0.49),
    (
        "/Users/stan/Documents/git/clark-desktop/promos/"
        "clark-code-ad-parallel-research-vertical-poster.png",
        0.51,
        0.45,
    ),
)


@dataclass(frozen=True)
class Policy:
    name: str
    mode: str | None
    group_size: int
    surface: str


POLICIES = (
    Policy("fp32", None, 0, "none"),
    Policy("linear_ternary_g32", "ternary", 32, "linear"),
    Policy("linear_ternary_g60", "ternary", 60, "linear"),
    Policy("linear_ternary_g120", "ternary", 120, "linear"),
    Policy("trunk_ternary_g60", "ternary", 60, "trunk"),
    Policy("all_ternary_g60", "ternary", 60, "all"),
    Policy("linear_binary_g60", "binary", 60, "linear"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/browser_swin2sr/out"),
    )
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--crop-size", type=int, default=128)
    return parser.parse_args()


def psnr(actual: np.ndarray, expected: np.ndarray) -> float:
    mse = float(np.mean((actual.astype(np.float32) - expected.astype(np.float32)) ** 2))
    return 99.0 if mse <= 1e-12 else 10 * math.log10(1.0 / mse)


def edge_psnr(actual: np.ndarray, expected: np.ndarray) -> float:
    def edges(image: np.ndarray) -> np.ndarray:
        dx = np.diff(image, axis=1, append=image[:, -1:, :])
        dy = np.diff(image, axis=0, append=image[-1:, :, :])
        return np.concatenate([dx, dy], axis=2)

    return psnr((edges(actual) + 1) / 2, (edges(expected) + 1) / 2)


def make_samples(crop_size: int) -> list[dict[str, object]]:
    samples = []
    for source_path, center_x, center_y in DEFAULT_IMAGES:
        source = Image.open(source_path).convert("RGB")
        left = max(0, min(source.width - crop_size, round(source.width * center_x - crop_size / 2)))
        top = max(0, min(source.height - crop_size, round(source.height * center_y - crop_size / 2)))
        high = source.crop((left, top, left + crop_size, top + crop_size))
        low = high.resize((crop_size // 2, crop_size // 2), Image.Resampling.BICUBIC)
        samples.append({"name": Path(source_path).stem, "high": high, "low": low})
    return samples


def reconstruct(model, processor, low: Image.Image, device: str, output_size: int) -> np.ndarray:
    inputs = processor(images=low, return_tensors="pt").pixel_values.to(device)
    with torch.inference_mode():
        output = model(inputs).reconstruction[:, :, :output_size, :output_size]
    return output.clamp(0, 1).float().cpu()[0].permute(1, 2, 0).numpy()


def quantized_weight_names(model, surface: str) -> set[str]:
    if surface == "none":
        return set()
    linear = {
        f"{name}.weight" if name else "weight"
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    if surface == "linear":
        return linear
    matrix_weights = {
        name
        for name, parameter in model.named_parameters()
        if name.endswith(".weight") and parameter.ndim >= 2
    }
    if surface == "trunk":
        return matrix_weights - PROTECTED_IMAGE_ISLANDS
    if surface == "all":
        return matrix_weights
    raise ValueError(f"unknown surface: {surface}")


def apply_policy(
    model,
    base_state: dict[str, torch.Tensor],
    policy: Policy,
    device: str,
) -> tuple[set[str], float]:
    model.load_state_dict(base_state)
    model.to(device).eval()
    selected = quantized_weight_names(model, policy.surface)
    if not selected:
        return selected, 0.0
    started = time.perf_counter()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in selected:
                continue
            if policy.mode == "ternary":
                quantized = quantize_ternary_mse(parameter, group_size=policy.group_size)
            else:
                quantized = quantize_binary(parameter, group_size=policy.group_size)
            parameter.copy_(quantized.dequantize().to(device=device, dtype=parameter.dtype))
    if device == "mps":
        torch.mps.synchronize()
    return selected, time.perf_counter() - started


def pack_artifact(
    path: Path,
    model,
    selected: set[str],
    policy: Policy,
) -> dict[str, int | float]:
    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, object] = {"format": "clark-air-v1", "policy": asdict(policy), "tensors": {}}
    selected_values = 0
    total_values = 0
    for index, (name, value) in enumerate(model.state_dict().items()):
        cpu = value.detach().float().cpu()
        key = f"t{index}"
        total_values += cpu.numel()
        if name in selected:
            selected_values += cpu.numel()
            if policy.mode == "ternary":
                quantized = quantize_ternary_mse(cpu, group_size=policy.group_size)
                packed, shape, pad = pack_trits_40(quantized.codes.numpy())
            else:
                quantized = quantize_binary(cpu, group_size=policy.group_size)
                packed, shape, pad = pack_bits(quantized.codes.numpy())
            arrays[f"{key}_codes"] = packed
            arrays[f"{key}_scales"] = quantized.scales.half().numpy()
            manifest["tensors"][name] = {
                "key": key,
                "kind": policy.mode,
                "shape": shape,
                "pad": pad,
                "group_size": policy.group_size,
            }
        else:
            if value.is_floating_point():
                stored = value.detach().half().cpu().numpy()
                dtype = "float16"
            else:
                stored = value.detach().cpu().numpy()
                dtype = str(stored.dtype)
            arrays[f"{key}_raw"] = stored
            manifest["tensors"][name] = {
                "key": key,
                "kind": "raw",
                "shape": list(value.shape),
                "dtype": dtype,
            }
    np.savez(path, **arrays)
    manifest_path = path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    size = path.stat().st_size + manifest_path.stat().st_size
    return {
        "artifact_bytes": size,
        "quantized_fraction": selected_values / max(1, total_values),
        "selected_values": selected_values,
        "total_values": total_values,
    }


def add_panel(canvas: Image.Image, image: np.ndarray, x: int, label: str) -> None:
    panel = Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255))
    canvas.paste(panel, (x, 34))
    ImageDraw.Draw(canvas).text((x + 4, 10), label, fill="white")


def write_montage(path: Path, sample, outputs: dict[str, np.ndarray], policies: list[Policy]) -> None:
    size = sample["high"].width
    columns = 2 + len(policies)
    montage = Image.new("RGB", (columns * size, size + 34), "#151515")
    high = np.asarray(sample["high"], dtype=np.float32) / 255
    bicubic = np.asarray(sample["low"].resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32) / 255
    add_panel(montage, high, 0, "ground truth")
    add_panel(montage, bicubic, size, "bicubic")
    for index, policy in enumerate(policies):
        add_panel(montage, outputs[policy.name][0], (index + 2) * size, policy.name)
    montage.save(path)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    samples = make_samples(args.crop_size)
    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=False
    )
    model = Swin2SRForImageSuperResolution.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    ).eval()
    base_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    results: dict[str, object] = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": args.device,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "fp32_parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
        "samples": [sample["name"] for sample in samples],
        "policies": {},
    }
    outputs: dict[str, list[np.ndarray]] = {}
    baseline_outputs: list[np.ndarray] | None = None

    for policy in POLICIES:
        selected, quantize_seconds = apply_policy(model, base_state, policy, args.device)
        started = time.perf_counter()
        policy_outputs = [
            reconstruct(model, processor, sample["low"], args.device, args.crop_size)
            for sample in samples
        ]
        if args.device == "mps":
            torch.mps.synchronize()
        inference_seconds = time.perf_counter() - started
        if baseline_outputs is None:
            baseline_outputs = [output.copy() for output in policy_outputs]
        ground_truth = [
            np.asarray(sample["high"], dtype=np.float32) / 255
            for sample in samples
        ]
        metrics = {
            "psnr_vs_ground_truth": float(np.mean([
                psnr(output, truth) for output, truth in zip(policy_outputs, ground_truth)
            ])),
            "edge_psnr_vs_ground_truth": float(np.mean([
                edge_psnr(output, truth) for output, truth in zip(policy_outputs, ground_truth)
            ])),
            "psnr_vs_fp32": float(np.mean([
                psnr(output, baseline) for output, baseline in zip(policy_outputs, baseline_outputs)
            ])),
        }
        artifact = pack_artifact(args.out / f"{policy.name}.npz", model, selected, policy)
        results["policies"][policy.name] = {
            **asdict(policy),
            **metrics,
            **artifact,
            "quantize_seconds": quantize_seconds,
            "inference_seconds": inference_seconds,
            "selected_tensors": len(selected),
        }
        outputs[policy.name] = policy_outputs
        print(policy.name, json.dumps(results["policies"][policy.name], sort_keys=True))

    write_montage(args.out / "comparison.png", samples[0], outputs, list(POLICIES))
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(args.out / "results.json")


if __name__ == "__main__":
    main()
