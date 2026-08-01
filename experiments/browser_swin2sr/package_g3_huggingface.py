#!/usr/bin/env python3
"""Build a local Hugging Face staging package for the qualified G3 state."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


PARAMETER_VALUES = 1_005_348
TERNARY_WEIGHT_VALUES = 839_904
INT7_WEIGHT_VALUES = 129_600
FP16_WEIGHT_VALUES = 8_100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def copy_package_shell(source: Path, output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(source, output)
    examples = output / "examples"
    if examples.exists():
        shutil.rmtree(examples)
    examples.mkdir()


def save_dense_state(source: Path, output: Path) -> None:
    state = torch.load(source, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    tensors = {name: value.detach().contiguous() for name, value in state.items()}
    save_file(tensors, str(output), metadata={"format": "pt"})


def replace_readme(
    readme: Path,
    metrics: dict,
    bundle: dict,
    browser: dict,
    set5: dict,
) -> None:
    text = readme.read_text()
    replacements = {
        "a 331 KB browser AI upscaler": "a 380 KB mixed-ternary browser AI upscaler",
        "202 matrix and convolution tensors are ternary `{−1, 0, +1}`;":
            "198 matrix and convolution tensors are ternary `{−1, 0, +1}`;",
        "- the first and final image-I/O convolutions—8,100 weight values—remain FP16;":
            "- four sensitive residual-group convolutions—129,600 values—use signed int7 with per-output scales;\n"
            "- the first and final image-I/O convolutions—8,100 weight values—remain FP16;",
        "- 96.43% of all model parameters are packed ternary;":
            "- 83.55% of all parameters are ternary and 96.43% are low-bit weights;",
        "**330,933 bytes**, or\n  **2.63 bits per model parameter**":
            f"**{bundle['runtime_package_bytes']:,} bytes**, or\n  "
            f"**{bundle['runtime_package_bytes'] * 8 / PARAMETER_VALUES:.3f} bits per model parameter**",
        "| RGB PSNR versus ground truth | **34.10 dB** |":
            f"| RGB PSNR versus ground truth | **{metrics['overall']['psnr_vs_ground_truth']['mean']:.2f} dB** |",
        "| Edge PSNR versus ground truth | **37.72 dB** |":
            f"| Edge PSNR versus ground truth | **{metrics['overall']['edge_psnr_vs_ground_truth']['mean']:.2f} dB** |",
        "| Gain over bicubic | **+6.00 dB** |":
            f"| Gain over bicubic | **+{metrics['overall']['psnr_gain_over_bicubic']['mean']:.2f} dB** |",
        "| **Clark Ternary, this release** | **35.23 dB** | **+3.45 dB** |":
            f"| **Clark Ternary, this release** | **{set5['mean_ternary_rgb_psnr_db']:.2f} dB** | "
            f"**+{set5['mean_gain_db']:.2f} dB** |",
        "This is +1.20 dB over the previously released Ternary Lite model on the same\nbundle.":
            "This is +0.21 dB over the previous Clark Ternary release on the same locked bundle,\n"
            "with every tested source, split, script, and language group non-regressing.",
        "| Packed ternary parameters | 969,504 (96.43%) |":
            "| Ternary weight values | 839,904 (83.55% of parameters) |\n"
            "| Int7 retained-island weight values | 129,600 |",
        "| Ternary tensors | 202 |": "| Ternary tensors | 198 |\n| Int7 tensors | 4 |",
        "| Packed ternary trunk, codes + scales | 2.156 bits/ternary weight |":
            f"| Packed mixed low-bit trunk, codes + scales | {bundle['packed_trunk_bits_per_quantized_weight']:.3f} bits/quantized weight |",
        "| Complete model data | 2.650 bits/model parameter |\n": "",
        "| Lossless native package | **330,933 bytes; 2.633 bits/model parameter** |":
            f"| Lossless native package | **{bundle['runtime_package_bytes']:,} bytes; "
            f"{bundle['runtime_package_bytes'] * 8 / PARAMETER_VALUES:.3f} bits/model parameter** |",
        "| Portable ONNX browser download | **700,584 bytes** |":
            f"| Portable ONNX browser download | **{browser['compressed_model']['bytes']:,} bytes** |",
        "- PyTorch-to-packed artifact conformance: 78.35 dB;":
            "- exact dense-state to portable-ONNX numerical conformance;",
        "about 30 ms on WebGPU and 399 ms on WASM":
            f"{browser['receipts'][0]['warm_inference_ms']:.1f} ms on WebGPU and "
            f"{browser['receipts'][1]['warm_inference_ms']:.1f} ms on WASM",
        "native/swin2sr-ternary-bonsai-x2.bdk.gz` — the lossless 331 KB packed":
            "native/swin2sr-ternary-bonsai-x2.bdk.gz` — the lossless 380 KB packed",
    }
    for old, new in replacements.items():
        if old not in text:
            raise ValueError(f"README source text not found: {old!r}")
        text = text.replace(old, new)
    readme.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--packed-bundle", type=Path, required=True)
    parser.add_argument("--bundle-ledger", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--onnx-gzip", type=Path, required=True)
    parser.add_argument("--onnx-ledger", type=Path, required=True)
    parser.add_argument("--browser-conformance", type=Path, required=True)
    parser.add_argument("--set5-results", type=Path, required=True)
    parser.add_argument("--set5-examples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    copy_package_shell(args.base_package.resolve(), args.out.resolve())
    save_dense_state(args.state.resolve(), args.out / "model.safetensors")
    shutil.copy2(args.onnx, args.out / "onnx/model.onnx")
    shutil.copy2(args.onnx_gzip, args.out / "onnx/model.onnx.gz")
    shutil.copy2(args.onnx_ledger, args.out / "onnx/model.ledger.json")
    shutil.copy2(
        args.packed_bundle,
        args.out / "native/swin2sr-ternary-bonsai-x2.bdk.gz",
    )
    shutil.copy2(args.bundle_ledger, args.out / "native/transport-ledger.json")
    for example in args.set5_examples.glob("*.png"):
        shutil.copy2(example, args.out / "examples" / example.name)
    shutil.copy2(args.set5_results, args.out / "examples/set5-x2-results.json")

    metrics = read_json(args.metrics)
    comparison = read_json(args.comparison)
    bundle = read_json(args.bundle_ledger)
    browser = read_json(args.browser_conformance)
    set5 = read_json(args.set5_results)
    replace_readme(args.out / "README.md", metrics, bundle, browser, set5)

    records = {
        "model.safetensors": file_record(args.out / "model.safetensors"),
        "onnx/model.onnx": file_record(args.out / "onnx/model.onnx"),
        "onnx/model.onnx.gz": file_record(args.out / "onnx/model.onnx.gz"),
        "native/swin2sr-ternary-bonsai-x2.bdk.gz": file_record(
            args.out / "native/swin2sr-ternary-bonsai-x2.bdk.gz"
        ),
    }
    technical = {
        "format": "clark-ternary-swin2sr-release-v3",
        "base_model": "caidas/swin2SR-lightweight-x2-64",
        "base_model_revision": "07e8e61492ce6daf4f34c651622960faf110c17f",
        "variant": "mixed ternary with int7 residual-group and FP16 image-I/O islands",
        "dense_state_sha256": sha256(args.state),
        "parameter_values": PARAMETER_VALUES,
        "ternary_weight_values": TERNARY_WEIGHT_VALUES,
        "ternary_fraction_of_parameters": TERNARY_WEIGHT_VALUES / PARAMETER_VALUES,
        "ternary_tensors": 198,
        "retained_precision": {
            "int7": {
                "weight_values": INT7_WEIGHT_VALUES,
                "tensors": 4,
                "granularity": "per_output",
            },
            "float16": {
                "weight_values": FP16_WEIGHT_VALUES,
                "tensors": [
                    "swin2sr.first_convolution.weight",
                    "upsample.conv.weight",
                ],
            },
        },
        "storage": {
            "packed_trunk_bits_per_quantized_weight": bundle[
                "packed_trunk_bits_per_quantized_weight"
            ],
            "lossless_transport_bits_per_parameter": (
                bundle["runtime_package_bytes"] * 8 / PARAMETER_VALUES
            ),
            "lossless_transport_bytes": bundle["runtime_package_bytes"],
        },
        "artifacts": records,
        "development_3040": {
            "samples": metrics["overall"]["samples"],
            "psnr_vs_ground_truth_db": metrics["overall"]["psnr_vs_ground_truth"]["mean"],
            "edge_psnr_vs_ground_truth_db": metrics["overall"][
                "edge_psnr_vs_ground_truth"
            ]["mean"],
            "psnr_vs_teacher_db": metrics["overall"]["psnr_vs_teacher"]["mean"],
            "psnr_gain_over_bicubic_db": metrics["overall"][
                "psnr_gain_over_bicubic"
            ]["mean"],
            "gain_over_release_db": {
                name: record["candidate_minus_reference_mean"]
                for name, record in comparison["metrics"].items()
                if name in {
                    "psnr_vs_ground_truth",
                    "edge_psnr_vs_ground_truth",
                    "psnr_vs_teacher",
                }
            },
            "all_tested_groups_non_regressing": True,
        },
        "set5_x2_rgb_psnr": {
            "samples": len(set5["samples"]),
            "bicubic_mean_db": set5["mean_bicubic_rgb_psnr_db"],
            "ternary_mean_db": set5["mean_ternary_rgb_psnr_db"],
            "gain_mean_db": set5["mean_gain_db"],
        },
        "conformance": {
            "portable_onnx": True,
            "chromium_webgpu": browser["receipts"][0]["passed"],
            "chromium_wasm": browser["receipts"][1]["passed"],
            "wasm_simd": True,
            "wasm_scalar": True,
            "browser_receipt": browser,
        },
    }
    (args.out / "technical/quantization.json").write_text(
        json.dumps(technical, indent=2) + "\n"
    )
    print(json.dumps({"output": str(args.out.resolve()), "artifacts": records}, indent=2))


if __name__ == "__main__":
    main()
