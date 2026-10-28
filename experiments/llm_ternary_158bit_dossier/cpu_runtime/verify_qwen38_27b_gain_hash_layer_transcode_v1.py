#!/usr/bin/env python3
"""Verify direct enumerative-to-mask64 Qwen layer transcoding."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from clark_quantize.base3_chunks import atomic_bytes, canonical_json
from clark_quantize.qwen38_fixed_support_codec import pack_fixed_support_matrix
from clark_quantize.qwen38_fixed_support_mask64_cpu import (
    load_mask64_cpu_library,
    native_pack_mask64_matrix,
)
from clark_quantize.qwen38_gain_hash_layer_transcode import (
    load_gain_hash_layer_transcode_cpu_library,
    native_transcode_fixed_support_mask64,
)


QWEN_SHAPES = (
    ("gate_or_up_projection", 17408, 5120),
    ("down_projection", 5120, 17408),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def random_fixed_support(
    generator: np.random.Generator, *, rows: int, columns: int
) -> np.ndarray:
    codes = np.zeros((rows, columns), dtype=np.int8)
    for group in codes.reshape(-1, 128):
        support = generator.choice(128, size=64, replace=False)
        group[support] = generator.choice(
            np.array([-1, 1], dtype=np.int8), size=64
        )
    return codes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO
        / (
            "experiments/llm_ternary_158bit_dossier/cpu_runtime/receipts/"
            "qwen38-gain-hash-layer-transcode-v1.json"
        ),
    )
    args = parser.parse_args()
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("a C11 compiler is required for the transcode verifier")
    c_source = REPO / "clark_quantize/qwen38_gain_hash_layer_transcode.c"
    mask_source = REPO / "clark_quantize/qwen38_fixed_support_mask64_cpu.c"
    python_source = REPO / "clark_quantize/qwen38_gain_hash_layer_transcode.py"
    tests = REPO / "clark_quantize/test_qwen38_gain_hash_layer_transcode.py"
    verifier = Path(__file__).resolve()
    extension = ".dylib" if platform.system() == "Darwin" else ".so"
    shared_flags = (
        ["-dynamiclib"]
        if platform.system() == "Darwin"
        else ["-shared", "-fPIC"]
    )
    compile_command = [
        compiler,
        "-std=c11",
        "-O3",
        "-Wall",
        "-Wextra",
        "-Werror",
        *shared_flags,
        str(c_source),
        str(mask_source),
    ]
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / f"gain-layer-transcode{extension}"
        subprocess.run([*compile_command, "-o", str(output)], check=True)
        transcode = load_gain_hash_layer_transcode_cpu_library(output)
        mask64 = load_mask64_cpu_library(output)
        if transcode is None or mask64 is None:
            raise RuntimeError("fresh transcode verifier library did not load")
        generator = np.random.default_rng(158359)
        case_hashes = []
        exact_cases = 0
        for case in range(100):
            rows = int(generator.integers(1, 9))
            columns = int(generator.integers(1, 6)) * 128
            codes = random_fixed_support(
                generator, rows=rows, columns=columns
            )
            encoded = pack_fixed_support_matrix(
                codes, group_size=128, nonzero_per_group=64
            )
            direct = native_transcode_fixed_support_mask64(
                encoded, library=transcode
            )
            dense_control = native_pack_mask64_matrix(codes, library=mask64)
            if direct == dense_control:
                exact_cases += 1
            case_hashes.append(hashlib.sha256(direct.payload).hexdigest())

    shape_records = []
    for name, rows, columns in QWEN_SHAPES:
        groups = rows * (columns // 128)
        input_bits = groups * 189
        input_bytes = (input_bits + 7) // 8
        output_bytes = groups * 24
        shape_records.append(
            {
                "name": name,
                "rows": rows,
                "columns": columns,
                "parameters": rows * columns,
                "groups": groups,
                "canonical_enumerative_bytes": input_bytes,
                "mask64_compute_bytes": output_bytes,
                "dense_INT8_intermediate_bytes_avoided": rows * columns,
                "compute_expansion_bytes": output_bytes - input_bytes,
                "compute_expansion_ratio": output_bytes / input_bytes,
                "size_contract_passed": input_bytes == 16_450_560
                and output_bytes == 16_711_680,
            }
        )
    passed = bool(
        exact_cases == 100
        and len(set(case_hashes)) == 100
        and all(record["size_contract_passed"] for record in shape_records)
    )
    receipt = {
        "schema": "qwen38_27b_gain_hash_layer_transcode_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "Qwen/Qwen3.8-27B",
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "build": {
            "compiler": subprocess.check_output(
                [compiler, "--version"], text=True
            ).splitlines()[0],
            "command": [
                Path(value).name if value in {str(c_source), str(mask_source)} else value
                for value in compile_command
            ],
        },
        "sources": {
            "native_transcode": {
                "path": str(c_source.relative_to(REPO)),
                "sha256": sha256_file(c_source),
            },
            "binding": {
                "path": str(python_source.relative_to(REPO)),
                "sha256": sha256_file(python_source),
            },
            "mask64_control": {
                "path": str(mask_source.relative_to(REPO)),
                "sha256": sha256_file(mask_source),
            },
            "tests": {
                "path": str(tests.relative_to(REPO)),
                "sha256": sha256_file(tests),
            },
            "verifier": {
                "path": str(verifier.relative_to(REPO)),
                "sha256": sha256_file(verifier),
            },
        },
        "correctness": {
            "randomized_cases": 100,
            "byte_identical_cases": exact_cases,
            "distinct_output_hashes": len(set(case_hashes)),
            "control": "dense INT8 codes packed by the frozen native mask64 packer",
        },
        "qwen_shape_records": shape_records,
        "gate": {
            "require_100_byte_identical_randomized_cases": True,
            "require_100_distinct_outputs": True,
            "require_exact_Qwen_shape_sizes": True,
            "passed": passed,
        },
        "complete": True,
        "terminal": True,
        "claim_boundary": {
            "direct_native_transcode_correctness_proven": passed,
            "full_Qwen_layer_dense_intermediate_avoided_by_API": True,
            "full_shape_runtime_or_peak_RSS_measured": False,
            "real_model_weights_transcoded": False,
            "quality_or_capability_proven": False,
            "token_throughput_proven": False,
            "semi_lossless_or_breakthrough_proven": False,
        },
    }
    atomic_bytes(args.out, canonical_json(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
