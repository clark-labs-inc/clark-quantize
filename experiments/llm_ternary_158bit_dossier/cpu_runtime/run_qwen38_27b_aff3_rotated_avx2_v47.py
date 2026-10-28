#!/usr/bin/env python3
"""Correctness harness for the fused Qwen3.8 AFF3 rotation/GEMV kernel."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RUNTIME = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import run_qwen38_27b_affine_avx2_x16_fused_v17 as v17

from clark_quantize.qwen38_aff3_runtime_image import Aff3RuntimeImage

MODEL = "Qwen/Qwen3.8-27B"
GROUP_SIZE = 128


def pointer(array: np.ndarray, ctype: type[ctypes._SimpleCData]) -> object:
    return array.ctypes.data_as(ctypes.POINTER(ctype))


def load_library(path: Path) -> ctypes.CDLL:
    library = v17.load_library(path)
    gemv_arguments = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.cq3_aff3_gemv_x16_register_v18.argtypes = gemv_arguments
    library.cq3_aff3_gemv_x16_register_v18.restype = ctypes.c_int
    rotate_arguments = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.cq3_aff3_rotate_quantize_v47.argtypes = rotate_arguments
    library.cq3_aff3_rotate_quantize_v47.restype = ctypes.c_int
    library.cq3_aff3_rotated_gemv_x16_v47.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
    ]
    library.cq3_aff3_rotated_gemv_x16_v47.restype = ctypes.c_int
    return library


def deterministic_rotation(columns: int, stages: int) -> tuple[np.ndarray, np.ndarray]:
    pairings = np.empty((stages, columns // 2, 2), dtype=np.uint16)
    base = np.arange(columns, dtype=np.uint16)
    for stage in range(stages):
        permutation = np.roll(base, stage * 17)
        pairings[stage] = permutation.reshape(-1, 2)
    angles = np.asarray(
        np.sin(np.arange(stages * columns // 2, dtype=np.float32) * 0.013) * 0.07,
        dtype=np.float16,
    ).reshape(stages, columns // 2)
    return angles, pairings


def reference_rotate_quantize(
    input_values: np.ndarray,
    angles: np.ndarray,
    pairings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotated = np.asarray(input_values, dtype=np.float32).copy()
    for stage_angles, stage_pairings in zip(angles, pairings, strict=True):
        for angle, (first, second) in zip(stage_angles, stage_pairings, strict=True):
            theta = np.float32(angle)
            cosine = np.cos(theta, dtype=np.float32)
            sine = np.sin(theta, dtype=np.float32)
            left = rotated[first]
            right = rotated[second]
            rotated[first] = np.float32(cosine * left - sine * right)
            rotated[second] = np.float32(sine * left + cosine * right)
    groups = rotated.reshape(-1, GROUP_SIZE)
    maximum = np.max(np.abs(groups), axis=1)
    scales = np.asarray(maximum / np.float32(127), dtype=np.float32)
    inverse = np.divide(
        np.float32(1),
        scales,
        out=np.zeros_like(scales),
        where=scales != 0,
    )
    quantized = np.asarray(
        np.clip(np.rint(groups * inverse[:, None]), -127, 127), dtype=np.int8
    )
    sums = quantized.astype(np.int32).sum(1, dtype=np.int32)
    return rotated, quantized.reshape(-1), scales, sums


def call_rotate_quantize(
    library: ctypes.CDLL,
    input_values: np.ndarray,
    angles: np.ndarray,
    pairings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    columns = input_values.size
    groups = columns // GROUP_SIZE
    rotated = np.empty(columns, dtype=np.float32)
    quantized = np.empty(columns, dtype=np.int8)
    scales = np.empty(groups, dtype=np.float32)
    sums = np.empty(groups, dtype=np.int32)
    status = library.cq3_aff3_rotate_quantize_v47(
        pointer(angles.view(np.uint16), ctypes.c_uint16),
        pointer(pairings.reshape(-1), ctypes.c_uint16),
        pointer(input_values, ctypes.c_float),
        columns,
        angles.shape[0],
        pointer(rotated, ctypes.c_float),
        pointer(quantized, ctypes.c_int8),
        pointer(scales, ctypes.c_float),
        pointer(sums, ctypes.c_int32),
    )
    if status:
        raise RuntimeError(f"native rotation/quantization failed: {status}")
    return rotated, quantized, scales, sums


def call_fused(
    library: ctypes.CDLL,
    weights: np.ndarray,
    params: tuple[np.ndarray, ...],
    angles: np.ndarray,
    pairings: np.ndarray,
    input_values: np.ndarray,
    rows: int,
    columns: int,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    rotated = np.empty(columns, dtype=np.float32)
    quantized = np.empty(columns, dtype=np.int8)
    scales = np.empty(columns // GROUP_SIZE, dtype=np.float32)
    sums = np.empty(columns // GROUP_SIZE, dtype=np.int32)
    output = np.empty(rows, dtype=np.float32)
    status = library.cq3_aff3_rotated_gemv_x16_v47(
        weights.ctypes.data,
        pointer(params[0].view(np.uint16), ctypes.c_uint16),
        pointer(params[1].view(np.uint16), ctypes.c_uint16),
        pointer(params[2].view(np.uint16), ctypes.c_uint16),
        pointer(params[3].view(np.uint16), ctypes.c_uint16),
        pointer(angles.view(np.uint16), ctypes.c_uint16),
        pointer(pairings.reshape(-1), ctypes.c_uint16),
        pointer(input_values, ctypes.c_float),
        rows,
        columns,
        angles.shape[0],
        pointer(rotated, ctypes.c_float),
        pointer(quantized, ctypes.c_int8),
        pointer(scales, ctypes.c_float),
        pointer(sums, ctypes.c_int32),
        pointer(output, ctypes.c_float),
    )
    if status:
        raise RuntimeError(f"native fused AFF3 operation failed: {status}")
    return output, (rotated, quantized, scales, sums)


def self_test(library: ctypes.CDLL) -> dict[str, object]:
    rows, columns, stages = 32, 512, 3
    groups = columns // GROUP_SIZE
    codes = v17.deterministic_codes(rows, columns, 1701)
    indices = v17.combined_indices(rows, groups, 1901)
    source = v17.row_major_image(codes, indices)
    _, weights = v17.repack(library, source, rows, groups)
    params = v17.parameters(rows)
    angles, pairings = deterministic_rotation(columns, stages)
    input_values = np.asarray(
        np.cos(np.arange(columns, dtype=np.float32) * 0.021) * 0.8,
        dtype=np.float32,
    )
    expected = reference_rotate_quantize(input_values, angles, pairings)
    observed = call_rotate_quantize(library, input_values, angles, pairings)
    if not np.allclose(observed[0], expected[0], rtol=0, atol=2e-7):
        raise AssertionError("native Givens rotation changed")
    if not np.array_equal(observed[1], expected[1]):
        raise AssertionError("native rotated A8 codes changed")
    if not np.allclose(observed[2], expected[2], rtol=0, atol=2e-9):
        raise AssertionError("native rotated A8 scales changed")
    if not np.array_equal(observed[3], expected[3]):
        raise AssertionError("native rotated A8 sums changed")

    fused, scratch = call_fused(
        library,
        weights,
        params,
        angles,
        pairings,
        input_values,
        rows,
        columns,
    )
    separate = np.empty(rows, dtype=np.float32)
    v17.call_gemv(
        library.cq3_aff3_gemv_x16_register_v18,
        weights,
        params,
        scratch[1],
        scratch[2],
        scratch[3],
        rows,
        columns,
        separate,
    )
    if not np.array_equal(fused.view(np.uint32), separate.view(np.uint32)):
        raise AssertionError("fused v47 changed v18 output bits")
    return {
        "schema": "qwen38_27b_aff3_rotated_avx2_native_correctness_v47",
        "model": MODEL,
        "shape": [rows, columns],
        "stages": stages,
        "rotation_maximum_absolute_error": float(
            np.max(np.abs(observed[0] - expected[0]))
        ),
        "rotated_a8_codes_exact": True,
        "rotated_a8_sums_exact": True,
        "fused_output_bit_exact_vs_composed_v18": True,
        "output_sha256": hashlib.sha256(fused.tobytes()).hexdigest(),
        "persistent_dense_weight_bytes": 0,
        "decoded_group_scale_matrix_bytes": 0,
        "passed": True,
        "complete": True,
        "terminal": True,
    }


def runtime_image_test(library: ctypes.CDLL, path: Path) -> dict[str, object]:
    with Aff3RuntimeImage(path) as image:
        image.verify_all_sections()
        weights = np.frombuffer(image.read_section("weights_x16"), dtype=np.uint8)
        params = tuple(
            np.frombuffer(image.read_section(name), dtype="<f2").copy()
            for name in (
                "alpha_anchors",
                "mu_anchors",
                "alpha_codebook",
                "mu_codebook",
            )
        )
        angles = (
            np.frombuffer(image.read_section("angles"), dtype="<f2")
            .reshape(image.stages, image.pairs)
            .copy()
        )
        pairings = (
            np.frombuffer(image.read_section("pairings"), dtype="<u2")
            .reshape(image.stages, image.pairs, 2)
            .copy()
        )
        input_values = np.asarray(
            np.sin(np.arange(image.columns, dtype=np.float32) * 0.009) * 0.7,
            dtype=np.float32,
        )
        fused, scratch = call_fused(
            library,
            weights,
            params,
            angles,
            pairings,
            input_values,
            image.rows,
            image.columns,
        )
        separate = np.empty(image.rows, dtype=np.float32)
        v17.call_gemv(
            library.cq3_aff3_gemv_x16_register_v18,
            weights,
            params,
            scratch[1],
            scratch[2],
            scratch[3],
            image.rows,
            image.columns,
            separate,
        )
        exact = np.array_equal(fused.view(np.uint32), separate.view(np.uint32))
        if not exact:
            raise AssertionError("actual runtime image fusion changed output bits")
        return {
            "schema": "qwen38_27b_aff3_rotated_avx2_runtime_image_v47",
            "model": MODEL,
            "runtime_image": str(path),
            "runtime_image_root_sha256": image.root_sha256,
            "layer": image.layer,
            "shape": [image.rows, image.columns],
            "stages": image.stages,
            "fused_output_bit_exact_vs_composed_v18": exact,
            "output_sha256": hashlib.sha256(fused.tobytes()).hexdigest(),
            "passed": True,
            "complete": True,
            "terminal": True,
        }


def atomic_json(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("self-test", "runtime-image"))
    parser.add_argument("--runtime-image", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    library = load_library(args.library)
    if args.mode == "self-test":
        result = self_test(library)
    else:
        if args.runtime_image is None:
            parser.error("--runtime-image is required")
        result = runtime_image_test(library, args.runtime_image)
    atomic_json(result, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
