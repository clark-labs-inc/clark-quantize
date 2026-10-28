from __future__ import annotations

import ctypes
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from clark_quantize.qwen38_cq3_exact_lut import (
    GROUP,
    build_int16_subset_lut,
    direct_ternary_group_dot,
    exact_lut_group_dot,
    pack_ternary_bitplane_quartets,
)


REPO = Path(__file__).resolve().parents[1]
PREREGISTRATION = REPO / (
    "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
    "qwen38_27b_cq3_exact_lut32_avx2_preregistration_v58.json"
)
NATIVE_SOURCE = REPO / "clark_quantize/qwen38_cq3_exact_lut_cpu.c"
NATIVE_HARNESS = REPO / (
    "experiments/llm_ternary_158bit_dossier/cpu_runtime/kernels/"
    "qwen38_cq3_exact_lut_native_v58.c"
)
NATIVE_BUILD = REPO / "scripts/cpu/build_qwen38_cq3_exact_lut_native_v58.sh"
BENCHMARK = REPO / (
    "experiments/llm_ternary_158bit_dossier/cpu_runtime/kernels/"
    "benchmark_qwen38_cq3_exact_lut_native_v58.c"
)
BENCHMARK_BUILD = REPO / (
    "scripts/cpu/build_qwen38_cq3_exact_lut_benchmark_v58.sh"
)
DEVELOPMENT_RECEIPT = REPO / (
    "experiments/llm_ternary_158bit_dossier/cpu_runtime/receipts/"
    "qwen38_27b_cq3_exact_lut32_avx2_v58/development-parity.json"
)


@pytest.mark.parametrize("seed", [20260819, 380027, 158])
def test_exact_lut_matches_direct_cq3_integer_algebra(seed: int) -> None:
    random = np.random.default_rng(seed)
    codes = random.integers(-1, 2, size=(32, GROUP), dtype=np.int8)
    activations = random.integers(-128, 128, size=GROUP, dtype=np.int16).astype(
        np.int8
    )

    packed = pack_ternary_bitplane_quartets(codes)
    table = build_int16_subset_lut(activations)
    actual = exact_lut_group_dot(packed, table, int(activations.sum(dtype=np.int32)))
    expected = direct_ternary_group_dot(codes, activations)

    np.testing.assert_array_equal(actual, expected)
    assert packed.nbytes == codes.size * 2 // 8
    assert table.shape == (GROUP // 4, 16)
    assert table.nbytes == 1024


@pytest.mark.parametrize(
    ("codes", "activations"),
    [
        (np.ones((32, GROUP), dtype=np.int8), np.full(GROUP, 127, dtype=np.int8)),
        (-np.ones((32, GROUP), dtype=np.int8), np.full(GROUP, -128, dtype=np.int8)),
        (
            np.tile(np.array([-1, 0, 1, 0], dtype=np.int8), (32, GROUP // 4)),
            np.tile(np.array([-128, 127, -127, 126], dtype=np.int8), GROUP // 4),
        ),
    ],
)
def test_exact_lut_adversarial_group_bounds(
    codes: np.ndarray, activations: np.ndarray
) -> None:
    packed = pack_ternary_bitplane_quartets(codes)
    table = build_int16_subset_lut(activations)
    actual = exact_lut_group_dot(packed, table, int(activations.sum(dtype=np.int32)))
    expected = direct_ternary_group_dot(codes, activations)

    np.testing.assert_array_equal(actual, expected)
    assert int(actual.min()) >= np.iinfo(np.int16).min
    assert int(actual.max()) <= np.iinfo(np.int16).max


def test_exact_lut_rejects_reserved_or_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="ternary"):
        pack_ternary_bitplane_quartets(np.array([[0, 1, 2, 0]], dtype=np.int8))
    with pytest.raises(ValueError, match="quartets"):
        pack_ternary_bitplane_quartets(np.zeros((1, 3), dtype=np.int8))
    with pytest.raises(TypeError, match="int8"):
        build_int16_subset_lut(np.zeros(GROUP, dtype=np.int16))


def test_v58_preregistration_freezes_material_native_and_claim_gates() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text())

    assert preregistration["model"] == "Qwen/Qwen3.8-27B"
    assert preregistration["status"] == "frozen_before_native_implementation"
    assert preregistration["frozen_shapes"] == [
        {"columns": 5120, "role": "MLP gate and up", "rows": 17408},
        {"columns": 17408, "role": "MLP down", "rows": 5120},
    ]
    assert (
        preregistration["implementation_contract"][
            "persistent_storage_expansion_fraction"
        ]
        == 0.0
    )
    assert preregistration["numerical_gates"]["integer_group_dot_bit_exact"]
    assert preregistration["numerical_gates"][
        "scaled_fp32_output_bit_exact_vs_x8"
    ]
    assert "1.15x over x8" in preregistration["decision_rules"]["pass"]
    assert "1.45x over ordinary" in preregistration["decision_rules"]["pass"]
    assert preregistration["literature_boundary"]["primary_prior_art"][0][
        "commit"
    ] == "7042f8f73330bd083bc1e4bc5ccb3f88a4904aee"
    assert "cannot establish" in preregistration["claim_boundary"]


def test_v58_native_scalar_repack_and_lut_match_direct_algebra(tmp_path: Path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler unavailable")
    library_path = tmp_path / "libqwen38_cq3_exact_lut.so"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            str(NATIVE_SOURCE),
            "-o",
            str(library_path),
        ],
        check=True,
    )
    library = ctypes.CDLL(str(library_path))

    class SourceBlock(ctypes.Structure):
        _fields_ = [
            ("qs", ctypes.c_uint8 * 64),
            ("d", ctypes.c_uint16 * 2),
        ]

    class Activation(ctypes.Structure):
        _fields_ = [
            ("d", ctypes.c_float),
            ("qs", ctypes.c_int8 * GROUP),
            ("sum", ctypes.c_int16),
        ]

    library.cq3_exact_lut_compute_image_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
    library.cq3_exact_lut_compute_image_bytes.restype = ctypes.c_size_t
    library.cq3_exact_lut_temporary_bytes.argtypes = [ctypes.c_int]
    library.cq3_exact_lut_temporary_bytes.restype = ctypes.c_size_t
    library.cq3_exact_lut_repack.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    library.cq3_exact_lut_build_tables.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    library.cq3_exact_lut_gemv_scalar_trusted.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]

    rows = 32
    columns = 256
    random = np.random.default_rng(20260819)
    codes = random.integers(-1, 2, size=(rows, columns), dtype=np.int8)
    q8 = random.integers(-128, 128, size=(2, GROUP), dtype=np.int16).astype(
        np.int8
    )
    weight_scales = np.array(
        [[0.00390625, 0.0078125] for _ in range(rows)], dtype=np.float16
    )
    activation_scales = np.array([0.015625, 0.03125], dtype=np.float32)

    source = (SourceBlock * rows)()
    for row in range(rows):
        for group in range(2):
            source[row].d[group] = int(weight_scales[row, group].view(np.uint16))
            for position in range(GROUP):
                lane = position & 31
                quarter = position >> 5
                digit = int(codes[row, group * GROUP + position]) + 1
                source[row].qs[group * 32 + lane] |= digit << (2 * quarter)

    activations = (Activation * 2)()
    for group in range(2):
        activations[group].d = float(activation_scales[group])
        activations[group].sum = int(q8[group].sum(dtype=np.int32))
        for position in range(GROUP):
            activations[group].qs[position] = int(q8[group, position])

    compute_bytes = library.cq3_exact_lut_compute_image_bytes(rows, columns)
    temporary_bytes = library.cq3_exact_lut_temporary_bytes(columns)
    assert compute_bytes == ctypes.sizeof(source)
    assert temporary_bytes == 2 * 1024
    compute_image = (ctypes.c_uint8 * compute_bytes)()
    tables = (ctypes.c_uint8 * temporary_bytes)()
    output = (ctypes.c_float * rows)()

    assert library.cq3_exact_lut_repack(source, compute_image, rows, columns) == 0
    assert library.cq3_exact_lut_build_tables(activations, tables, columns) == 0
    assert (
        library.cq3_exact_lut_gemv_scalar_trusted(
            compute_image, activations, tables, output, rows, columns
        )
        == 0
    )

    expected = np.zeros(rows, dtype=np.float32)
    for group in range(2):
        integer_dot = (
            codes[:, group * GROUP : (group + 1) * GROUP].astype(np.int32)
            @ q8[group].astype(np.int32)
        )
        scale = np.float32(weight_scales[:, group].astype(np.float32)) * np.float32(
            activation_scales[group]
        )
        expected = np.float32(expected + integer_dot.astype(np.float32) * scale)
    np.testing.assert_allclose(
        np.ctypeslib.as_array(output), expected, rtol=2e-6, atol=2e-6
    )


def test_v58_native_harness_checks_direct_scalar_and_avx2_bits() -> None:
    source = NATIVE_HARNESS.read_text()
    build = NATIVE_BUILD.read_text()

    assert "scalar output changed direct algebra" in source
    assert "AVX2 output changed scalar bits" in source
    assert "compute_bytes != sizeof(source)" in source
    assert "avx2_scalar_bit_exact" in source
    assert "-mavx2 -mfma -mf16c" in build
    assert "-Wall -Wextra -Werror" in build
    assert "mktemp -d" in build


def test_v58_development_receipt_is_bounded_and_source_authenticated() -> None:
    receipt = json.loads(DEVELOPMENT_RECEIPT.read_text())

    assert receipt["parent_preregistration_commit"] == "1c353c3"
    assert receipt["result"]["scalar_direct_bit_exact"]
    assert receipt["result"]["avx2_scalar_bit_exact"]
    assert receipt["result"]["compute_bytes"] == 32 * 68
    assert receipt["sources"]["native_source_sha256"] == (
        "880386b55ce01c8457c258fa0d1c9b36e1503ca1c6541b0166a2206bf4b6d497"
    )
    assert receipt["sources"]["native_header_sha256"] == (
        "46f028717bdce7343131c554aed98a9600af55cdbdc05dd7c96a543d19673308"
    )
    assert "not the preregistered Milan full-shape timing" in receipt[
        "claim_boundary"
    ]


def test_v58_full_shape_harness_matches_frozen_abba_contract() -> None:
    source = BENCHMARK.read_text()
    build = BENCHMARK_BUILD.read_text()

    assert "WARMUP_PAIRS = 2" in source
    assert "MEASURED_PAIRS = 21" in source
    assert "EVICTION_BYTES = 256 * 1024 * 1024" in source
    assert '(shape_spec){"gate_up", 2, 17408, 5120}' in source
    assert '(shape_spec){"down", 1, 5120, 17408}' in source
    assert "run_x8, run_lut32" in source
    assert "run_ordinary, run_lut32" in source
    assert "full-shape LUT32 output differs from x8 bits" in source
    assert "maximum_scaled_delta > 0.00005f" in source
    assert "cq3_exact_lut_build_tables" in source
    assert "-mavx2 -mfma -mf16c" in build
    assert "-Wall -Wextra -Werror" in build
