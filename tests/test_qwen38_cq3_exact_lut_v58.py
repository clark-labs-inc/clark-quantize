from __future__ import annotations

import json
from pathlib import Path

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
