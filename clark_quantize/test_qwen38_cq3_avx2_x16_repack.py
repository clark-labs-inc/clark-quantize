from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_cq3_avx2_x16_repack_v16.json"


def test_v16_is_storage_neutral_and_qwen38_27b_only() -> None:
    prereg = json.loads(PREREG.read_text())
    scope = prereg["scope"]
    assert scope["model"] == "Qwen/Qwen3.8-27B"
    assert scope["qwen_4b_compute_forbidden"] is True
    assert scope["weight_bits_per_weight"] == 2.125
    layout = prereg["runtime_layout"]
    assert layout["rows_per_tile"] == 16
    assert layout["runtime_bytes_per_tile"] == layout["source_bytes_per_tile"]
    assert layout["storage_expansion_fraction"] == 0.0


def test_v16_requires_material_gain_over_x8_and_ordinary_cq3() -> None:
    prereg = json.loads(PREREG.read_text())
    correctness = prereg["sequential_gates"]["gate_0_repack_and_correctness"]
    assert correctness["x16_output_must_match_x8_bits"] is True
    assert correctness["integer_dot_must_match_scalar_exactly"] is True
    performance = prereg["sequential_gates"]["gate_1_exact_shape_microbenchmark"]
    assert performance["rows_per_matrix"] == 17_408
    assert performance["columns"] == 5_120
    assert performance["repetitions"] == 9
    assert performance["minimum_x8_over_x16_speedup"] >= 1.08
    assert performance["minimum_ordinary_over_x16_speedup"] >= 1.4
    assert performance["maximum_x16_sample_range_fraction"] <= 0.05


def test_v16_does_not_repeat_layer0_only_model_gate() -> None:
    prereg = json.loads(PREREG.read_text())
    deployment = prereg["sequential_gates"]["gate_2_model_deployment"]
    quality = prereg["sequential_gates"]["gate_3_quality"]
    assert deployment["run_in_this_experiment"] is False
    assert quality["run_in_this_experiment"] is False
    assert "deeper CQ3" in deployment["reason"]
