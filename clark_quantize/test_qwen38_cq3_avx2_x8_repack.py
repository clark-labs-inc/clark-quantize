from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_cq3_avx2_x8_repack_v15.json"
PATCH = RUNTIME / "kernels/llama_cpp_cq3_avx2_x8_repack_v15.patch"


def test_v15_is_storage_neutral_and_qwen38_27b_only() -> None:
    prereg = json.loads(PREREG.read_text())
    scope = prereg["scope"]
    assert scope["model"] == "Qwen/Qwen3.8-27B"
    assert scope["qwen_4b_compute_forbidden"] is True
    assert scope["source_weight_bits_per_weight"] == 2.125
    assert scope["runtime_weight_bits_per_weight"] == 2.125
    layout = prereg["runtime_layout"]
    assert layout["rows_per_tile"] == 8
    assert layout["runtime_bytes_per_tile"] == layout["source_bytes_per_tile"]
    assert layout["storage_expansion_fraction"] == 0.0
    assert layout["repack_is_lossless"] is True


def test_v15_requires_architecture_sized_speedup_before_model_eval() -> None:
    prereg = json.loads(PREREG.read_text())
    native = prereg["sequential_gates"]["gate_1_native_correctness"]
    assert native["integer_dot_must_match_scalar_exactly"] is True
    assert native["maximum_scaled_float_error"] <= 5e-5
    performance = prereg["sequential_gates"]["gate_2_exact_shape_microbenchmark"]
    assert performance["rows_per_matrix"] == 17_408
    assert performance["columns"] == 5_120
    assert performance["repetitions"] == 9
    assert performance["minimum_control_over_candidate_speedup"] >= 1.25
    assert performance["maximum_candidate_sample_range_fraction"] <= 0.05
    integration = prereg["sequential_gates"]["gate_3_selective_27b_integration"]
    assert integration["run_only_if_all_native_gates_pass"] is True
    quality = prereg["sequential_gates"]["gate_4_quality"]
    assert quality["run_only_if_all_performance_gates_pass"] is True


def test_v15_maps_vector_lanes_to_outputs_without_dense_weights() -> None:
    prereg = json.loads(PREREG.read_text())
    algebra = prereg["kernel_algebra"]
    assert "output row" in algebra["lane_mapping"]
    assert algebra["chunk_weights_per_row"] == 16
    assert algebra["chunks_per_128_weight_group"] == 8
    assert algebra["high_precision_weights_materialized"] is False
    assert "maddubs" in algebra["decode"]
    source = PATCH.read_text()
    assert "block_cq3_0x8" in source
    assert "8*sizeof(block_cq3_0)" in source
    assert "ggml_repack_cq3_0_to_cq3_0x8" in source
    assert "_mm256_set1_epi32(values_word0)" in source
    assert "_mm256_storeu_ps(s + row, sumf)" in source
