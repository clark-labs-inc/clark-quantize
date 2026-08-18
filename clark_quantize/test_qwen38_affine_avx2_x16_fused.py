from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PREREG = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "qwen38_27b_affine_avx2_x16_fused_v17.json"
)
KERNEL = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime/kernels"
    / "qwen38_affine_avx2_x16_fused_v17.cpp"
)
RUNNER = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "run_qwen38_27b_affine_avx2_x16_fused_v17.py"
)


def load_preregistration() -> dict[str, object]:
    return json.loads(PREREG.read_text())


def test_v17_is_qwen38_only_and_targets_the_actual_amd_isa() -> None:
    prereg = load_preregistration()
    scope = prereg["scope"]
    assert scope["model"] == "Qwen/Qwen3.8-27B"
    assert scope["qwen_4b_compute_forbidden"] is True
    assert scope["physical_cores"] == 4
    assert scope["required_isa"] == ["AVX2", "F16C", "FMA"]
    assert "AVX512_VNNI" in scope["forbidden_isa_assumptions"]


def test_v17_runtime_layout_is_low_bit_and_never_reconstructs_weights() -> None:
    prereg = load_preregistration()
    layout = prereg["runtime_layout"]
    assert layout["rows_per_tile"] == 16
    assert layout["weights_per_group"] == 128
    assert layout["tile_bytes_per_group"] == 16 * (32 + 1)
    assert layout["tile_payload_bits_per_weight_excluding_row_anchors"] == 2.0625
    assert layout["runtime_compute_image_maximum_bpw_at_5120_columns"] <= 2.06875
    assert layout["canonical_mapping_must_round_trip"] is True
    assert layout["high_precision_weights_materialized"] is False


def test_v17_fuses_the_mu_term_with_the_exact_ternary_dot() -> None:
    algebra = load_preregistration()["kernel_algebra"]
    assert "mu[row,g] * sum_q8[g]" in algebra["group_contribution"]
    assert "alpha[row,g] * dot[row,g]" in algebra["group_contribution"]
    assert algebra["lane_mapping"].startswith("two AVX2 vectors")
    assert algebra["butterfly_dense_matrix_materialized"] is False
    assert algebra["group_order_must_match_reference"] is True


def test_v17_has_material_native_and_end_to_end_gates() -> None:
    gates = load_preregistration()["sequential_gates"]
    native = gates["gate_2_exact_shape_microbenchmark"]
    assert native["rows_per_matrix"] == 17_408
    assert native["columns"] == 5_120
    assert native["timed_batches"] == 11
    assert native["inner_iterations_per_timed_batch"] == 8
    assert native["minimum_affine_x8_over_affine_x16_speedup"] >= 1.06
    assert native["minimum_affine_ordinary_over_affine_x16_speedup"] >= 1.35
    assert native["maximum_candidate_median_absolute_deviation_fraction"] <= 0.03
    model = gates["gate_4_model_integration"]
    assert model["minimum_tokens_per_second"] >= 2.75
    assert model["minimum_speedup_vs_same_harness_q4_k_m"] >= 1.4
    assert model["minimum_deeper_coverage"].startswith("all rank-2")


def test_v17_cannot_bypass_representation_or_quality_gates() -> None:
    prereg = load_preregistration()
    dependency = prereg["scientific_dependency"]
    assert dependency["model_promotion_requires_three_block_qat_gate_pass"] is True
    assert dependency["kernel_result_cannot_override_a_failed_representation_gate"] is True
    quality = prereg["sequential_gates"]["gate_5_quality"]
    assert quality["run_only_if_model_integration_performance_passes"] is True
    assert quality["cute_quick_denominator"] == 20
    assert quality["math500_quick_denominator"] == 32
    assert quality["mmlu_denominator"] == 64
    assert quality["maximum_wikitext_perplexity_ratio_vs_q4_control"] <= 1.05


def test_v17_kernel_implements_the_frozen_low_bit_algebra() -> None:
    source = KERNEL.read_text()
    assert "struct Aff3X16Group" in source
    assert "static_assert(sizeof(Aff3X16Group) == 528" in source
    assert "cq3_aff3_repack_x16_v17" in source
    assert "cq3_aff3_gemv_x16_v17" in source
    assert "cq3_aff3_debug_dot_x16_v17" in source
    assert "cq3_aff3_debug_decode_scales_x16_v17" in source
    assert "_mm256_maddubs_epi16" in source
    assert "_mm256_i32gather_ps" in source
    assert "activation_sums[group]" in source
    assert "floating-point weight matrix" in source


def test_v17_x16_shares_activation_broadcasts_across_two_output_vectors() -> None:
    source = KERNEL.read_text()
    x16 = source.split("int cq3_aff3_gemv_x16_v17(", 1)[1].split(
        "int cq3_aff3_debug_dot_x16_v17(", 1
    )[0]
    assert "dot0" in x16 and "dot1" in x16
    assert "total0" in x16 and "total1" in x16
    assert x16.count("load_activation_words(group_activation, chunk") == 1
    assert "packed + 0" in x16
    assert "packed + 32" in x16


def test_v17_runner_enforces_correctness_stability_and_host_isolation() -> None:
    source = RUNNER.read_text()
    assert "pack_block203_rows" in source
    assert "unpack_block203_rows" in source
    assert "transcode_block203_rows_to_group128_2bit" in source
    assert "reserved two-bit codes" in source
    assert "x8/x16 outputs changed bits" in source
    assert "len(os.sched_getaffinity(0)) != 1" in source
    assert "affine PTQ service is active" in source
    assert "INNER_ITERATIONS = 8" in source
    assert "TIMED_BATCHES = 11" in source
    assert "median_absolute_deviation_fraction" in source
    assert "p90_over_p10_ratio" in source
