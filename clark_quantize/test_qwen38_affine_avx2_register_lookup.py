from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PREREG = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "qwen38_27b_affine_avx2_register_lookup_v18.json"
)
KERNEL = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime/kernels"
    / "qwen38_affine_avx2_register_lookup_v18.cpp"
)
RUNNER = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "run_qwen38_27b_affine_avx2_register_lookup_v18.py"
)
RECEIPTS = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime/receipts"
    / "qwen38_27b_affine_avx2_register_lookup_v18"
)


def load_preregistration() -> dict[str, object]:
    return json.loads(PREREG.read_text())


def test_v18_is_frozen_to_qwen38_and_amd_avx2() -> None:
    prereg = load_preregistration()
    assert prereg["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert prereg["scope"]["qwen_4b_compute_forbidden"] is True
    assert prereg["scope"]["baseline_kernel"] == "AFF3_X16 v17"
    assert prereg["scope"]["required_isa"] == ["AVX2", "SSSE3", "F16C", "FMA"]


def test_v18_is_a_zero_bpw_lookup_only_change() -> None:
    prereg = load_preregistration()
    runtime = prereg["runtime_contract"]
    assert runtime["runtime_compute_image_bpw_at_5120_columns"] == 2.06875
    assert runtime["additional_persistent_bytes"] == 0
    assert runtime["additional_runtime_compute_image_bytes"] == 0
    assert runtime["decoded_per_group_scale_matrix_materialized"] is False
    frozen = prereg["frozen_kernel_change"]
    assert any("Replace only" in item for item in frozen["allowed"])
    assert any("accumulation order" in item for item in frozen["forbidden"])


def test_v18_has_exactness_and_material_speed_gates() -> None:
    gates = load_preregistration()["sequential_gates"]
    correctness = gates["gate_1_native_correctness"]
    assert correctness["all_16_codebook_indices_must_be_exercised"] is True
    assert correctness["lookup_fp32_bits_must_match_scalar_fp16_replay"] is True
    assert correctness["v17_v18_output_bits_must_match"] is True
    assert correctness["actual_authenticated_qwen_tensor_required"] is True
    timing = gates["gate_2_exact_shape_microbenchmark"]
    assert timing["rows_per_matrix"] == 17_408
    assert timing["columns"] == 5_120
    assert timing["minimum_v17_over_v18_speedup"] >= 1.05
    assert timing["maximum_candidate_median_absolute_deviation_fraction"] <= 0.03
    assert timing["maximum_candidate_p90_over_p10_ratio"] <= 1.08


def test_v18_cannot_be_promoted_from_a_microkernel_result() -> None:
    prereg = load_preregistration()
    model = prereg["sequential_gates"]["gate_3_model_integration"]
    assert model[
        "run_only_if_native_gates_pass_and_a_separately_preregistered_representation_gate_passes"
    ] is True
    assert model["minimum_tokens_per_second"] >= 2.75
    assert model["minimum_speedup_vs_same_harness_q4_k_m"] >= 1.4
    assert "Whole-model" in prereg["claim_boundary"]


def test_v18_kernel_uses_register_lookup_without_gathers() -> None:
    source = KERNEL.read_text()
    assert "cq3_aff3_gemv_x16_register_v18" in source
    assert "_mm_shuffle_epi8" in source
    assert "_mm256_cvtph_ps" in source
    assert "_mm256_i32gather_ps" not in source
    assert "Aff3X16Group" in source
    assert "additional runtime" in source


def test_v18_runner_compares_v17_and_v18_under_frozen_gates() -> None:
    source = RUNNER.read_text()
    assert "cq3_aff3_gemv_x16_v17" in source
    assert "cq3_aff3_gemv_x16_register_v18" in source
    assert "v17/v18 output bits changed" in source
    assert "all sixteen codebook indices were not exercised" in source
    assert "len(os.sched_getaffinity(0)) != 1" in source
    assert "minimum_v17_over_v18_speedup" in source
    assert "median_absolute_deviation_fraction" in source
    assert "p90_over_p10_ratio" in source
    assert "QAT service is active" in source


def test_v18_native_receipts_are_exact_and_authenticated() -> None:
    synthetic = json.loads((RECEIPTS / "native-self-test.json").read_text())
    actual = json.loads((RECEIPTS / "actual-artifact.json").read_text())
    build = json.loads((RECEIPTS / "build-manifest.json").read_text())
    assert synthetic["all_sixteen_codebook_indices_exercised"] is True
    assert synthetic["lookup_fp32_bits_match_scalar_fp16_replay"] is True
    assert synthetic["additional_runtime_compute_image_bytes"] == 0
    assert all(shape["integer_dot_exact"] for shape in synthetic["shapes"])
    assert all(shape["v17_v18_output_bit_exact"] for shape in synthetic["shapes"])
    assert actual["all_source_blobs_authenticated"] is True
    assert actual["shape"] == [48, 5120]
    assert actual["v17_v18_output_bit_exact"] is True
    assert actual["baseline_output_sha256"] == actual["candidate_output_sha256"]
    assert build["timing_execution"]["infrastructure_valid"] is True
    assert len(build["sha256"]["v17_runner_dependency"]) == 64
    assert build["claim_boundary"].startswith("Authenticated native primitive")


def test_v18_materially_beats_v17_on_the_frozen_exact_shape() -> None:
    receipt = json.loads((RECEIPTS / "exact-shape-benchmark.json").read_text())
    assert receipt["rows_per_matrix"] == 17_408
    assert receipt["columns"] == 5_120
    assert receipt["matrices"] == 2
    assert receipt["timed_batches"] == 11
    assert receipt["inner_iterations_per_timed_batch"] == 8
    assert receipt["v17_over_v18_speedup"] >= receipt["minimum_v17_over_v18_speedup"]
    assert receipt["samples"]["v18"]["median_absolute_deviation_fraction"] <= 0.03
    assert receipt["samples"]["v18"]["p90_over_p10_ratio"] <= 1.08
    assert all(receipt["checks"].values())
    assert receipt["passed"] is True
    assert receipt["complete"] is True
    assert receipt["terminal"] is True
