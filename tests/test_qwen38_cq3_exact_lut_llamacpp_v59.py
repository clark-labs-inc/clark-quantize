from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREGISTRATION = (
    RUNTIME / "qwen38_27b_cq3_exact_lut32_llamacpp_preregistration_v59.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v59_is_frozen_before_integration_and_milan_result() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text())

    assert preregistration["status"] == (
        "frozen_before_llamacpp_implementation_and_before_milan_v58_result"
    )
    assert preregistration["scope"]["active_model_only"] == "Qwen/Qwen3.8-27B"
    assert preregistration["scope"]["qwen_4b_compute_forbidden"]
    assert preregistration["admission_rule"]["required_parent_passed"]
    assert preregistration["admission_rule"][
        "minimum_geometric_mean_x8_over_lut32"
    ] == 1.15
    assert "Do not implement" in preregistration["admission_rule"]["rule"]


def test_v59_authenticates_exact_v57_and_v58_sources() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text())
    parents = preregistration["frozen_parents"]

    assert parents["v57_execution_receipt_sha256"] == sha256(
        RUNTIME
        / "receipts/qwen38_27b_cq3_aff3_residual_llamacpp_v57-execution.json"
    )
    assert parents["v57_transform_script_sha256"] == sha256(
        RUNTIME / "llama_cpp/prepare_llama_cpp_qwen38_cq3_aff3_residual_v57.py"
    )
    assert parents["v58_preregistration_sha256"] == sha256(
        RUNTIME / "qwen38_27b_cq3_exact_lut32_avx2_preregistration_v58.json"
    )
    assert parents["v58_native_source_sha256"] == sha256(
        REPO / "clark_quantize/qwen38_cq3_exact_lut_cpu.c"
    )
    assert parents["v58_native_header_sha256"] == sha256(
        REPO / "clark_quantize/qwen38_cq3_exact_lut_cpu.h"
    )
    assert parents["v58_full_shape_benchmark_sha256"] == sha256(
        RUNTIME / "kernels/benchmark_qwen38_cq3_exact_lut_native_v58.c"
    )


def test_v59_freezes_storage_math_source_and_whole_model_gates() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text())
    implementation = preregistration["implementation_contract"]
    model_gates = preregistration["model_gates"]

    assert implementation["runtime_compute_image_bytes_per_tile"] == 2176
    assert implementation["source_bytes_per_tile"] == 2176
    assert implementation["persistent_storage_expansion_fraction"] == 0.0
    assert implementation["ggml_repack_template"] == {
        "activation_interleave": 8,
        "output_chunk_rows": 32,
        "weight_type": "block_cq3_0",
        "activation_type": "GGML_TYPE_CQ8_0",
    }
    assert implementation["eligible_output_dimensions"] == [5120, 17408]
    assert implementation["high_precision_weight_materialization"] is False
    assert len(preregistration["source_change_boundary"]["allowed_llamacpp_files"]) == 6
    assert model_gates["required_quality_suites"] == [
        "WikiText",
        "CUTE character-level",
        "MATH-500",
        "code",
        "MMLU",
    ]
    assert model_gates["minimum_decode_tokens_per_second"] == 2.75
    assert model_gates["minimum_decode_speedup_vs_q4_k_m"] == 1.4
    assert model_gates["maximum_wikitext_perplexity_ratio_vs_q4_k_m"] == 1.05
    assert "Microkernel" in preregistration["decision_rules"]["no_claim"]
