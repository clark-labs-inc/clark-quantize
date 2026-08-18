from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_cq3_avx2_dual_chain_v11.json"
PATCH = RUNTIME / "kernels/llama_cpp_cq3_avx2_dual_chain_v11.patch"


def test_dual_chain_prereg_targets_only_qwen38_27b_cq3() -> None:
    prereg = json.loads(PREREG.read_text())
    assert prereg["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert prereg["scope"]["qwen_4b_compute_forbidden"] is True
    assert prereg["scope"]["weight_bits_per_weight"] == 2.125
    gate = prereg["sequential_gates"]["gate_2_exact_shape_microbenchmark"]
    assert gate["rows"] == 17_408
    assert gate["columns"] == 5_120
    assert gate["repetitions"] == 9
    assert gate["minimum_control_over_candidate_speedup"] >= 1.08


def test_dual_chain_patch_preserves_integer_dot_and_splits_fp_dependencies() -> None:
    source = PATCH.read_text()
    assert "ggml_cq3_dot_scaled_group_dual_chain_v11" in source
    assert "sumf_group0" in source
    assert "sumf_group1" in source
    assert "_mm256_maddubs_epi16(digits0, values0)" in source
    assert "_mm256_maddubs_epi16(digits3, values3)" in source
    assert "activation->sum" in source
    assert "weight->d[group]" in source
    assert "_mm256_add_ps(sumf_group0, sumf_group1)" in source
