from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_q4_k_ragged_rows_kernel_v1.json"
PATCH = RUNTIME / "kernels/llama_cpp_q4_k_ragged_rows_v1.patch"


def test_ragged_kernel_is_qwen38_only_and_source_pinned() -> None:
    payload = json.loads(PREREG.read_text())
    assert payload["scope"]["model"] == "Qwen/Qwen3.8-27B"
    assert payload["scope"]["qwen_4b_compute_forbidden"] is True
    assert payload["scope"]["llama_cpp_revision"] == (
        "4df29be4f4c3673f428170fda944a5b19f743bb8"
    )
    assert payload["qwen_shapes"]["hidden_size"] == 5120
    assert payload["qwen_shapes"]["intermediate_size"] == 17408
    assert payload["qwen_shapes"]["ragged_rows"] == [2, 3]


def test_ragged_kernel_has_correctness_and_material_speed_gates() -> None:
    payload = json.loads(PREREG.read_text())
    correctness = payload["promotion_gate"]["correctness"]
    performance = payload["promotion_gate"]["performance"]
    assert correctness["maximum_nmse"] <= 5e-4
    assert correctness["exact_greedy_token_parity"] is True
    assert performance["minimum_width_2_microkernel_speedup_vs_upstream"] >= 1.25
    assert performance["minimum_width_3_microkernel_speedup_vs_upstream"] >= 1.5
    assert performance["minimum_width_3_whole_model_accepted_tps_speedup_vs_upstream"] >= 1.15
    assert payload["protocol"]["whole_model"]["run_alone"] is True


def test_patch_fuses_ragged_dispatch_and_uses_output_stride() -> None:
    source = PATCH.read_text()
    assert "ragged_rows, ncols" in source
    assert "for (int64_t x = 0; x < nc / 8; x++)" in source
    assert "for (int64_t y = 0; y < nr; y++)" in source
    assert "y * bs + x * 8" in source
    assert "y * nr + x * 8" in source
    assert "block_q4_K" in source
