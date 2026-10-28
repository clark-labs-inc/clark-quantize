from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RECEIPT = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime/receipts"
    / "qwen38_27b_affine_butterfly_block_gate_v1/terminal-decision.json"
)


def test_affine_ptq_gate_is_terminal_and_preserves_the_middle_block_failure() -> None:
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["model"] == "Qwen/Qwen3.8-27B"
    assert receipt["complete"] is True
    assert receipt["terminal"] is True
    assert receipt["gate"]["passed"] is False
    assert receipt["gate"]["all_payload_pack_unpack_roundtrips_exact"] is True
    assert receipt["gate"]["all_values_finite"] is True
    assert receipt["gate"]["observed_projected_complete_text_file_bpw"] <= 1.7
    layers = {item["layer"]: item for item in receipt["layers"]}
    assert set(layers) == {0, 32, 63}
    assert layers[0]["passed"] is True
    assert layers[32]["passed"] is False
    assert layers[32]["candidate_complete_block_validation_nmse"] > layers[32][
        "direct_complete_block_validation_nmse"
    ]
    assert layers[32]["relative_validation_nmse_reduction_vs_direct"] < 0
    assert layers[63]["passed"] is True

