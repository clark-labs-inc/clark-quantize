from __future__ import annotations

import json
from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_all_mlp_cq3_template_preregistration_v55.json"
RUNNER = RUNTIME / "run_qwen38_27b_all_mlp_cq3_template_v55.sh"
LAUNCHER = RUNTIME / "launch_qwen38_27b_all_mlp_cq3_template_v55.sh"


def test_v55_regex_is_exactly_all_192_mlp_tensors() -> None:
    prereg = json.loads(PREREG.read_text())
    pattern = re.compile(prereg["tensor_scope"]["target_regex"])
    expected = {
        f"blk.{layer}.ffn_{kind}.weight"
        for layer in range(64)
        for kind in ("down", "gate", "up")
    }

    assert len(expected) == 192
    assert all(pattern.fullmatch(name) for name in expected)
    assert not pattern.fullmatch("blk.64.ffn_up.weight")
    assert not pattern.fullmatch("blk.0.attn_output.weight")


def test_v55_expected_size_is_derived_from_exact_payload_delta() -> None:
    prereg = json.loads(PREREG.read_text())
    baseline = prereg["baseline"]["bytes"]
    q4_per_tensor = 50_135_040
    cq3_per_tensor = 23_674_880
    target_count = prereg["tensor_scope"]["target_count"]
    expected = baseline - target_count * (q4_per_tensor - cq3_per_tensor)

    assert expected == 13_893_519_712
    assert prereg["output"]["expected_bytes"] == expected
    assert prereg["output"]["expected_saved_bytes_vs_baseline"] == baseline - expected


def test_v55_runner_is_atomic_authenticated_and_parent_serialized() -> None:
    runner = RUNNER.read_text()
    launcher = LAUNCHER.read_text()

    assert 'partial="${candidate}.partial"' in runner
    assert 'mv "$partial" "$candidate"' in runner
    assert 'rm -f -- "$partial"' in runner
    assert runner.index('PYTHONPATH="$llama_source/gguf-py"') < runner.index(
        'mv "$partial" "$candidate"'
    )
    assert "--expected-target-count 192" in runner
    assert 'scope.get("non_target_count") != 659' in runner
    assert "all_non_target_types_sizes_and_payloads_identical" in runner
    assert "sha256sum -c" in runner
    assert "recovered authenticated v55 completion marker" in runner
    for parent in (
        "quant-qwen38-27b-all-layer-mixed-trained-v44.service",
        "quant-qwen38-27b-all-layer-sensitivity-residual-v45.service",
    ):
        assert parent in runner
        assert parent in launcher
    assert "MemorySwapMax=0" in launcher
    assert "AllowedCPUs=0,2,4,6" in launcher


def test_v55_claim_boundary_forbids_template_promotion() -> None:
    prereg = json.loads(PREREG.read_text())

    assert prereg["scope"]["active_model_only"] == "Qwen/Qwen3.8-27B"
    assert prereg["scope"]["qwen_4b_compute_forbidden"] is True
    assert "not trained-model quality" in prereg["claim_boundary"]
    assert "scientific-breakthrough" in prereg["claim_boundary"]
