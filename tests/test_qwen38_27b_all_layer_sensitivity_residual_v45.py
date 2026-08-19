from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "run_qwen38_27b_all_layer_sensitivity_residual_v45.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("residual_v45", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_layers(module, *, active_reduction: float = 0.2):
    layers = []
    for layer in range(64):
        active = layer in (0, 63)
        base = 0.01 if active else 0.0005
        final = base * (1.0 - active_reduction) if active else base
        layers.append(
            {
                "layer": layer,
                "mode": "rank16_int8_residual" if active else "exact_bypass",
                "base_calibration_nmse": base,
                "base_validation_nmse": base,
                "final_calibration_nmse": final,
                "final_validation_nmse": final,
                "relative_validation_nmse_reduction": (
                    active_reduction if active else 0.0
                ),
                "persistent_payload_bytes": module.ADAPTER_BYTES if active else 0,
                "bypass_tensor_exact": None if active else True,
                "pilot_mode_exact": True if layer in module.PILOT_LAYERS else None,
                "pilot_payload_exact": True if layer in (0, 63) else None,
                "payload": f"layers/{layer:02d}/layer-{layer}.csr" if active else None,
                "terminal": True,
            }
        )
    return layers


def test_v45_fixed_sensitivity_rule() -> None:
    module = load_module()
    assert module.RANK == 16
    assert module.RIDGE_FRACTION == 0.01
    assert module.SENSITIVITY_THRESHOLD == 0.001
    assert module.classify_sensitivity(0.000999) == "exact_bypass"
    assert module.classify_sensitivity(0.001) == "rank16_int8_residual"
    with pytest.raises(ValueError):
        module.classify_sensitivity(float("nan"))


def test_v45_gate_passes_complete_fixed_example() -> None:
    module = load_module()
    gate = module.gate_from_layers(
        synthetic_layers(module),
        minimum_mean_active_validation_reduction=0.1,
        maximum_final_validation_nmse=0.05,
        maximum_residual_bytes=12_000_000,
    )
    assert gate["passed"] is True
    assert gate["observed_layers"] == 64
    assert gate["active_residual_layers"] == 2
    assert gate["exact_bypass_layers"] == 62


@pytest.mark.parametrize(
    "mutation",
    ["denominator", "active_regression", "bypass_not_exact", "pilot_drift"],
)
def test_v45_gate_rejects_changed_evidence(mutation) -> None:
    module = load_module()
    layers = synthetic_layers(module)
    if mutation == "denominator":
        layers.pop()
    elif mutation == "active_regression":
        layers[0]["final_validation_nmse"] = 0.02
    elif mutation == "bypass_not_exact":
        layers[1]["bypass_tensor_exact"] = False
    else:
        layers[32]["pilot_mode_exact"] = False
    if mutation == "denominator":
        with pytest.raises(ValueError, match="denominator"):
            module.gate_from_layers(
                layers,
                minimum_mean_active_validation_reduction=0.1,
                maximum_final_validation_nmse=0.05,
                maximum_residual_bytes=12_000_000,
            )
    else:
        gate = module.gate_from_layers(
            layers,
            minimum_mean_active_validation_reduction=0.1,
            maximum_final_validation_nmse=0.05,
            maximum_residual_bytes=12_000_000,
        )
        assert gate["passed"] is False


def test_v45_is_inert_without_post_v44_preregistration() -> None:
    source = SCRIPT.read_text()
    assert "qwen38_27b_all_layer_sensitivity_residual_v45.json" in source
    assert not (
        SCRIPT.parent / "qwen38_27b_all_layer_sensitivity_residual_v45.json"
    ).exists()
    assert '"frozen_after_v44_pass_before_residual_fit"' in source
    assert 'prereg.get("parent_training", {}).get("receipt_sha256")' in source
    assert 'prereg.get("frozen_inputs", {}).get("runner_sha256")' in source


def test_v45_uses_only_calibration_for_fit_and_validation_for_selection() -> None:
    source = SCRIPT.read_text()
    fit_start = source.index("fit_float_residual_adapter(")
    fit_end = source.index("\n            )", fit_start)
    fit = source[fit_start:fit_end]
    assert "calibration_inputs" in fit
    assert "calibration_candidate" in fit
    assert "calibration_teacher" in fit
    assert "validation" not in fit
    assert "classify_sensitivity(base_validation_nmse)" in source
    assert 'context["validation_flat_rows"]' in source
    assert "selected_rows=calibration_rows" in source
    assert "candidate_mlp.selected_rows = validation_rows.long().contiguous()" in source
    assert source.count("qat.forward_with_grad(") >= 3
    assert "fresh_confirmation" not in source
    for benchmark in ("CUTE", "MATH-500", "MMLU", "WikiText"):
        assert benchmark not in source


def test_v45_preserves_exact_bypass_and_pilot_payload_reproduction() -> None:
    source = SCRIPT.read_text()
    assert 'mode == "rank16_int8_residual"' in source
    assert 'mode == "exact_bypass"' in source
    assert "torch.equal(corrected_calibration, calibration_candidate)" in source
    assert "v45 pilot residual payload reproduction changed" in source
    assert "v45 pilot residual source payload changed" in source
    assert "v45 pilot sensitivity mode reproduction changed" in source
