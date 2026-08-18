from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "run_qwen38_27b_mixed_confirmation_v1.py"
)


def load_module():
    runtime = SCRIPT.parent
    sys.path.insert(0, str(runtime))
    try:
        spec = importlib.util.spec_from_file_location("mixed_confirmation", SCRIPT)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(runtime))


def test_mixed_confirmation_is_qwen38_only_and_has_no_training_path() -> None:
    module = load_module()
    source = SCRIPT.read_text()
    assert module.MODEL_ID == "Qwen/Qwen3.8-27B"
    assert module.TARGET_LAYERS == (0, 32, 63)
    assert "Qwen3.5-4B" not in source
    assert "train_row_shard" not in source
    assert "trust_region" not in source
    assert 'cache["validation_flat_rows"]' not in source
    assert 'cache["confirmation_flat_rows"]' in source
    assert source.count('"step_zero_direct_mixed", "final_trained_mixed"') == 1


def test_mixed_confirmation_gate_passes_only_both_frozen_numeric_rules() -> None:
    module = load_module()
    layers = [
        {
            "layer": layer,
            "direct_confirmation_complete_block_nmse": direct,
            "trained_confirmation_complete_block_nmse": trained,
            "relative_confirmation_nmse_reduction_vs_direct": (
                direct - trained
            )
            / direct,
        }
        for layer, direct, trained in (
            (0, 0.04, 0.03),
            (32, 0.02, 0.019),
            (63, 0.06, 0.049),
        )
    ]
    gate = module.confirmation_gate(
        layers, maximum_final_nmse=0.05, minimum_mean_relative_reduction=0.01
    )
    assert gate["passed"] is True
    assert gate["exact_candidates_scored_per_layer"] == 2
    regressed = [dict(item) for item in layers]
    regressed[0]["trained_confirmation_complete_block_nmse"] = 0.041
    assert (
        module.confirmation_gate(
            regressed,
            maximum_final_nmse=0.05,
            minimum_mean_relative_reduction=0.01,
        )["passed"]
        is False
    )
    above_maximum = [dict(item) for item in layers]
    above_maximum[2]["trained_confirmation_complete_block_nmse"] = 0.051
    assert (
        module.confirmation_gate(
            above_maximum,
            maximum_final_nmse=0.05,
            minimum_mean_relative_reduction=0.01,
        )["passed"]
        is False
    )


def test_mixed_confirmation_is_terminal_and_selection_free() -> None:
    source = SCRIPT.read_text()
    assert '"checkpoint_or_hyperparameter_selection_performed": False' in source
    assert "exact_candidates_scored_per_layer" in source
    assert "maximum_final_fresh_confirmation_complete_block_nmse_per_layer" in source
    assert "minimum_mean_relative_fresh_confirmation_nmse_reduction_vs_direct_mixed" in source
    assert 'raise SystemExit(0 if gate["passed"] else 2)' in source
    assert "no complete-model conversion" in source
