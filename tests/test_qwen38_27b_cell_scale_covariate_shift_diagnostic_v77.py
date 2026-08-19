from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
MODULE_PATH = RUNTIME / "run_qwen38_27b_cell_scale_covariate_shift_diagnostic_v77.py"
PREREG = RUNTIME / "qwen38_27b_cell_scale_covariate_shift_diagnostic_preregistration_v77.json"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUNTIME))
SPEC = importlib.util.spec_from_file_location("cell_scale_diagnostic_v77", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_vector_geometry_recovers_scale_mismatch_and_capacity() -> None:
    residual = torch.tensor([2.0, -4.0, 1.0])
    correction = residual / 2
    geometry = MODULE.vector_geometry(residual, correction)

    assert abs(geometry["residual_correction_cosine"] - 1.0) < 1e-12
    assert abs(geometry["unconstrained_optimal_scalar"] - 2.0) < 1e-12
    assert abs(geometry["nonnegative_scalar_directional_capacity"] - 1.0) < 1e-12
    assert abs(geometry["unconstrained_optimal_scalar_relative_sse_improvement"] - 1.0) < 1e-12
    assert abs(geometry["deployed_relative_sse_improvement"] - 0.75) < 1e-12
    assert abs(geometry["deployed_fraction_of_nonnegative_directional_capacity"] - 0.75) < 1e-12


def test_per_tile_optimal_scalars_are_independent() -> None:
    residual = torch.tensor([[[2.0], [3.0]], [[4.0], [6.0]]])
    correction = torch.tensor([[[1.0], [3.0]], [[2.0], [6.0]]])
    actual = MODULE.per_tile_optimal_scalars(residual, correction)
    torch.testing.assert_close(actual, torch.tensor([2.0, 1.0], dtype=torch.float64))


def test_v77_preregistration_is_diagnostic_only_and_rule_is_frozen() -> None:
    prereg = json.loads(PREREG.read_text())
    frozen = prereg["frozen_diagnostics"]
    rules = prereg["classification_rule"]

    assert prereg["status"] == "frozen_postmortem_before_vector_diagnostics"
    assert prereg["parent_result"]["gate_passed"] is False
    assert prereg["parent_result"]["fresh_data_is_now_architecture_contaminated"] is True
    assert frozen["no_parameter_fit_or_artifact_write"] is True
    assert frozen["diagnostic_scalars_are_not_deployable_candidates"] is True
    assert rules["minimum_median_prompt_directional_capacity_for_magnitude_shift"] == 0.5
    assert rules["maximum_median_deployed_fraction_of_capacity_for_magnitude_shift"] == 0.8
    assert prereg["decision"]["new_untouched_prompts_required_for_any_future_promotion"] is True
    assert prereg["decision"]["no_breakthrough_claim"] is True
