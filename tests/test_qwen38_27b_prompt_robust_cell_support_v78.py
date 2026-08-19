from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
MODULE_PATH = RUNTIME / "run_qwen38_27b_prompt_robust_cell_support_v78.py"
PREREG = RUNTIME / "qwen38_27b_prompt_robust_cell_support_preregistration_v78.json"
RECEIPTS = RUNTIME / "receipts/qwen38_27b_prompt_robust_cell_support_v78"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUNTIME))
SPEC = importlib.util.spec_from_file_location("prompt_robust_support_v78", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_prompt_robust_selection_prefers_stable_environment_gain() -> None:
    samples, tiles, groups, rows = 8, 1, 3, 1
    owners = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    code = torch.zeros(samples, tiles, groups, rows)
    base = torch.zeros_like(code)
    residual = torch.ones(samples, tiles, rows)
    code[:4, 0, 0, 0] = 1.0
    code[4:, 0, 0, 0] = -1.0
    code[:, 0, 1, 0] = 0.5
    code[:, 0, 2, 0] = 0.1

    selected, scales, final, history = MODULE.greedy_select_prompt_robust_cells(
        code, base, residual, owners, steps=1, risk_radius=1.0
    )

    assert selected.tolist() == [[1]]
    assert scales.shape == (1, 1, 1)
    assert final.square().sum() < residual.square().sum()
    assert history[0]["minimum_selected_robust_score"] > 0


def test_environment_owners_remaps_disjoint_prompt_ids() -> None:
    raw = torch.tensor([8, 8, 9, 11, 11])
    assert MODULE.environment_owners(raw).tolist() == [0, 0, 1, 2, 2]


def test_v78_preregistration_freezes_same_cost_robust_development_screen() -> None:
    prereg = json.loads(PREREG.read_text())
    estimator = prereg["estimator"]
    layout = prereg["canonical_layout"]
    gate = prereg["development_gate"]

    assert prereg["status"] == "frozen_development_screen_before_execution"
    assert prereg["scope"]["layer"] == 59
    assert estimator["risk_radius"] == 1.0
    assert estimator["greedy_steps_per_output_tile"] == 16
    assert estimator["hyperparameter_sweep"] is False
    assert estimator["v42_validation_used_for_fit_or_selection"] is False
    assert estimator["v76_prompts_used_for_fit_or_selection"] is False
    assert layout["selected_cells"] == 10_240
    assert layout["hybrid_down_payload_bytes"] == 26_307_264
    assert layout["additional_runtime_bytes_vs_v75"] == 0
    assert layout["additional_runtime_operations_vs_v75"] == 0
    assert gate["minimum_relative_v76_development_complete_mlp_nmse_improvement"] == 0.7
    assert gate["minimum_v76_median_prompt_directional_capacity"] == 0.3
    assert prereg["data_status"]["promotion_or_scientific_claim_from_v78_forbidden"] is True
    assert prereg["decision"]["no_breakthrough_claim"] is True


def test_v78_terminal_receipt_rejects_forced_robust_support() -> None:
    receipt = json.loads((RECEIPTS / "receipt.json").read_text())

    assert receipt["complete"] is True
    assert receipt["terminal"] is True
    assert receipt["gate"] == {
        "decision": "reject_mean_minus_std_prompt_robust_cell_support",
        "passed": False,
    }
    assert receipt["metrics"]["fit"]["relative_improvement"] > 0.79
    assert receipt["metrics"]["v42_validation_development"]["relative_improvement"] < 0.70
    assert receipt["metrics"]["v76_development"]["relative_improvement"] < 0.57
    assert receipt["metrics"]["v76_development"]["improved_prompt_fraction"] == 0.75
    assert receipt["estimator"]["zero_selected_scales"] > 6_000
    assert receipt["storage"]["additional_runtime_bytes_vs_v75"] == 0
