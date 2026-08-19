from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
MODULE_PATH = RUNTIME / "run_qwen38_27b_calibration_optimal_cell_scale_v75.py"
PREREG = RUNTIME / "qwen38_27b_calibration_optimal_cell_scale_preregistration_v75.json"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUNTIME))
SPEC = importlib.util.spec_from_file_location("optimal_cell_scale_v75", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_conditional_scale_fit_updates_exact_residual_and_selects_unique_cells() -> None:
    generator = torch.Generator().manual_seed(7501)
    code = torch.randn(9, 3, 5, 2, generator=generator)
    base = 0.1 * torch.randn(9, 3, 5, 2, generator=generator)
    planted_scale = torch.tensor([0.5, 0.75])
    residual = planted_scale * code[:, :, 2, :] - base[:, :, 2, :]
    selected, scales, final_residual, history = (
        MODULE.greedy_select_optimal_scale_cells(
            code, base, residual, steps=2
        )
    )
    correction = MODULE.selected_optimal_scale_correction(
        code, base, selected, scales
    )

    assert selected.shape == (3, 2)
    assert scales.shape == (3, 2, 2)
    assert scales.dtype == torch.float16
    assert all(len(set(row)) == 2 for row in selected.tolist())
    assert selected[:, 0].tolist() == [2, 2, 2]
    torch.testing.assert_close(final_residual, residual - correction)
    assert len(history) == 2


def test_installed_scales_change_only_selected_x8_cells() -> None:
    parent = torch.ones(MODULE.HIDDEN, MODULE.INPUT_GROUPS, dtype=torch.float16)
    selected = torch.zeros(MODULE.OUTPUT_TILES, 1, dtype=torch.long)
    scales = torch.full(
        (MODULE.OUTPUT_TILES, 1, MODULE.TILE_ROWS),
        2.0,
        dtype=torch.float16,
    )
    installed = MODULE.install_selected_scales(parent, selected, scales)
    tiled = installed.reshape(
        MODULE.OUTPUT_TILES, MODULE.TILE_ROWS, MODULE.INPUT_GROUPS
    )

    assert torch.equal(tiled[:, :, 0], torch.full_like(tiled[:, :, 0], 2.0))
    assert torch.equal(tiled[:, :, 1:], torch.ones_like(tiled[:, :, 1:]))


def test_v75_preregistration_freezes_same_rate_development_only_screen() -> None:
    prereg = json.loads(PREREG.read_text())
    status = prereg["data_status"]
    cell = prereg["int4_cell"]
    layout = prereg["canonical_layout"]

    assert prereg["status"] == "frozen_development_screen_before_execution"
    assert prereg["scope"]["layer"] == 59
    assert status["development_confirmation_is_architecture_contaminated"] is True
    assert status["promotion_or_scientific_claim_from_v75_forbidden"] is True
    assert cell["greedy_steps_per_output_tile"] == 16
    assert cell["selected_cells"] == 10_240
    assert cell["additional_runtime_bytes_vs_v74"] == 0
    assert cell["additional_runtime_operations_vs_v74"] == 0
    assert cell["sweep"] is False
    assert layout["hybrid_down_payload_bytes"] == 26_307_264
    assert layout["hybrid_three_projection_mlp_payload_bytes"] == 73_657_024
    assert layout["relative_full_mlp_payload_increase"] < 0.04
    assert layout["high_precision_weight_materialization_at_runtime"] is False
