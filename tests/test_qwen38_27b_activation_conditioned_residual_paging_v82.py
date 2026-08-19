from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

from clark_quantize.qwen38_mixed_cq3_runtime import StreamedCq3Linear


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
MODULE_PATH = RUNTIME / "run_qwen38_27b_activation_conditioned_residual_paging_v82.py"
PREREG = RUNTIME / "qwen38_27b_activation_conditioned_residual_paging_preregistration_v82.json"
LAUNCHER = RUNTIME / "launch_qwen38_27b_activation_conditioned_residual_paging_v82.sh"
SPEC = importlib.util.spec_from_file_location("residual_paging_v82", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v82_is_a_cross_layer_fixed_budget_upper_bound() -> None:
    data = json.loads(PREREG.read_text())
    assert data["status"] == "frozen_before_execution"
    assert data["scope"]["layers"] == [1, 23, 45]
    assert data["scope"]["operator_count"] == 9
    assert data["scope"]["no_layer_count_rank_seed_or_budget_sweep"] is True
    assert data["upper_bound_status"]["residual_values"].startswith("exact BF16")
    assert MODULE.LAYERS == (1, 23, 45)


def test_v82_fixed_group_budget_matches_gate_up_and_down_shapes() -> None:
    assert MODULE.fixed_group_count(5120) == 4
    assert MODULE.fixed_group_count(17408) == 14
    assert MODULE.fixed_group_count(5120) / 40 == 0.1
    assert MODULE.fixed_group_count(17408) / 136 < 0.103


def test_v82_selector_recovers_only_bound_selected_residual_groups() -> None:
    rows, columns = 3, 1280
    codes = torch.zeros(rows, columns, dtype=torch.int8)
    scales = torch.ones(rows, columns // 128, dtype=torch.float16)
    base = StreamedCq3Linear(codes, scales, row_shard=2, output_dtype=torch.float32)
    source = torch.zeros(rows, columns)
    source[:, :128] = 2.0
    source[:, 128:256] = 1.0
    inputs = torch.zeros(2, columns)
    inputs[0, :128] = 1.0
    inputs[1, 128:256] = 1.0
    result = MODULE.activation_conditioned_residual_linear(source, base, inputs, row_shard=2)
    assert result["metrics"]["selected_groups_per_input"] == 1
    assert result["selected"].tolist() == [[0], [1]]
    assert torch.equal(result["candidate"], result["exact"])
    assert result["metrics"]["relative_nmse_reduction"] == 1.0


def test_v82_gate_requires_large_recovery_before_any_fresh_replication() -> None:
    data = json.loads(PREREG.read_text())
    assert data["gate"] == {
        "minimum_operator_nmse_reduction_each": 0.5,
        "minimum_complete_mlp_nmse_reduction_each_layer": 0.7,
        "minimum_median_complete_mlp_nmse_reduction": 0.8,
        "maximum_residual_group_fraction_each": 0.103,
        "all_exact_layer_replays_required": True,
        "all_values_finite_required": True,
    }
    assert "do not tune" in data["next_if_failed"]
    launcher = LAUNCHER.read_text()
    assert "AllowedCPUs=0,2,4,6" in launcher
    assert "MemoryMax=30000000000" in launcher
    assert "MemorySwapMax=0" in launcher
