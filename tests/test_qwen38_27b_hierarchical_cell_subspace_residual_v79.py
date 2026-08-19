from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
MODULE_PATH = RUNTIME / "run_qwen38_27b_hierarchical_cell_subspace_residual_v79.py"
PREREG = RUNTIME / "qwen38_27b_hierarchical_cell_subspace_residual_preregistration_v79.json"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUNTIME))
SPEC = importlib.util.spec_from_file_location("hierarchical_residual_v79", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_relative_improvement_uses_shared_exact_denominator() -> None:
    exact = torch.tensor([[2.0, -1.0]])
    baseline = torch.zeros_like(exact)
    candidate = torch.tensor([[1.0, -0.5]])
    base, corrected, improvement = MODULE.relative_improvement(
        baseline, candidate, exact
    )
    assert base == 1.0
    assert corrected == 0.25
    assert improvement == 0.75


def test_v79_adapter_size_and_compute_are_exact() -> None:
    assert MODULE.ADAPTER_BYTES == 370_752
    assert MODULE.RANK * (MODULE.INTERMEDIATE + MODULE.HIDDEN) == 360_448


def test_v79_preregistration_freezes_nonredundant_hierarchical_composition() -> None:
    prereg = json.loads(PREREG.read_text())
    adapter = prereg["adapter"]
    composition = prereg["canonical_composition"]
    gate = prereg["development_gate"]

    assert prereg["status"] == "frozen_development_screen_before_execution"
    assert prereg["scope"]["layer"] == 59
    assert prereg["scope"]["base_candidate"].startswith("frozen v75")
    assert adapter["rank"] == 16
    assert adapter["ridge_fraction"] == 0.01
    assert adapter["rank_or_ridge_sweep"] is False
    assert adapter["persistent_bytes"] == 370_752
    assert adapter["macs_per_token"] == 360_448
    assert composition["composed_three_projection_mlp_payload_bytes"] == 74_027_776
    assert composition["relative_full_mlp_payload_increase"] < 0.043
    assert gate["minimum_relative_v76_development_complete_mlp_nmse_improvement"] == 0.72
    assert gate["minimum_absolute_v76_improvement_over_v75"] == 0.04
    assert prereg["data_status"]["promotion_or_scientific_claim_from_v79_forbidden"] is True
    assert prereg["decision"]["no_breakthrough_claim"] is True
