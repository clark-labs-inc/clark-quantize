from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
MODULE_PATH = RUNTIME / "run_qwen38_27b_optimal_cell_scale_fresh_replication_v76.py"
PREREG = RUNTIME / "qwen38_27b_optimal_cell_scale_fresh_replication_preregistration_v76.json"
RECEIPTS = RUNTIME / "receipts/qwen38_27b_optimal_cell_scale_fresh_replication_v76"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUNTIME))
SPEC = importlib.util.spec_from_file_location("fresh_replication_v76", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_hybrid_correction_matches_dense_selected_cell_delta() -> None:
    generator = torch.Generator().manual_seed(7601)
    samples, tiles, rows, groups, width, selected = 5, 4, 3, 7, 6, 2
    inputs = torch.randn(samples, groups, width, generator=generator)
    base_codes = torch.randint(
        -1, 2, (tiles, rows, groups, width), generator=generator, dtype=torch.int8
    )
    base_scales = torch.rand(tiles, rows, groups, generator=generator).half()
    mask = torch.zeros(tiles, groups, dtype=torch.bool)
    mask[:, [1, 5]] = True
    int4_codes = torch.randint(
        -8,
        8,
        (tiles * selected * rows, 1, width),
        generator=generator,
        dtype=torch.int8,
    )
    int4_scales = torch.rand(
        tiles * selected * rows, 1, generator=generator
    ).half()

    actual = MODULE.frozen_hybrid_correction(
        inputs,
        base_codes,
        base_scales,
        mask,
        int4_codes,
        int4_scales,
        tile_chunk=2,
    )
    selected_groups = mask.nonzero(as_tuple=False)[:, 1].reshape(tiles, selected)
    int4_weight = int4_codes.reshape(tiles, selected, rows, width).float()
    int4_weight *= int4_scales.reshape(tiles, selected, rows).float().unsqueeze(-1)
    base_weight = base_codes.float() * base_scales.float().unsqueeze(-1)
    expected = torch.zeros(samples, tiles, rows)
    for tile in range(tiles):
        for slot, group in enumerate(selected_groups[tile]):
            expected[:, tile] += torch.einsum(
                "rd,nd->nr",
                int4_weight[tile, slot] - base_weight[tile, :, group],
                inputs[:, group],
            )
    torch.testing.assert_close(actual, expected)


def test_v76_preregistration_is_fresh_frozen_and_not_a_benchmark_claim() -> None:
    prereg = json.loads(PREREG.read_text())
    protocol = prereg["fresh_prompt_protocol"]
    gate = prereg["replication_gate"]
    frozen = prereg["frozen_candidate"]

    assert prereg["status"] == "frozen_before_fresh_prompt_tokenization_or_layer_propagation"
    assert prereg["scope"]["layer"] == 59
    assert len(protocol["prompts"]) == len(set(protocol["prompts"])) == 8
    assert protocol["benchmark_items_used"] is False
    assert protocol["fresh_activations_may_not_change_artifact_or_gate"] is True
    assert frozen["v75_receipt_sha256"] == "7028466f1798e425f2fa9a9574214aeef96864db754797590133115c7f8ef5e5"
    assert frozen["hybrid_payload_bytes"] == 26_307_264
    assert frozen["selected_cells"] == 10_240
    assert gate["minimum_relative_fresh_complete_mlp_nmse_improvement"] == 0.7
    assert gate["minimum_fresh_positive_output_tile_fraction"] == 0.95
    assert gate["minimum_improved_prompt_fraction"] == 0.75
    assert prereg["decision"]["no_breakthrough_claim"] is True


def test_v76_terminal_receipt_preserves_strict_near_miss() -> None:
    receipt_path = RECEIPTS / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    marker = json.loads((RECEIPTS / "EXPERIMENT_COMPLETE.json").read_text())

    assert receipt["complete"] is True
    assert receipt["terminal"] is True
    assert receipt["gate"] == {
        "decision": "reject_v75_as_nonreplicating_development_result",
        "passed": False,
    }
    assert receipt["fresh_data"]["artifact_fit_or_selection_operations_on_fresh_data"] == 0
    assert receipt["fresh_data"]["prior_prompt_overlap_count"] == 0
    assert receipt["checks"]["minimum_fresh_improvement_met"] is False
    assert receipt["metrics"]["relative_fresh_complete_mlp_nmse_improvement"] > 0.67
    assert receipt["metrics"]["relative_fresh_complete_mlp_nmse_improvement"] < 0.70
    assert receipt["metrics"]["fresh_positive_output_tile_fraction"] > 0.99
    assert receipt["metrics"]["improved_prompt_fraction"] == 1.0
    assert marker["passed"] is False
    assert marker["receipt_sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
