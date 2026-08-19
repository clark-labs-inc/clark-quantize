from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
PREREG = RUNTIME / "qwen38_27b_sensitivity_gated_residual_v37.json"
V36 = (
    RUNTIME
    / "receipts/qwen38_27b_calibration_span_residual_v36/terminal-decision.json"
)
RUNNER = RUNTIME / "run_qwen38_27b_sensitivity_gated_residual_v37.py"
SHELL_RUNNER = RUNTIME / "run_qwen38_27b_sensitivity_gated_residual_v37.sh"
COLLECTOR = RUNTIME / "collect_qwen38_27b_mixed_confirmation_cache_v1.py"


def test_v37_sensitivity_admission_is_derived_exactly_from_v36() -> None:
    prereg = json.loads(PREREG.read_text())
    v36 = json.loads(V36.read_text())
    threshold = prereg["sensitivity_rule"]["minimum_base_nmse_for_residual"]
    base_by_layer = {
        row["layer"]: row["base_validation_nmse"] for row in v36["grid"][0]["layers"]
    }
    active = sorted(layer for layer, value in base_by_layer.items() if value >= threshold)
    bypass = sorted(layer for layer, value in base_by_layer.items() if value < threshold)
    assert active == prereg["sensitivity_rule"]["active_layers"] == [0, 63]
    assert bypass == prereg["sensitivity_rule"]["exact_bypass_layers"] == [32]


def test_v37_rank_and_ridge_are_frozen_from_the_v36_efficiency_rule() -> None:
    prereg = json.loads(PREREG.read_text())
    v36 = json.loads(V36.read_text())
    qualifying = []
    for row in v36["grid"]:
        reductions = {item["layer"]: item["relative_reduction"] for item in row["layers"]}
        if reductions[0] >= 0.19 and reductions[63] >= 0.35:
            active_mean_nmse = sum(
                item["corrected_validation_nmse"]
                for item in row["layers"]
                if item["layer"] in (0, 63)
            ) / 2
            qualifying.append((row["rank"], active_mean_nmse, -row["ridge_fraction"], row))
    minimum_rank = min(row[0] for row in qualifying)
    selected = min(row for row in qualifying if row[0] == minimum_rank)[3]
    assert minimum_rank == prereg["fixed_adapter"]["rank"] == 16
    assert selected["ridge_fraction"] == prereg["fixed_adapter"]["ridge_fraction"] == 0.01


def test_v37_projection_matches_rank16_payload_and_mac_formulas() -> None:
    prereg = json.loads(PREREG.read_text())
    hidden = prereg["scope"]["hidden_size"]
    rank = prereg["fixed_adapter"]["rank"]
    per_layer_bytes = 2 * hidden * rank + 4 * rank + 2 * hidden
    per_layer_macs = 2 * hidden * rank
    assert prereg["storage_and_compute"]["persistent_bytes_per_active_layer"] == per_layer_bytes
    assert prereg["storage_and_compute"]["macs_per_active_layer_per_token"] == per_layer_macs
    projection = prereg["storage_and_compute"]["conservative_all_64_layers_projection"]
    assert projection["persistent_bytes"] == 64 * per_layer_bytes
    assert projection["macs_per_token"] == 64 * per_layer_macs
    assert projection["weight_traffic_fraction_vs_base_compute_image"] < 0.001


def test_v37_execution_opens_holdout_only_after_frozen_payloads() -> None:
    source = SHELL_RUNNER.read_text()
    calibration = source.index("--stage calibration")
    holdout = source.index(
        'taskset -c 0,2,4,6 "$python_bin" "$collector"', calibration
    )
    confirmation = source.index("--stage confirmation")
    assert calibration < holdout < confirmation
    assert "v36 holdout was unexpectedly materialized" in source
    assert "! -name artifacts.sha256.tmp" in source
    assert source.index('mv "$state_dir/artifacts.sha256.tmp"') < source.index(
        'mv "$state_dir/EXPERIMENT_COMPLETE.tmp"'
    )
    assert "EXPECTED_REPOSITORY_COMMIT" in source


def test_v37_has_no_new_grid_and_collector_accepts_its_frozen_status() -> None:
    prereg = json.loads(PREREG.read_text())
    runner = RUNNER.read_text()
    collector = COLLECTOR.read_text()
    assert prereg["fixed_adapter"]["new_grid_or_hyperparameter_search_forbidden"] is True
    assert "candidate_ranks" not in runner
    assert "for rank in" not in runner
    assert prereg["status"] in collector
