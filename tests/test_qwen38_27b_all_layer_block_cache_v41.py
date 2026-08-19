from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
SCRIPT = RUNTIME / "collect_qwen38_27b_all_layer_block_cache_v41.py"
PREREG = RUNTIME / "qwen38_27b_all_layer_block_cache_v41.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("all_layer_cache_v41", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preregistration() -> dict[str, object]:
    return json.loads(PREREG.read_text())


def synthetic_layers(count: int, calibration_rows: int, validation_rows: int):
    layers = {}
    for index in range(count):
        layer = {
            "type": (
                "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
            )
        }
        for split, rows in (
            ("calibration", calibration_rows),
            ("validation", validation_rows),
        ):
            for endpoint in ("input", "output"):
                layer[f"{split}_{endpoint}"] = torch.zeros(
                    rows, 5120, dtype=torch.bfloat16
                )
        layers[str(index)] = layer
    return layers


def test_v41_is_frozen_for_all_qwen38_layers_only() -> None:
    data = preregistration()
    assert data["status"] == "frozen_before_all_layer_cache_execution"
    assert data["scope"]["active_model_only"] == "Qwen/Qwen3.8-27B"
    assert data["scope"]["target_layers"] == list(range(64))
    assert data["scope"]["qwen_4b_compute_forbidden"] is True
    assert data["data_protocol"]["fresh_holdout_rows"] == 0


def test_v41_stores_only_selected_exact_rows() -> None:
    data = preregistration()
    protocol = data["data_protocol"]
    assert protocol["calibration_rows_per_layer"] == 64
    assert protocol["internal_validation_rows_per_layer"] == 32
    assert protocol["stored_dtype"] == "BF16"
    assert protocol["full_sequence_activations_persisted"] is False
    assert len(protocol["stored_tensors_per_layer"]) == 4
    source = SCRIPT.read_text()
    assert "selected_rows(hidden, calibration_rows)" in source
    assert "selected_rows(hidden, validation_rows)" in source
    assert '"input_ids": token_ids' not in source
    assert '"attention_mask": attention_mask' not in source


def test_v41_selected_rows_preserve_exact_values() -> None:
    module = load_module()
    source = torch.arange(4 * 3 * 8, dtype=torch.float32).reshape(4, 3, 8)
    rows = torch.tensor([0, 5, 11], dtype=torch.int64)
    observed = module.selected_rows(source, rows)
    expected = source.reshape(-1, 8).index_select(0, rows)
    assert torch.equal(observed, expected)
    assert observed.is_contiguous()


def test_v41_layer_cache_validation_is_fail_closed() -> None:
    module = load_module()
    layers = synthetic_layers(4, 2, 1)
    module.validate_layer_cache(
        layers, expected_count=4, calibration_rows=2, validation_rows=1
    )
    layers["3"]["type"] = "linear_attention"
    with pytest.raises(ValueError, match="layer type"):
        module.validate_layer_cache(
            layers, expected_count=4, calibration_rows=2, validation_rows=1
        )


def test_v41_checkpoint_is_bounded_and_authenticated() -> None:
    data = preregistration()
    assert data["execution"]["checkpoint_interval_layers"] == 8
    assert data["gate"]["maximum_cache_bytes"] == 200_000_000
    assert data["gate"]["maximum_checkpoint_bytes"] == 220_000_000
    source = SCRIPT.read_text()
    assert "next_layer % CHECKPOINT_INTERVAL" in source
    assert "parent_cache_sha256" in source
    assert "pilot.atomic_torch_save(" in source
    assert "args.checkpoint.unlink()" in source
    assert source.index("pilot.atomic_json(receipt, args.receipt)") < source.index(
        "args.checkpoint.unlink()"
    )


def test_v41_pilot_layers_must_reproduce_bit_exactly() -> None:
    data = preregistration()
    assert data["gate"]["pilot_layers_must_be_bit_exact"] == [0, 32, 63]
    source = SCRIPT.read_text()
    assert "pilot block input reproduction changed" in source
    assert "pilot block output reproduction changed" in source
    assert source.count("torch.equal(") >= 3


def test_v41_frozen_inputs_match_hashes() -> None:
    for descriptor in preregistration()["frozen_inputs"].values():
        path = RUNTIME / descriptor["path"]
        assert path.is_file()
        assert sha256(path) == descriptor["sha256"]


def test_v41_claim_boundary_forbids_quality_or_speed_claims() -> None:
    data = preregistration()
    assert "does not establish approximation quality" in data["claim_boundary"]
    rules = "\n".join(data["promotion_rules"])
    assert "not a quantized model" in rules
    assert "fresh confirmation remains sealed" in rules
