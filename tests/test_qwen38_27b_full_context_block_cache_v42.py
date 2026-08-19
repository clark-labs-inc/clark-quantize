from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
SCRIPT = RUNTIME / "collect_qwen38_27b_full_context_block_cache_v42.py"
PREREG = RUNTIME / "qwen38_27b_full_context_block_cache_v42.json"
RECEIPTS = RUNTIME / "receipts/qwen38_27b_full_context_block_cache_v42"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("full_context_cache_v42", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preregistration() -> dict[str, object]:
    return json.loads(PREREG.read_text())


def synthetic_context(module, preregistration_sha256: str):
    return {
        "schema": module.CONTEXT_SCHEMA,
        "model": module.MODEL_ID,
        "revision": module.MODEL_REVISION,
        "preregistration_sha256": preregistration_sha256,
        "parent_cache_sha256": module.EXPECTED_PARENT_CACHE_SHA256,
        "calibration_texts": [f"cal-{index}" for index in range(8)],
        "validation_texts": [f"val-{index}" for index in range(4)],
        "input_ids": torch.zeros(12, 64, dtype=torch.int64),
        "attention_mask": torch.ones(12, 64, dtype=torch.int8),
        "text_position_ids": torch.arange(64, dtype=torch.int64).repeat(12, 1),
        "calibration_flat_rows": torch.arange(64, dtype=torch.int64),
        "validation_flat_rows": torch.arange(32, dtype=torch.int64),
    }


def test_v42_is_frozen_for_full_context_qwen38_only() -> None:
    data = preregistration()
    assert data["status"] == "frozen_before_full_context_execution"
    assert data["scope"]["active_model_only"] == "Qwen/Qwen3.8-27B"
    assert data["scope"]["target_layers"] == list(range(64))
    assert data["scope"]["qwen_4b_compute_forbidden"] is True
    assert data["data_protocol"]["full_sequence_activations_persisted"] is True
    assert data["data_protocol"]["full_context_shape_per_endpoint"] == [
        12,
        64,
        5120,
    ]


def test_v42_explicitly_rejects_the_selected_row_design() -> None:
    data = preregistration()
    correction = data["design_correction"]
    assert correction["rejected_experiment"].endswith("v41")
    assert correction["reuse_of_v41_checkpoint_forbidden"] is True
    assert "complete 64-token sequence" in correction["fault"]
    assert "every BF16 token state" in correction["correction"]
    source = SCRIPT.read_text()
    assert "selected_rows(" not in source
    assert '"input": exact_input' in source
    assert '"output": exact_output' in source


def test_v42_context_identity_is_tensor_exact_and_fail_closed() -> None:
    module = load_module()
    prereg_sha = "a" * 64
    expected = synthetic_context(module, prereg_sha)
    observed = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in expected.items()
    }
    module.validate_context(observed, prereg_sha)
    assert module.contexts_equal(expected, observed)
    observed["input_ids"][0, 0] = 1
    assert not module.contexts_equal(expected, observed)
    observed.pop("validation_texts")
    assert not module.contexts_equal(expected, observed)


def test_v42_shards_require_complete_bf16_sequence_endpoints() -> None:
    module = load_module()
    prereg_sha = "b" * 64
    producer_commit = "c" * 40
    shard = {
        "schema": module.SHARD_SCHEMA,
        "model": module.MODEL_ID,
        "revision": module.MODEL_REVISION,
        "preregistration_sha256": prereg_sha,
        "parent_cache_sha256": module.EXPECTED_PARENT_CACHE_SHA256,
        "producer_commit": producer_commit,
        "layer": 0,
        "type": "linear_attention",
        "input": torch.zeros(12, 64, 5120, dtype=torch.bfloat16),
        "output": torch.zeros(12, 64, 5120, dtype=torch.bfloat16),
    }
    module.validate_shard_tensors(
        shard,
        0,
        preregistration_sha256=prereg_sha,
        producer_commit=producer_commit,
    )
    shard["producer_commit"] = "d" * 40
    with pytest.raises(ValueError, match="identity"):
        module.validate_shard_tensors(
            shard,
            0,
            preregistration_sha256=prereg_sha,
            producer_commit=producer_commit,
        )


def test_v42_resume_prefix_rejects_gaps(tmp_path: Path) -> None:
    module = load_module()
    (tmp_path / "01.pt").write_bytes(b"incomplete")
    (tmp_path / "01.json").write_text("{}")
    with pytest.raises(RuntimeError, match="prefix has a gap"):
        module.scan_layer_prefix(
            tmp_path,
            preregistration_sha256="e" * 64,
            producer_commit="f" * 40,
        )


def test_v42_each_layer_is_an_atomic_terminal_checkpoint() -> None:
    data = preregistration()
    execution = data["execution"]
    assert execution["checkpoint_interval_layers"] == 1
    assert "contiguous descriptor prefix" in execution["checkpoint_unit"]
    source = SCRIPT.read_text()
    assert source.index("pilot.atomic_torch_save(shard, shard_path)") < source.index(
        "pilot.atomic_json(descriptor, descriptor_path)"
    )
    assert "if shard_path.exists() != descriptor_path.exists()" in source
    assert "if gap_seen:" in source


def test_v42_terminal_validation_rehashes_context_and_all_shards() -> None:
    source = SCRIPT.read_text()
    function = source[source.index("def validate_final(") : source.index("def main()")]
    assert "scan_layer_prefix(" in function
    assert "pilot.sha256_file(context_path)" in function
    assert 'records != receipt["layers"]' in function
    assert 'marker.get("total_shard_bytes")' in function


def test_v42_storage_and_resource_bounds_are_frozen() -> None:
    data = preregistration()
    assert data["gate"]["maximum_individual_shard_bytes"] == 17_000_000
    assert data["gate"]["maximum_total_shard_bytes"] == 1_100_000_000
    assert data["resources"]["memory_max_bytes"] == 31_000_000_000
    assert data["resources"]["memory_swap_max_bytes"] == 0
    assert data["resources"]["maximum_restarts"] == 2


def test_v42_frozen_inputs_match_hashes() -> None:
    for descriptor in preregistration()["frozen_inputs"].values():
        path = RUNTIME / descriptor["path"]
        assert path.is_file()
        assert sha256(path) == descriptor["sha256"]


def test_v42_claim_boundary_forbids_quality_or_speed_claims() -> None:
    data = preregistration()
    assert "does not establish approximation quality" in data["claim_boundary"]
    rules = "\n".join(data["promotion_rules"])
    assert "not a quantized model" in rules
    assert "fresh confirmation remains sealed" in rules


def test_v42_terminal_receipts_are_authenticated_and_complete() -> None:
    receipt = json.loads((RECEIPTS / "receipt.json").read_text())
    marker = json.loads((RECEIPTS / "complete.json").read_text())
    execution = json.loads((RECEIPTS / "execution.json").read_text())
    module_schema = "qwen38_27b_full_context_block_cache_receipt_v42"
    assert receipt["schema"] == module_schema
    assert module_schema.endswith("v42")
    assert receipt["complete"] is True and receipt["terminal"] is True
    assert len(receipt["layers"]) == 64
    assert [record["layer"] for record in receipt["layers"]] == list(range(64))
    assert all(
        receipt["layers"][index]["output"]["sha256"]
        == receipt["layers"][index + 1]["input"]["sha256"]
        for index in range(63)
    )
    assert marker["receipt_sha256"] == sha256(RECEIPTS / "receipt.json")
    assert execution["terminal_artifacts"]["remote_receipt"]["sha256"] == sha256(
        RECEIPTS / "receipt.json"
    )
    assert execution["terminal_artifacts"]["remote_completion_marker"][
        "sha256"
    ] == sha256(RECEIPTS / "complete.json")


def test_v42_terminal_decision_admits_training_only() -> None:
    execution = json.loads((RECEIPTS / "execution.json").read_text())
    decision = json.loads((RECEIPTS / "terminal-decision.json").read_text())
    assert execution["systemd"]["restarts"] == 0
    assert execution["systemd"]["journal_started_layer_events"] == 64
    assert execution["systemd"]["journal_terminal_layer_events"] == 64
    assert execution["gate"]["passed"] is True
    assert decision["execution_receipt_sha256"] == sha256(
        RECEIPTS / "execution.json"
    )
    assert decision["decision"] == (
        "admit_full_context_cache_to_frozen_all_layer_training_only"
    )
    assert len(decision["forbidden_claims"]) == 4
