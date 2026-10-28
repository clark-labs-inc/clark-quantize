from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

RUNTIME = (
    Path(__file__).resolve().parents[1]
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
)
SOURCE = RUNTIME / "materialize_qwen38_27b_all_layer_trained_cq3_v46.py"
SPEC = importlib.util.spec_from_file_location("all_layer_cq3_v46", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_training_fixture(root: Path, *, layers: int = 2) -> tuple[Path, Path]:
    artifact_root = root / "artifact"
    repository_commit = "a" * 40
    preregistration_sha256 = "b" * 64
    direct_receipt_sha256 = "c" * 64
    full_context_receipt_sha256 = "d" * 64
    shapes = {
        "ffn_gate": (2, 256),
        "ffn_up": (2, 256),
        "ffn_down": (1, 512),
    }
    layer_records = []
    for layer in range(layers):
        descriptors = []
        training = []
        for tensor, shape in MODULE.expected_targets(
            layer, target_shapes=shapes
        ).items():
            relative = Path("layers") / f"{layer:02d}" / "cq3" / f"{tensor}.json"
            descriptor_path = artifact_root / relative
            descriptor_path.parent.mkdir(parents=True, exist_ok=True)
            payload = bytes(
                [layer + len(training) + 1]
            ) * MODULE.expected_payload_bytes(shape)
            payload_path = descriptor_path.with_suffix(".cq3")
            payload_path.write_bytes(payload)
            payload_record = {
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
            descriptor = {
                "schema": MODULE.TENSOR_SCHEMA,
                "repository_commit": repository_commit,
                "preregistration_sha256": preregistration_sha256,
                "training_amendment_sha256": preregistration_sha256,
                "training_cache_sha256": full_context_receipt_sha256,
                "layer": layer,
                "gguf_tensor": tensor,
                "shape": list(shape),
                "group_size": 128,
                "row_shard": 128,
                "payload": payload_record,
                "all_pack_unpack_mappings_exact": True,
                "complete": True,
                "terminal": True,
            }
            descriptor_path.write_text(json.dumps(descriptor))
            descriptors.append(relative.as_posix())
            training.append({"gguf_tensor": tensor, "payload": payload_record})
        layer_records.append(
            {
                "schema": MODULE.LAYER_SCHEMA,
                "model": MODULE.MODEL_ID,
                "revision": MODULE.MODEL_REVISION,
                "repository_commit": repository_commit,
                "preregistration_sha256": preregistration_sha256,
                "direct_receipt_sha256": direct_receipt_sha256,
                "layer": layer,
                "cq3_descriptors": descriptors,
                "cq3_training": training,
                "passed": True,
                "complete": True,
                "terminal": True,
            }
        )
    receipt = {
        "schema": MODULE.TRAINING_SCHEMA,
        "model": MODULE.MODEL_ID,
        "revision": MODULE.MODEL_REVISION,
        "repository_commit": repository_commit,
        "preregistration_sha256": preregistration_sha256,
        "direct_receipt_sha256": direct_receipt_sha256,
        "full_context_receipt_sha256": full_context_receipt_sha256,
        "layers": layer_records,
        "gate": {"passed": True},
        "complete": True,
        "terminal": True,
    }
    receipt_path = root / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    return receipt_path, artifact_root


def test_collects_exact_complete_trained_inventory(tmp_path: Path) -> None:
    receipt_path, artifact_root = build_training_fixture(tmp_path)
    shapes = {
        "ffn_gate": (2, 256),
        "ffn_up": (2, 256),
        "ffn_down": (1, 512),
    }
    records = MODULE.collect_training_payloads(
        receipt_path,
        artifact_root,
        required_layers=2,
        target_shapes=shapes,
    )
    assert len(records) == 6
    assert {record.layer for record in records} == {0, 1}
    assert all(record.path.is_file() for record in records)


def test_rejects_missing_layer_tensor(tmp_path: Path) -> None:
    receipt_path, artifact_root = build_training_fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["layers"][1]["cq3_descriptors"].pop()
    receipt_path.write_text(json.dumps(receipt))
    shapes = {
        "ffn_gate": (2, 256),
        "ffn_up": (2, 256),
        "ffn_down": (1, 512),
    }
    with pytest.raises(ValueError, match="denominator"):
        MODULE.collect_training_payloads(
            receipt_path,
            artifact_root,
            required_layers=2,
            target_shapes=shapes,
        )


def test_rejects_descriptor_path_escape(tmp_path: Path) -> None:
    receipt_path, artifact_root = build_training_fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["layers"][0]["cq3_descriptors"][0] = "../../outside.json"
    receipt_path.write_text(json.dumps(receipt))
    shapes = {
        "ffn_gate": (2, 256),
        "ffn_up": (2, 256),
        "ffn_down": (1, 512),
    }
    with pytest.raises(ValueError, match="escapes"):
        MODULE.collect_training_payloads(
            receipt_path,
            artifact_root,
            required_layers=2,
            target_shapes=shapes,
        )


def test_slot_validation_and_overlay_preserve_non_targets(tmp_path: Path) -> None:
    payload_a_path = tmp_path / "a.cq3"
    payload_b_path = tmp_path / "b.cq3"
    payload_a_path.write_bytes(b"A" * 8)
    payload_b_path.write_bytes(b"B" * 8)
    payloads = [
        MODULE.PayloadRecord(
            "blk.0.ffn_gate.weight",
            0,
            (1, 256),
            payload_a_path,
            8,
            sha256_bytes(b"A" * 8),
        ),
        MODULE.PayloadRecord(
            "blk.0.ffn_up.weight",
            0,
            (1, 256),
            payload_b_path,
            8,
            sha256_bytes(b"B" * 8),
        ),
    ]
    slots = {
        payloads[0].tensor: MODULE.TensorSlot(43, (1, 256), 8, 8),
        payloads[1].tensor: MODULE.TensorSlot(43, (1, 256), 24, 8),
    }
    ranges = MODULE.validate_tensor_slots(slots, payloads)
    template = tmp_path / "template.gguf"
    template.write_bytes(bytes(range(40)))
    output = tmp_path / "candidate.gguf"
    MODULE.atomic_overlay(template, output, ranges)
    exclusions = [(start, stop) for start, stop, _ in ranges]
    assert MODULE.sha256_excluding(template, exclusions) == MODULE.sha256_excluding(
        output, exclusions
    )
    assert output.read_bytes()[8:16] == b"A" * 8
    assert output.read_bytes()[24:32] == b"B" * 8


def test_rejects_wrong_gguf_slot_type() -> None:
    payload = MODULE.PayloadRecord(
        "blk.0.ffn_gate.weight", 0, (1, 256), Path("unused"), 68, "0" * 64
    )
    slots = {payload.tensor: MODULE.TensorSlot(12, (1, 256), 0, 68)}
    with pytest.raises(ValueError, match="does not fit"):
        MODULE.validate_tensor_slots(slots, [payload])


def test_template_scope_binds_exact_candidate_and_targets(tmp_path: Path) -> None:
    template = tmp_path / "template.gguf"
    template.write_bytes(b"structural-template")
    payloads = [
        MODULE.PayloadRecord(
            "blk.0.ffn_gate.weight", 0, (1, 256), Path("a"), 68, "1" * 64
        ),
        MODULE.PayloadRecord(
            "blk.0.ffn_up.weight", 0, (1, 256), Path("b"), 68, "2" * 64
        ),
    ]
    scope = {
        "schema": MODULE.SCOPE_SCHEMA,
        "valid": True,
        "target_count": 2,
        "target_names": [item.tensor for item in payloads],
        "expected_types": {"candidate": 43},
        "checks": {
            "all_non_target_types_sizes_and_payloads_identical": True,
            "candidate_target_types_valid": True,
            "exact_target_count": True,
        },
        "models": {
            "candidate": {
                "bytes": template.stat().st_size,
                "sha256": MODULE.sha256_file(template),
            }
        },
    }
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps(scope))
    assert MODULE.validate_template_scope(template, scope_path, payloads)["valid"]
    scope["models"]["candidate"]["sha256"] = "0" * 64
    scope_path.write_text(json.dumps(scope))
    with pytest.raises(ValueError, match="not authenticated"):
        MODULE.validate_template_scope(template, scope_path, payloads)
