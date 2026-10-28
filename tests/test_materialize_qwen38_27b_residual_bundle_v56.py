from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / (
    "experiments/llm_ternary_158bit_dossier/cpu_runtime/"
    "materialize_qwen38_27b_residual_bundle_v56.py"
)
SPEC = importlib.util.spec_from_file_location("residual_bundle_v56", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixture(root: Path) -> tuple[Path, Path, Path]:
    artifact = root / "artifacts"
    artifact.mkdir()
    training = {
        "schema": MODULE.TRAINING_SCHEMA,
        "model": MODULE.MODEL,
        "revision": MODULE.REVISION,
        "gate": {"passed": True},
        "complete": True,
        "terminal": True,
    }
    training_path = root / "training.json"
    training_path.write_text(json.dumps(training))
    training_sha = sha256(training_path)
    prereg_sha = "a" * 64
    repository_commit = "b" * 40
    layers = []
    for layer in range(64):
        active = layer in (0, 63)
        record = {
            "schema": MODULE.LAYER_SCHEMA,
            "model": MODULE.MODEL,
            "revision": MODULE.REVISION,
            "repository_commit": repository_commit,
            "preregistration_sha256": prereg_sha,
            "training_receipt_sha256": training_sha,
            "layer": layer,
            "mode": "rank16_int8_residual" if active else "exact_bypass",
            "descriptor": None,
            "payload": None,
            "persistent_payload_bytes": 0,
            "bypass_tensor_exact": True if not active else None,
            "complete": True,
            "terminal": True,
        }
        if active:
            layer_root = artifact / f"layers/{layer:02d}"
            layer_root.mkdir(parents=True)
            payload_path = layer_root / f"layer-{layer}.csr"
            payload_path.write_bytes(bytes([layer + 1]) * MODULE.PAYLOAD_BYTES)
            descriptor_path = layer_root / f"layer-{layer}.json"
            descriptor = {
                "schema": MODULE.PAYLOAD_SCHEMA,
                "model": MODULE.MODEL,
                "revision": MODULE.REVISION,
                "layer": layer,
                "rank": MODULE.RANK,
                "input_features": MODULE.INPUT_FEATURES,
                "output_features": MODULE.OUTPUT_FEATURES,
                "format": MODULE.PAYLOAD_FORMAT,
                "payload": {
                    "bytes": MODULE.PAYLOAD_BYTES,
                    "sha256": sha256(payload_path),
                },
                "preregistration_sha256": prereg_sha,
                "training_receipt_sha256": training_sha,
                "complete": True,
                "terminal": True,
            }
            descriptor_path.write_text(json.dumps(descriptor))
            record.update(
                {
                    "descriptor": str(descriptor_path.relative_to(artifact)),
                    "payload": str(payload_path.relative_to(artifact)),
                    "persistent_payload_bytes": MODULE.PAYLOAD_BYTES,
                    "bypass_tensor_exact": None,
                }
            )
        layers.append(record)
    residual = {
        "schema": MODULE.RESIDUAL_SCHEMA,
        "model": MODULE.MODEL,
        "revision": MODULE.REVISION,
        "repository_commit": repository_commit,
        "preregistration_sha256": prereg_sha,
        "training_receipt_sha256": training_sha,
        "layers": layers,
        "gate": {
            "passed": True,
            "observed_layers": 64,
            "active_residual_layers": 2,
            "exact_bypass_layers": 62,
            "all_layers_terminal": True,
            "all_active_layers_calibration_not_regressed": True,
            "all_active_layers_validation_not_regressed": True,
            "all_bypass_layers_exact": True,
            "residual_payload_within_maximum": True,
            "total_residual_payload_bytes": 2 * MODULE.PAYLOAD_BYTES,
        },
        "complete": True,
        "terminal": True,
    }
    residual_path = root / "residual.json"
    residual_path.write_text(json.dumps(residual))
    return training_path, residual_path, artifact


def test_v56_builds_and_authenticates_active_and_bypass_layers(tmp_path: Path) -> None:
    training_path, residual_path, artifact = build_fixture(tmp_path)
    records, _, residual = MODULE.collect_payloads(
        training_path, residual_path, artifact
    )
    output = tmp_path / "residual.q38r16"
    MODULE.write_bundle(
        records,
        output,
        training_sha256=sha256(training_path),
        residual_sha256=sha256(residual_path),
        preregistration_sha256=residual["preregistration_sha256"],
    )
    summary = MODULE.validate_bundle(
        output,
        training_sha256=sha256(training_path),
        residual_sha256=sha256(residual_path),
        preregistration_sha256=residual["preregistration_sha256"],
    )

    assert summary.active_layers == [0, 63]
    assert len(summary.bypass_layers) == 62
    assert summary.bytes == MODULE.PAYLOAD_OFFSET + 2 * MODULE.PAYLOAD_BYTES
    assert summary.root_sha256 == MODULE.sha256_with_zeroed_root(output)


def test_v56_embedded_root_rejects_payload_tamper(tmp_path: Path) -> None:
    training_path, residual_path, artifact = build_fixture(tmp_path)
    records, _, residual = MODULE.collect_payloads(
        training_path, residual_path, artifact
    )
    output = tmp_path / "residual.q38r16"
    MODULE.write_bundle(
        records,
        output,
        training_sha256=sha256(training_path),
        residual_sha256=sha256(residual_path),
        preregistration_sha256=residual["preregistration_sha256"],
    )
    with output.open("r+b") as stream:
        stream.seek(MODULE.PAYLOAD_OFFSET)
        stream.write(b"X")
    with pytest.raises(ValueError, match="header authentication"):
        MODULE.validate_bundle(
            output,
            training_sha256=sha256(training_path),
            residual_sha256=sha256(residual_path),
            preregistration_sha256=residual["preregistration_sha256"],
        )


def test_v56_rejects_payload_on_exact_bypass(tmp_path: Path) -> None:
    training_path, residual_path, artifact = build_fixture(tmp_path)
    residual = json.loads(residual_path.read_text())
    residual["layers"][1]["payload"] = "unexpected.csr"
    residual_path.write_text(json.dumps(residual))

    with pytest.raises(ValueError, match="bypass layer"):
        MODULE.collect_payloads(training_path, residual_path, artifact)


def test_v56_source_forbids_dense_residual_materialization() -> None:
    source = SOURCE.read_text()

    assert '"persistent_dense_residual_bytes": 0' in source
    assert '"rank_scratch_bytes_per_task": 64' in source
    assert "PAYLOAD_BYTES = 174_144" in source
    assert "LAYER_COUNT = 64" in source
