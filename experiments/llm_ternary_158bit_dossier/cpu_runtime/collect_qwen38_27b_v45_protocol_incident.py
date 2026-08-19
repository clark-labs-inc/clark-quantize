#!/usr/bin/env python3
"""Retain the authenticated v45 layer-32 protocol contradiction."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PREREGISTRATION_SHA256 = (
    "c4b64d7bdb6af2e6e09fc857cf0defd7bcd9a2b24903beb3b5023c3692a51664"
)
RUNNER_SHA256 = "c262ea82c9a9ce706baadd2b6902c7243b34a45a6cfd27a91b00339da46e7ef1"
LAYER_SCHEMA = "qwen38_27b_all_layer_sensitivity_residual_layer_v45"
FAILURE_TEXT = "ValueError: v45 pilot sensitivity mode reproduction changed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_partial_layers(layers: list[dict[str, Any]]) -> dict[str, Any]:
    if (
        len(layers) != 32
        or [layer.get("layer") for layer in layers] != list(range(32))
        or not all(layer.get("schema") == LAYER_SCHEMA for layer in layers)
        or not all(layer.get("model") == MODEL_ID for layer in layers)
        or not all(layer.get("revision") == MODEL_REVISION for layer in layers)
        or not all(layer.get("complete") is True for layer in layers)
        or not all(layer.get("terminal") is True for layer in layers)
    ):
        raise ValueError("v45 partial terminal layer denominator changed")
    active = [layer for layer in layers if layer.get("mode") == "rank16_int8_residual"]
    bypass = [layer for layer in layers if layer.get("mode") == "exact_bypass"]
    if len(active) + len(bypass) != len(layers):
        raise ValueError("v45 partial sensitivity mode changed")
    reductions = [
        float(layer["relative_validation_nmse_reduction"]) for layer in active
    ]
    regressions = [
        int(layer["layer"])
        for layer in active
        if float(layer["final_validation_nmse"])
        > float(layer["base_validation_nmse"])
    ]
    ordered = sorted(reductions)
    median = (ordered[15] + ordered[16]) / 2 if len(ordered) == 32 else None
    return {
        "terminal_layers": len(layers),
        "active_residual_layers": len(active),
        "exact_bypass_layers": len(bypass),
        "mean_active_relative_validation_nmse_reduction": (
            sum(reductions) / len(reductions) if reductions else 0.0
        ),
        "median_active_relative_validation_nmse_reduction": median,
        "regressed_layers": regressions,
        "regressed_layer_count": len(regressions),
        "partial_residual_payload_bytes": sum(
            int(layer["persistent_payload_bytes"]) for layer in layers
        ),
        "frozen_mean_reduction_gate_already_missed": (
            bool(reductions) and sum(reductions) / len(reductions) < 0.1
        ),
        "frozen_no_validation_regression_gate_already_missed": bool(regressions),
    }


def validate_protocol_evidence(service_log: str, terminal_state: str) -> None:
    required_log = (
        FAILURE_TEXT,
        '"event": "v45_layer_started", "layer": 32',
        "Scheduled restart job, restart counter is at 1",
    )
    required_state = (
        "ActiveState=inactive",
        "SubState=dead",
        "receipt.json=absent",
        "complete.json=absent",
        "\n32\n",
        "/layers/32/layer-32.csr",
        "/layers/32/layer-32.json",
    )
    if not all(fragment in service_log for fragment in required_log):
        raise ValueError("v45 service failure transcript changed")
    if not all(fragment in terminal_state for fragment in required_state):
        raise ValueError("v45 stopped terminal-state snapshot changed")


def validate_manifest(root: Path, manifest_path: Path) -> list[str]:
    entries: list[str] = []
    for line in manifest_path.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"v45 partial artifact changed: {relative}")
        entries.append(relative)
    observed = sorted(
        str(path.relative_to(root)) for path in (root / "layers").rglob("*") if path.is_file()
    )
    if sorted(entries) != observed:
        raise ValueError("v45 partial artifact manifest denominator changed")
    return entries


def atomic_no_overwrite(payload: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError("refusing to overwrite a retained v45 incident")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence_root
    preregistration = root / "preregistration.json"
    runner = root / "runner.py"
    service_log = root / "service.log"
    terminal_state = root / "terminal-state.txt"
    manifest = root / "remote-artifacts.sha256"
    if (
        sha256_file(preregistration) != PREREGISTRATION_SHA256
        or sha256_file(runner) != RUNNER_SHA256
    ):
        raise ValueError("v45 frozen source identity changed")
    entries = validate_manifest(root, manifest)
    log_text = service_log.read_text()
    state_text = terminal_state.read_text()
    validate_protocol_evidence(log_text, state_text)
    layer_paths = sorted((root / "layers").glob("*/terminal-result.json"))
    layers = [json.loads(path.read_text()) for path in layer_paths]
    summary = summarize_partial_layers(layers)
    if (root / "layers/32/terminal-result.json").exists():
        raise ValueError("v45 layer 32 unexpectedly became terminal")
    layer32_files = [root / "layers/32/layer-32.csr", root / "layers/32/layer-32.json"]
    if not all(path.is_file() for path in layer32_files):
        raise ValueError("v45 layer 32 partial payload evidence changed")
    payload = {
        "schema": "qwen38_27b_all_layer_sensitivity_residual_protocol_incident_v45",
        "experiment_id": "qwen38_27b_all_layer_sensitivity_residual_v45",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "classification": "terminal_protocol_negative_no_v45_receipt",
        "infrastructure_valid": True,
        "blocked_layer": 32,
        "protocol_contradiction": {
            "pilot_selection_statistic": "calibration complete-block NMSE",
            "v45_selection_statistic": "held-out validation complete-block NMSE",
            "simultaneous_constraint": "v45 layer 32 mode must equal the pilot mode",
            "observed_result": "v45 selected an active residual where the pilot selected exact bypass",
            "failure": FAILURE_TEXT,
        },
        "partial_scientific_result": summary,
        "evidence": {
            "collector_sha256": sha256_file(Path(__file__)),
            "preregistration_sha256": sha256_file(preregistration),
            "runner_sha256": sha256_file(runner),
            "service_log_sha256": sha256_file(service_log),
            "terminal_state_sha256": sha256_file(terminal_state),
            "remote_artifact_manifest_sha256": sha256_file(manifest),
            "remote_artifact_files": len(entries),
            "partial_terminal_result_sha256": {
                f"{layer['layer']:02d}": sha256_file(path)
                for layer, path in zip(layers, layer_paths)
            },
            "layer32_partial_payload_sha256": {
                path.name: sha256_file(path) for path in layer32_files
            },
            "v45_terminal_receipt_present": False,
            "v45_completion_marker_present": False,
        },
        "decision": {
            "v45_residual_rejected": True,
            "original_v60_residual_candidate_admission_failed": True,
            "do_not_resume_the_incompatible_v45_runner": True,
            "do_not_materialize_or_execute_v45_payloads": True,
        },
        "claim_boundary": (
            "This is a retained protocol-negative incident with 32 authenticated "
            "partial layer results. It is not a 64-layer v45 receipt and cannot "
            "qualify the residual runtime."
        ),
        "complete": True,
        "terminal": True,
    }
    atomic_no_overwrite(payload, args.output)
    print(json.dumps({"event": "v45_protocol_incident_retained", "sha256": sha256_file(args.output), "terminal": True}, sort_keys=True))


if __name__ == "__main__":
    main()
