from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
MODULE_PATH = RUNTIME / "materialize_qwen38_27b_sensitivity_selected_cq3_v67.py"
PREREG = RUNTIME / "qwen38_27b_sensitivity_selected_cq3_half_preregistration_v67.json"
PREREG_V68 = RUNTIME / "qwen38_27b_sensitivity_selected_cq3_48_preregistration_v68.json"
TRAINING = RUNTIME / "receipts/qwen38_27b_all_layer_mixed_train_v44/terminal-receipt.json"

sys.path.insert(0, str(RUNTIME))
SPEC = importlib.util.spec_from_file_location("materialize_v67", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@dataclass(frozen=True)
class FakePayload:
    layer: int
    tensor: str


def test_layer_parser_is_canonical() -> None:
    assert MODULE.parse_layers("0,2,63") == (0, 2, 63)
    for value in ("", "2,1", "1,1", "-1", "64", "x"):
        with pytest.raises(Exception):
            MODULE.parse_layers(value)


def test_payload_selection_has_exact_triplet_denominator() -> None:
    payloads = [
        FakePayload(layer=layer, tensor=f"{layer}:{kind}")
        for layer in range(4)
        for kind in ("down", "gate", "up")
    ]
    selected = MODULE.selected_payloads(payloads, (1, 3))
    assert len(selected) == 6
    assert {payload.layer for payload in selected} == {1, 3}
    with pytest.raises(ValueError, match="denominator"):
        MODULE.selected_payloads(selected[:-1], (1, 3))


def test_in_place_overlay_requires_partial_suffix(tmp_path: Path) -> None:
    final = tmp_path / "candidate.gguf"
    final.write_bytes(b"01234567")
    with pytest.raises(ValueError, match="partial"):
        MODULE.overlay_in_place(final, [])


def test_frozen_v44_ranking_reproduces_preregistered_half() -> None:
    prereg = json.loads(PREREG.read_text())
    training = json.loads(TRAINING.read_text())
    ranking = []
    for layer in training["layers"]:
        score = max(
            item["aggregate_calibration_metrics"]["selected_calibration_nmse"]
            for item in layer["cq3_training"]
        )
        ranking.append({"layer": layer["layer"], "score": score})
    ranking.sort(key=lambda item: (item["score"], item["layer"]))
    digest = hashlib.sha256(
        json.dumps(ranking, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    selected = sorted(item["layer"] for item in ranking[:32])

    assert digest == prereg["selection"]["ranking_sha256"]
    assert selected == prereg["selection"]["admitted_layers"]
    assert ranking[31]["score"] == prereg["selection"]["maximum_admitted_score"]
    assert ranking[32]["score"] == prereg["selection"]["minimum_rejected_score"]
    assert sorted(selected + prereg["selection"]["q4_retained_layers"]) == list(
        range(64)
    )


def test_candidate_size_and_tensor_denominator_are_exact() -> None:
    prereg = json.loads(PREREG.read_text())
    candidate = prereg["candidate"]
    q4_bytes = prereg["model"]["q4_control_bytes"]
    per_tensor_savings = 50135040 - 23674880
    assert candidate["selected_mlp_tensors"] == 32 * 3
    assert candidate["expected_saved_bytes_vs_q4"] == 32 * 3 * per_tensor_savings
    assert candidate["expected_bytes"] == q4_bytes - candidate[
        "expected_saved_bytes_vs_q4"
    ]


def test_v68_is_the_frozen_48_layer_binary_search_midpoint() -> None:
    prereg = json.loads(PREREG_V68.read_text())
    training = json.loads(TRAINING.read_text())
    ranking = []
    for layer in training["layers"]:
        score = max(
            item["aggregate_calibration_metrics"]["selected_calibration_nmse"]
            for item in layer["cq3_training"]
        )
        ranking.append((score, layer["layer"]))
    ranking.sort()
    selected = sorted(layer for _, layer in ranking[:48])
    candidate = prereg["candidate"]

    assert prereg["binary_search"]["midpoint_layers"] == 48
    assert selected == prereg["selection"]["admitted_layers"]
    assert ranking[47][0] == prereg["selection"]["maximum_admitted_score"]
    assert ranking[48][0] == prereg["selection"]["minimum_rejected_score"]
    assert sorted(selected + prereg["selection"]["q4_retained_layers"]) == list(
        range(64)
    )
    assert candidate["selected_mlp_tensors"] == 48 * 3
    assert candidate["expected_bytes"] == (
        prereg["model"]["q4_control_bytes"]
        - candidate["expected_saved_bytes_vs_q4"]
    )
