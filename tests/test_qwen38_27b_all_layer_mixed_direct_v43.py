from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
SCRIPT = RUNTIME / "run_qwen38_27b_all_layer_mixed_direct_v43.py"
PREREG = RUNTIME / "qwen38_27b_all_layer_mixed_direct_v43.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("all_layer_direct_v43", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preregistration() -> dict[str, object]:
    return json.loads(PREREG.read_text())


def synthetic_manifest() -> dict[str, object]:
    section = {
        "indices": {"sha256": "indices"},
        "anchors": {"sha256": "anchors"},
        "codebook": {"sha256": "codebook"},
        "unpacked_indices_sha256": "unpacked",
    }
    return {
        "site": "attention_output",
        "tensor_names": ["linear_attn.out_proj.weight"],
        "input_dimension": 6144,
        "pairing_seed": 7,
        "angles": {"sha256": "angles", "shape": [6, 3072]},
        "tensors": [
            {
                "tensor": "layers.0.linear_attn.out_proj.weight",
                "shape": [5120, 6144],
                "group_size": 128,
                "codes": {"sha256": "codes", "unpacked_sha256": "raw-codes"},
                "alpha": dict(section),
                "mu": dict(section),
                "storage_bytes": 123,
            }
        ],
        "actual_storage_bytes": 456,
        "fit_metrics_before_parameter_codebooks": {"validation_nmse": 999.0},
    }


def test_v43_is_frozen_for_all_qwen38_layers_only() -> None:
    data = preregistration()
    assert data["status"] == "frozen_before_all_layer_direct_execution"
    assert data["scope"]["active_model_only"] == "Qwen/Qwen3.8-27B"
    assert data["scope"]["target_layers"] == list(range(64))
    assert data["scope"]["qwen_4b_compute_forbidden"] is True


def test_v43_qwen_layer_pattern_is_exact() -> None:
    module = load_module()
    observed = [module.expected_layer_type(index) for index in range(64)]
    assert observed.count("linear_attention") == 48
    assert observed.count("full_attention") == 16
    assert all(
        value == ("full_attention" if (index + 1) % 4 == 0 else "linear_attention")
        for index, value in enumerate(observed)
    )


def test_v43_uses_complete_context_but_calibration_rows_only() -> None:
    data = preregistration()
    protocol = data["data_protocol"]
    assert protocol["complete_sequence_context_required_for_forward"] is True
    assert protocol["calibration_sequences"] == 8
    assert protocol["selected_calibration_rows_per_layer"] == 64
    assert protocol["internal_validation_sequences_used"] == 0
    assert protocol["fresh_confirmation_sequences_used"] == 0
    source = SCRIPT.read_text()
    assert 'validation=attention_inputs' in source
    assert 'context["validation_flat_rows"]' not in source
    assert "full_hidden = shard[\"input\"].contiguous()" in source


def test_v43_payload_fingerprint_excludes_reporting_metrics() -> None:
    module = load_module()
    left = synthetic_manifest()
    right = synthetic_manifest()
    right["fit_metrics_before_parameter_codebooks"] = {"validation_nmse": -1.0}
    assert module.manifest_payload_fingerprint(left) == (
        module.manifest_payload_fingerprint(right)
    )
    right["angles"]["sha256"] = "different"
    assert module.manifest_payload_fingerprint(left) != (
        module.manifest_payload_fingerprint(right)
    )


def test_v43_requires_exact_pilot_algebra_and_scores() -> None:
    data = preregistration()
    pilot = data["pilot_reproduction"]
    assert pilot["layers"] == [0, 32, 63]
    assert "payload hash" in pilot["aff3_requirement"]
    assert "payload hashes" in pilot["cq3_requirement"]
    assert "absolute 1e-8" in pilot["score_requirement"]
    source = SCRIPT.read_text()
    assert "manifest_payload_fingerprint(old_manifest)" in source
    assert "sha256_file(old_payload)" in source
    assert "abs(calibration_nmse - old_score) <= 1e-8" in source


def test_v43_uses_the_frozen_successful_representation() -> None:
    representation = preregistration()["representation"]
    assert representation["attention_output"]["format"] == "affine base-3"
    assert representation["attention_output"]["rotation_stages"] == 6
    assert representation["attention_output"]["rotation_steps"] == 8
    assert representation["attention_output"]["rotation_learning_rate"] == 0.002
    assert representation["mlp"]["format"] == "CQ3_0"
    assert representation["mlp"]["stage"].startswith("deterministic direct")
    assert representation["dense_reconstructed_weights_persisted"] is False


def test_v43_layer_terminal_is_the_restart_boundary() -> None:
    data = preregistration()
    assert "one complete layer" in data["execution"]["restart_boundary"]
    source = SCRIPT.read_text()
    assert source.index("cache_v1.atomic_json(result, terminal_path)") < source.index(
        '"event": "all_layer_direct_layer_terminal"'
    )
    assert "validate_terminal_layer(" in source
    assert "all_layer_direct_layer_reused" in source


def test_v43_gate_forbids_fallback_or_partial_denominators() -> None:
    data = preregistration()
    gate = data["gate"]
    assert gate["required_layers"] == 64
    assert gate["required_calibration_rows_per_layer"] == 64
    assert gate["maximum_calibration_nmse_per_layer"] == 0.12
    assert gate["maximum_selected_payload_bytes"] == 5_100_000_000
    rules = "\n".join(data["promotion_rules"])
    assert "no failing layer may silently fall back" in rules
    assert "fresh confirmation and whole-model benchmarks remain sealed" in rules


def test_v43_resources_are_nonspot_safe_and_bounded() -> None:
    resources = preregistration()["resources"]
    assert resources["cpu_quota_percent"] == 400
    assert resources["memory_max_bytes"] == 31_000_000_000
    assert resources["memory_swap_max_bytes"] == 0
    assert resources["minimum_free_disk_bytes"] == 20 * 1024**3
    assert resources["maximum_restarts"] == 2


def test_v43_frozen_inputs_match_hashes() -> None:
    for descriptor in preregistration()["frozen_inputs"].values():
        path = RUNTIME / descriptor["path"]
        assert path.is_file()
        assert sha256(path) == descriptor["sha256"]


def test_v43_claim_boundary_forbids_training_or_breakthrough_claims() -> None:
    boundary = preregistration()["claim_boundary"]
    assert "does not establish training success" in boundary
    assert "end-to-end capability or speed" in boundary
    assert "scientific breakthrough" in boundary
