from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
SCRIPT = RUNTIME / "run_qwen38_27b_all_layer_mixed_train_v44.py"


def load_module():
    spec = importlib.util.spec_from_file_location("all_layer_train_v44", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    }


def test_v44_scaffold_is_qwen38_all_layer_only() -> None:
    module = load_module()
    assert module.MODEL_ID == "Qwen/Qwen3.8-27B"
    assert module.MODEL_REVISION == (
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    )
    source = SCRIPT.read_text()
    assert "for layer_index in range(64):" in source
    assert 'prereg.get("scope", {}).get("target_layers") != list(range(64))' in source
    assert 'prereg.get("scope", {}).get("qwen_4b_compute_forbidden") is not True' in source


def test_v44_cannot_run_before_a_frozen_v43_pass_handoff() -> None:
    source = SCRIPT.read_text()
    assert 'prereg.get("status") != "frozen_after_v43_pass_before_training"' in source
    assert 'direct_receipt.get("gate", {}).get("passed") is not True' in source
    assert 'prereg.get("parent_direct", {}).get("receipt_sha256")' in source
    assert "!= direct_receipt_sha" in source


def test_v44_training_has_no_validation_or_confirmation_input() -> None:
    source = SCRIPT.read_text()
    assert 'context["validation_flat_rows"]' not in source
    assert "validation_rows" not in source
    assert "confirmation" not in source.lower()
    assert 'context["attention_mask"].long()[:8]' in source
    assert 'context["text_position_ids"].long()[:8]' in source


def test_v44_reuses_the_frozen_successful_optimizers() -> None:
    source = SCRIPT.read_text()
    assert "pilot_training.train_cq3_tensor(" in source
    assert "calibration_only_hard_forward_trust_region(" in source
    assert 'training["mlp_cq3"]["steps_per_row_shard"]' in source
    assert 'aff3_config["outer_steps"]' in source
    assert 'aff3_config["backtracking_factors"]' in source


def test_v44_trained_payload_fingerprint_is_algebra_only() -> None:
    module = load_module()
    left = synthetic_manifest()
    right = synthetic_manifest()
    right["repository_commit"] = "ignored"
    assert module.trained_aff3_fingerprint(left) == (
        module.trained_aff3_fingerprint(right)
    )
    right["tensors"][0]["codes"]["sha256"] = "different"
    assert module.trained_aff3_fingerprint(left) != (
        module.trained_aff3_fingerprint(right)
    )


def test_v44_requires_pilot_payload_and_score_reproduction() -> None:
    source = SCRIPT.read_text()
    assert "PILOT_LAYERS = (0, 32, 63)" in source
    assert "pilot_payload_exact = cq3_exact and trained_aff3_fingerprint(" in source
    assert "pilot_score_exact = abs(" in source
    assert "all_pilot_payload_algebras_exact" in source
    assert "all_pilot_scores_exact_within_1e_8" in source


def test_v44_each_terminal_layer_is_restartable() -> None:
    source = SCRIPT.read_text()
    assert "all_layer_trained_layer_reused" in source
    assert source.index("cache_v1.atomic_json(result, terminal_path)") < source.index(
        '"event": "all_layer_trained_layer_terminal"'
    )
    assert "validate_terminal_layer(" in source
    assert "pilot_training.train_cq3_tensor(" in source


def test_v44_does_not_make_a_quality_or_breakthrough_claim() -> None:
    source = SCRIPT.read_text()
    assert '"claim_boundary": prereg["claim_boundary"]' in source
    assert "internal-validation" not in source.lower()
    assert "breakthrough" not in source.lower()
