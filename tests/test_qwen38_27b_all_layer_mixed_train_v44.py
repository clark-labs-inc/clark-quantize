from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


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


def test_v44_preauthenticates_all_direct_parent_layers(monkeypatch) -> None:
    module = load_module()
    seen: list[int] = []

    def fake_validate(result, **kwargs):
        seen.append(result["layer"])
        assert kwargs["artifact_root"] == Path("/direct")
        assert kwargs["aff3_root"] == Path("/direct/aff3-base")
        assert kwargs["full_context_receipt_sha256"] == (
            module.FULL_CONTEXT_RECEIPT_SHA256
        )

    monkeypatch.setattr(module.direct, "validate_terminal_layer", fake_validate)
    module.validate_direct_parent_artifacts(
        {
            "repository_commit": "direct-commit",
            "preregistration_sha256": "direct-prereg",
            "layers": [
                {"layer": layer, "passed": True} for layer in range(64)
            ],
        },
        direct_artifact_root=Path("/direct"),
    )
    assert seen == list(range(64))


@pytest.mark.parametrize(
    "layers",
    [
        [{"layer": layer, "passed": True} for layer in range(63)],
        [
            {"layer": layer, "passed": layer != 31}
            for layer in range(64)
        ],
        [
            {"layer": (1 if layer == 0 else layer), "passed": True}
            for layer in range(64)
        ],
    ],
)
def test_v44_rejects_changed_direct_parent_denominator(layers) -> None:
    module = load_module()
    with pytest.raises(ValueError, match="parent denominator"):
        module.validate_direct_parent_artifacts(
            {
                "repository_commit": "direct-commit",
                "preregistration_sha256": "direct-prereg",
                "layers": layers,
            },
            direct_artifact_root=Path("/direct"),
        )


def test_v44_preauthenticates_all_context_shards(tmp_path, monkeypatch) -> None:
    module = load_module()
    context = tmp_path / "context.pt"
    context.write_bytes(b"context")
    layers_root = tmp_path / "layers"
    layers_root.mkdir()
    layers = []
    for layer in range(64):
        path = layers_root / f"{layer:02d}.pt"
        payload = f"layer-{layer}".encode()
        path.write_bytes(payload)
        layers.append(
            {
                "layer": layer,
                "shard": {"bytes": len(payload), "sha256": f"hash-{layer}"},
            }
        )

    def fake_sha(path):
        if path == context:
            return "context-hash"
        return f"hash-{int(path.stem)}"

    monkeypatch.setattr(module, "sha256_file", fake_sha)
    module.validate_full_context_parent_artifacts(
        {
            "context": {"bytes": len(b"context"), "sha256": "context-hash"},
            "layers": layers,
        },
        full_context_root=tmp_path,
    )


def test_v44_preflight_is_before_artifact_creation() -> None:
    source = SCRIPT.read_text()
    assert source.index("if args.preflight_only:") < source.index(
        "args.artifact_root.mkdir"
    )
    assert '"writes": 0' in source
