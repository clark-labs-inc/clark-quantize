from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import torch


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "collect_qwen38_27b_mixed_confirmation_cache_v1.py"
)
RECEIPT = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime/receipts"
    / "qwen38_27b_calibration_ranked_aff3_cq3_mixed_v1/confirmation-cache.json"
)


def test_confirmation_collector_is_qwen38_only_and_exactly_three_blocks() -> None:
    source = SCRIPT.read_text()
    assert 'MODEL_ID = "Qwen/Qwen3.8-27B"' in source
    assert "TARGET_LAYERS = (0, 32, 63)" in source
    assert "Qwen3.5-4B" not in source
    assert "Qwen3_5DecoderLayer" in source
    assert "hidden.dtype != torch.bfloat16" in source


def test_confirmation_collector_enforces_the_sealed_denominator() -> None:
    source = SCRIPT.read_text()
    assert 'or len(texts) != 4' in source
    assert 'protocol.get("selected_rows_per_layer") != 32' in source
    assert "confirmation_rows.numel() != 32" in source
    assert '"confirmation_rows": confirmation_rows.numel()' in source
    assert "checkpoint_or_hyperparameter_selection_forbidden" in source
    assert "old_texts & set(texts)" in source


def test_confirmation_collector_is_restartable_and_authenticates_every_tensor() -> None:
    source = SCRIPT.read_text()
    assert "validate_existing(" in source
    assert "base.sha256_file(cache_path)" in source
    assert "base.atomic_torch_save(cache, args.cache)" in source
    assert "base.tensor_sha256(value)" in source
    assert '"complete": True' in source
    assert '"terminal": True' in source


def test_confirmation_selected_rows_are_deterministic_and_unique() -> None:
    runtime = SCRIPT.parent
    sys.path.insert(0, str(runtime))
    try:
        spec = importlib.util.spec_from_file_location("mixed_confirmation_cache", SCRIPT)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        mask = torch.ones(4, 64, dtype=torch.long)
        selected = module.base.selected_flat_rows(mask, range(4), 32)
    finally:
        sys.path.remove(str(runtime))
    assert selected.numel() == 32
    assert len(set(selected.tolist())) == 32
    assert int(selected.min()) == 0
    assert int(selected.max()) == 255


def test_confirmation_receipt_is_exact_finite_and_terminal() -> None:
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["schema"] == "qwen38_27b_mixed_confirmation_cache_receipt_v1"
    assert receipt["model"] == "Qwen/Qwen3.8-27B"
    assert receipt["target_layers"] == [0, 32, 63]
    assert receipt["target_layer_types"] == [
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    assert receipt["confirmation_texts"] == 4
    assert receipt["confirmation_rows"] == 32
    assert receipt["finite"] is True
    assert receipt["cache"]["bytes"] == 15_736_197
    assert len(receipt["cache"]["sha256"]) == 64
    assert set(receipt["cached_tensors"]) == {"0", "32", "63"}
    assert all(
        values[kind]["shape"] == [4, 64, 5120]
        for values in receipt["cached_tensors"].values()
        for kind in ("input", "output")
    )
    assert receipt["complete"] is True
    assert receipt["terminal"] is True
    assert receipt["claim_boundary"].startswith(
        "fresh exact BF16 three-block confirmation cache only"
    )
