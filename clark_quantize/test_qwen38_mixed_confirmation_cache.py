from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "collect_qwen38_27b_mixed_confirmation_cache_v1.py"
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
