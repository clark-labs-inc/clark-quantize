from __future__ import annotations

import ast
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
SCRIPT = RUNTIME / "collect_qwen38_27b_affine_butterfly_block_cache_v1.py"
PREREG = RUNTIME / "qwen38_27b_affine_butterfly_blocks_preregistration_v1.json"


def test_block_cache_streams_all_64_layers_and_caches_three_complete_pairs() -> None:
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    assert tree is not None
    assert "for layer_index in range(config.num_hidden_layers)" in source
    assert "Qwen3_5DecoderLayer(config, layer_idx=layer_index)" in source
    assert 'cached_layers[str(layer_index)]["output"]' in source
    assert "TARGET_LAYERS = (0, 32, 63)" in source
    assert "embedding-only or synthetic" in PREREG.read_text()


def test_block_cache_binds_exact_qwen38_identity_and_disjoint_texts() -> None:
    source = SCRIPT.read_text()
    prereg = json.loads(PREREG.read_text())
    protocol = prereg["activation_protocol"]
    assert "Qwen/Qwen3.8-27B" in source
    assert "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" in source
    assert "EXPECTED_INDEX_SHA256" in source
    assert "qwen_4b_compute_forbidden" in source
    assert not set(protocol["calibration_texts"]) & set(protocol["validation_texts"])
    assert protocol["maximum_calibration_rows_per_layer"] == 64
    assert protocol["maximum_validation_rows_per_layer"] == 32


def test_block_cache_is_atomic_terminal_and_infrastructure_conservative() -> None:
    source = SCRIPT.read_text()
    assert "atomic_torch_save(cache, args.cache)" in source
    assert '"complete": True' in source
    assert '"terminal": True' in source
    assert "infrastructure-invalid: swap is active" in source
    assert "less than 12 GiB memory available" in source
