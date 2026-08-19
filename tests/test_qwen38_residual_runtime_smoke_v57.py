from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
BUNDLE_SCRIPT = RUNTIME / "build_qwen38_residual_smoke_bundle_v57.py"
SMOKE = RUNTIME / "llama_cpp/qwen38-residual-runtime-smoke.cpp"
BUILD = REPO / "scripts/cpu/build_qwen38_residual_llamacpp_runtime_smoke_v57.sh"
SPEC = importlib.util.spec_from_file_location("smoke_bundle_v57", BUNDLE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_v57_smoke_bundle_identities_are_exact_pilot_evidence() -> None:
    assert MODULE.TRAINING_SHA256 == (
        "6871879dd4413419bfb36e233275481bce291d6bac1c884770c2514c876eeb73"
    )
    assert MODULE.RESIDUAL_SHA256 == (
        "26dae44ab53bf194b89c46e7a175808865b8c8eb40207dd83ef4edaefec7ba36"
    )
    assert MODULE.PREREGISTRATION_SHA256 == (
        "4d5b8646887ccee1959fcd12fb8056deaa57ce06941a0f721c57843ba15dbd36"
    )


def test_v57_smoke_checks_active_parity_and_zero_node_bypass() -> None:
    source = SMOKE.read_text()

    assert "cq3_calibration_residual_f32_avx2_trusted" in source
    assert "graph output differs from exact native algebra" in source
    assert "maximum_absolute_error" in source
    assert "bit_exact" in source
    assert "exact bypass unexpectedly built a graph node" in source
    assert "states.size() != 1" in source


def test_v57_smoke_builder_uses_transformed_sources_and_strict_flags() -> None:
    source = BUILD.read_text()

    assert "mktemp -d" in source
    assert 'model_source="$llama_source/src/models"' in source
    assert "qwen38-residual-runtime-smoke.cpp" in source
    assert "qwen38-residual-native.o" in source
    assert "-mavx2 -mfma -mf16c" in source
    assert "-Wall -Wextra -Werror" in source
    assert "-lggml-cpu -lggml-base -lggml" in source
