from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
MODULE_PATH = RUNTIME / "run_qwen38_27b_swiglu_diagonal_gauge_pilot_v71.py"
PREREG = RUNTIME / "qwen38_27b_swiglu_diagonal_gauge_pilot_preregistration_v71.json"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUNTIME))
SPEC = importlib.util.spec_from_file_location("swiglu_gauge_v71", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_swiglu_up_down_diagonal_gauge_is_exact_in_fp64() -> None:
    generator = torch.Generator().manual_seed(20260820)
    x = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    gate = torch.randn(11, 5, generator=generator, dtype=torch.float64)
    up = torch.randn(11, 5, generator=generator, dtype=torch.float64)
    down = torch.randn(5, 11, generator=generator, dtype=torch.float64)
    gauge = torch.exp(torch.randn(11, generator=generator, dtype=torch.float64))

    control = functional.linear(
        functional.silu(functional.linear(x, gate)) * functional.linear(x, up),
        down,
    )
    candidate = functional.linear(
        functional.silu(functional.linear(x, gate))
        * functional.linear(x, gauge[:, None] * up),
        down / gauge[None, :],
    )

    torch.testing.assert_close(candidate, control, rtol=1e-12, atol=1e-12)


def test_column_rms_gauge_normalizes_unclipped_columns() -> None:
    down = torch.tensor(
        [[1.0, 2.0, 4.0], [-1.0, -2.0, -4.0]], dtype=torch.float32
    )
    gauge = MODULE.derive_column_rms_gauge(
        down, epsilon=1e-12, minimum=0.25, maximum=4.0
    )
    torch.testing.assert_close(gauge, torch.tensor([0.5, 1.0, 2.0]))
    normalized = down / gauge[None, :]
    rms = normalized.square().mean(dim=0).sqrt()
    torch.testing.assert_close(rms, torch.full_like(rms, 2.0))


def test_gauge_rejects_invalid_bounds_and_zero_normalization() -> None:
    with pytest.raises(ValueError):
        MODULE.derive_column_rms_gauge(
            torch.ones(2, 2), epsilon=0, minimum=0.25, maximum=4
        )
    with pytest.raises(ValueError):
        MODULE.derive_column_rms_gauge(
            torch.ones(2, 2), epsilon=1e-12, minimum=2, maximum=4
        )


def test_preregistration_freezes_no_sweep_storage_neutral_pilot() -> None:
    prereg = json.loads(PREREG.read_text())
    assert prereg["status"] == "frozen_before_execution"
    assert prereg["scope"]["layer"] == 59
    assert prereg["gauge"]["sweep"] is False
    assert prereg["gauge"]["learned_parameters"] == 0
    assert prereg["pilot_gate"]["runtime_sidecar_bytes"] == 0
    assert prereg["pilot_gate"]["additional_runtime_operations"] == 0
    assert prereg["pilot_gate"][
        "minimum_relative_complete_mlp_nmse_improvement"
    ] == 0.2
    assert prereg["scope"]["whole_model_generation_forbidden"] is True
    assert prereg["scope"]["performance_timing_forbidden"] is True
