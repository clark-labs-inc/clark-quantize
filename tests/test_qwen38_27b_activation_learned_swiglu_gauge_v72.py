from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
MODULE_PATH = RUNTIME / "run_qwen38_27b_activation_learned_swiglu_gauge_v72.py"
PREREG = RUNTIME / "qwen38_27b_activation_learned_swiglu_gauge_preregistration_v72.json"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUNTIME))
SPEC = importlib.util.spec_from_file_location("activation_gauge_v72", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_activation_gauge_optimizer_is_bounded_and_records_frozen_steps() -> None:
    generator = torch.Generator().manual_seed(20260820)
    source = torch.randn(4, 128, generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    teacher = torch.randn(6, 128, generator=generator, dtype=torch.float32)
    candidate = teacher + 0.05 * torch.randn(
        6, 128, generator=generator, dtype=torch.float32
    )
    gauge, history = MODULE.learn_activation_gauge(
        source,
        candidate,
        teacher,
        row_shard=2,
        group_size=128,
        outer_steps=2,
        learning_rate=0.01,
        gradient_clip_norm=1.0,
        log_l2_weight=0.0001,
        minimum=0.5,
        maximum=2.0,
    )

    assert len(history) == 3
    assert [item["step"] for item in history] == [0, 1, 2]
    assert float(gauge.min()) >= 0.5
    assert float(gauge.max()) <= 2.0
    assert torch.isfinite(gauge).all()
    assert all(item["hard_calibration_nmse"] >= 0 for item in history)


def test_evaluate_identity_gauge_is_finite() -> None:
    generator = torch.Generator().manual_seed(72)
    source = torch.randn(3, 128, generator=generator).to(torch.bfloat16)
    inputs = torch.randn(5, 128, generator=generator)
    score = MODULE.evaluate_gauge_nmse(
        source,
        inputs,
        inputs,
        torch.ones(128),
        row_shard=1,
        group_size=128,
    )
    assert score >= 0
    assert torch.isfinite(torch.tensor(score))


def test_v72_preregistration_freezes_interleaved_confirmation_protocol() -> None:
    prereg = json.loads(PREREG.read_text())
    assert prereg["status"] == "frozen_before_execution"
    assert prereg["scope"]["layer"] == 59
    assert prereg["scope"]["calibration_rows"] == 32
    assert prereg["scope"]["confirmation_rows"] == 32
    assert prereg["row_split"] == {
        "source_order": "the frozen 64 calibration_flat_rows from v42",
        "train_start": 0,
        "confirmation_start": 1,
        "stride": 2,
        "training_positions": "even positions 0 through 62",
        "confirmation_positions": "odd positions 1 through 63",
        "split_selected_before_execution": True,
    }
    assert prereg["gauge_optimization"]["confirmation_rows_observed"] == 0
    assert prereg["down_training"]["confirmation_rows_observed"] == 0
    assert prereg["gauge_optimization"]["sweep"] is False
    assert prereg["pilot_gate"]["runtime_sidecar_bytes"] == 0
    assert prereg["pilot_gate"]["additional_runtime_operations"] == 0
