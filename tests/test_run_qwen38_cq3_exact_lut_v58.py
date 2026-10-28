from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
RUNNER = RUNTIME / "run_qwen38_27b_cq3_exact_lut32_avx2_v58.py"
LAUNCHER = RUNTIME / "launch_qwen38_27b_cq3_exact_lut32_avx2_v58.sh"


def load_runner():
    spec = importlib.util.spec_from_file_location("qwen38_v58_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shape_record(shape: str, x8: float, ordinary: float, mad: float = 0.01):
    x8_ratios = [x8] * 21
    ordinary_ratios = [ordinary] * 21
    return {
        "schema": "qwen38_cq3_exact_lut_native_benchmark_v58",
        "shape": shape,
        "warmup_pairs": 2,
        "measured_pairs": 21,
        "cache_eviction_bytes": 268_435_456,
        "x8_lut32_bit_exact": True,
        "maximum_scaled_delta_vs_ordinary": 0.0,
        "source_bytes": 2176,
        "x8_bytes": 2176,
        "lut32_bytes": 2176,
        "x8_over_lut32_ratios": x8_ratios,
        "ordinary_over_lut32_ratios": ordinary_ratios,
        "x8_over_lut32_median": x8,
        "ordinary_over_lut32_median": ordinary,
        "x8_ratio_mad_fraction": 0.0,
        "ordinary_ratio_mad_fraction": 0.0,
        "complete": True,
        "terminal": True,
    }


def test_v58_decision_requires_every_frozen_speed_gate() -> None:
    runner = load_runner()
    passed = runner.decide(
        {
            "gate_up": shape_record("gate_up", 1.21, 1.50),
            "down": shape_record("down", 1.10, 1.45),
        }
    )
    assert passed["passed"]
    assert passed["geometric_mean_x8_over_lut32"] >= 1.15
    assert passed["geometric_mean_ordinary_over_lut32"] >= 1.45

    failed = runner.decide(
        {
            "gate_up": shape_record("gate_up", 1.30, 1.60),
            "down": shape_record("down", 1.09, 1.60),
        }
    )
    assert not failed["passed"]
    assert not failed["gates"]["each_shape_at_least_1_10x_x8"]


def test_v58_shape_validation_recomputes_reported_statistics() -> None:
    runner = load_runner()
    record = shape_record("gate_up", 1.20, 1.50)
    record["x8_over_lut32_median"] = 1.21
    with pytest.raises(ValueError, match="reported x8 median"):
        runner.validate_shape(record, "gate_up")


def test_v58_native_validation_matches_frozen_denominators() -> None:
    runner = load_runner()
    assert runner.validate_native(
        {
            "schema": "qwen38_cq3_exact_lut_native_validation_v58",
            "random_groups": 1024,
            "adversarial_groups": 8,
            "reserved_code_checks": 8,
            "direct_scalar_bit_exact": True,
            "scalar_avx2_bit_exact": True,
            "storage_neutral": True,
            "complete": True,
            "terminal": True,
        }
    )


def test_v58_runner_and_launcher_enforce_idle_milan_receipts() -> None:
    runner = RUNNER.read_text()
    launcher = LAUNCHER.read_text()
    preregistration = json.loads(
        (RUNTIME / "qwen38_27b_cq3_exact_lut32_avx2_preregistration_v58.json").read_text()
    )

    assert "AMD EPYC Milan" in preregistration["resources"]["physical_cpu"]
    assert '"Milan" not in cpu.get("Model name", "")' in runner
    assert 'set(records) == set(SHAPES)' in runner
    assert 'x8_geometric >= 1.15' in runner
    assert 'ordinary_geometric >= 1.45' in runner
    assert 'unit != ALLOWED_UNIT' in runner
    assert "MemorySwapMax=0" in launcher
    assert "CPUQuota=100%" in launcher
    assert "AllowedCPUs=0,2,4,6" in launcher
    assert "Restart=no" in launcher
