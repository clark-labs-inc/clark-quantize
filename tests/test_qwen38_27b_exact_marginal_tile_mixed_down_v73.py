from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
MODULE_PATH = RUNTIME / "run_qwen38_27b_exact_marginal_tile_mixed_down_v73.py"
PREREG = (
    RUNTIME
    / "qwen38_27b_exact_marginal_tile_mixed_down_preregistration_v73.json"
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUNTIME))
SPEC = importlib.util.spec_from_file_location("tile_mixed_down_v73", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_signed_int4_quantizer_and_packing_are_exact_and_bounded() -> None:
    generator = torch.Generator().manual_seed(20260820)
    source = torch.randn(3, 64, generator=generator)
    source[0, :32] = 0
    codes, scales = MODULE.quantize_signed_int4_grouped(
        source, group_size=32, iterations=4
    )

    assert codes.dtype == torch.int8
    assert scales.dtype == torch.float16
    assert int(codes.min()) >= -8
    assert int(codes.max()) <= 7
    assert float(scales[0, 0]) == 0
    payload = MODULE.pack_signed_int4(codes, scales)
    assert len(payload) == 3 * 2 * 18
    checked_codes, checked_scales = MODULE.unpack_signed_int4(
        payload, rows=3, columns=64
    )
    assert torch.equal(checked_codes, codes)
    assert torch.equal(checked_scales, scales)
    reconstructed = MODULE.dequantize_signed_int4(checked_codes, checked_scales)
    assert reconstructed.shape == source.shape
    assert torch.isfinite(reconstructed).all()


def test_fixed_budget_selection_is_exact_with_stable_tie_break() -> None:
    benefits = torch.tensor([1.0, 4.0, 4.0, -2.0, 3.0])
    assert MODULE.select_exact_fixed_budget_tiles(benefits, count=3) == [1, 2, 4]


def test_tile_marginals_equal_direct_tile_sse_reductions() -> None:
    baseline = torch.tensor([[3.0, 2.0, 1.0, 0.0], [1.0, 2.0, 3.0, 4.0]])
    alternative = torch.tensor([[1.0, 2.0, 0.0, 0.0], [1.0, 0.0, 3.0, 2.0]])
    target = torch.zeros_like(baseline)
    benefits = MODULE.tile_marginal_benefits(
        baseline,
        alternative,
        target,
        torch.tensor([0, 1]),
        tile_rows=2,
    )
    expected = torch.tensor(
        [
            (baseline[:, :2].square() - alternative[:, :2].square()).sum(),
            (baseline[:, 2:].square() - alternative[:, 2:].square()).sum(),
        ]
    )
    torch.testing.assert_close(benefits, expected)


def test_v73_preregistration_freezes_x8_exact_marginal_layout() -> None:
    prereg = json.loads(PREREG.read_text())
    selection = prereg["tile_selection"]
    layout = prereg["canonical_layout"]

    assert prereg["status"] == "frozen_before_execution"
    assert prereg["scope"]["layer"] == 59
    assert selection["output_tile_rows"] == 8
    assert selection["total_tiles"] == 640
    assert selection["int4_tiles"] == 80
    assert selection["confirmation_rows_observed"] == 0
    assert selection["sweep"] is False
    assert prereg["int4_quantizer"]["sweep"] is False
    assert layout["hybrid_down_payload_bytes"] == (
        layout["header_bytes"]
        + layout["tile_mode_bitmap_bytes"]
        + layout["unselected_cq3_rows"] * layout["cq3_bytes_per_row"]
        + layout["selected_int4_rows"] * layout["int4_bytes_per_row"]
    )
    assert layout["hybrid_three_projection_mlp_payload_bytes"] == (
        2 * 23_674_880 + layout["hybrid_down_payload_bytes"]
    )
    observed_increase = (
        layout["hybrid_three_projection_mlp_payload_bytes"]
        - layout["baseline_three_cq3_mlp_payload_bytes"]
    ) / layout["baseline_three_cq3_mlp_payload_bytes"]
    assert observed_increase == layout["relative_full_mlp_payload_increase"]
    assert observed_increase < 0.05
    assert layout["high_precision_weight_materialization_at_runtime"] is False
    assert MODULE.HEADER.size == 64
