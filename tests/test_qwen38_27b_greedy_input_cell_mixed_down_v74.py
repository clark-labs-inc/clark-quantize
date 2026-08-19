from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
MODULE_PATH = RUNTIME / "run_qwen38_27b_greedy_input_cell_mixed_down_v74.py"
PREREG = (
    RUNTIME / "qwen38_27b_greedy_input_cell_mixed_down_preregistration_v74.json"
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RUNTIME))
SPEC = importlib.util.spec_from_file_location("cell_mixed_down_v74", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_standalone_cq3_group_stream_roundtrips_exactly() -> None:
    generator = torch.Generator().manual_seed(74)
    codes = torch.randint(-1, 2, (3, 2, 128), generator=generator, dtype=torch.int8)
    scales = torch.rand(3, 2, generator=generator).to(torch.float16)
    payload = MODULE.pack_cq3_group_stream(codes, scales)

    assert len(payload) == 3 * 2 * 34
    checked_codes, checked_scales = MODULE.unpack_cq3_group_stream(
        payload, rows=3, groups=2
    )
    assert torch.equal(checked_codes, codes)
    assert torch.equal(checked_scales, scales)


def test_standalone_int4_group_stream_roundtrips_exactly() -> None:
    generator = torch.Generator().manual_seed(20260820)
    source = torch.randn(3, 256, generator=generator)
    codes, scales = MODULE.v73.quantize_signed_int4_grouped(
        source, group_size=128, iterations=4
    )
    payload = MODULE.pack_int4_group_stream(codes, scales)

    assert len(payload) == 3 * 2 * 66
    checked_codes, checked_scales = MODULE.unpack_int4_group_stream(
        payload, rows=3, groups=2
    )
    assert torch.equal(checked_codes, codes)
    assert torch.equal(checked_scales, scales)


def test_greedy_selection_updates_the_exact_current_residual() -> None:
    generator = torch.Generator().manual_seed(7401)
    delta = torch.randn(5, 3, 4, 2, generator=generator)
    residual = delta[:, :, 0, :] + delta[:, :, 2, :]
    selected, final_residual, history = MODULE.greedy_select_cells(
        delta, residual, steps=2
    )
    correction = MODULE.selected_cell_correction(delta, selected)

    assert selected.shape == (3, 2)
    assert all(len(set(row)) == 2 for row in selected.tolist())
    torch.testing.assert_close(final_residual, residual - correction)
    assert len(history) == 2
    assert MODULE.selection_mask(selected, groups=4).sum() == 6


def test_v74_preregistration_freezes_disjoint_prompt_cell_protocol() -> None:
    prereg = json.loads(PREREG.read_text())
    data = prereg["data_split"]
    cells = prereg["cell_selection"]
    layout = prereg["canonical_layout"]

    assert prereg["status"] == "frozen_before_layer59_validation_open"
    assert prereg["scope"]["layer"] == 59
    assert data["selection_row_count"] == 64
    assert data["confirmation_row_count"] == 32
    assert data["confirmation_layer59_activations_opened_before_v74"] is False
    assert data["selection_and_confirmation_prompts_disjoint"] is True
    assert cells["output_tile_rows"] == 8
    assert cells["input_group_columns"] == 128
    assert cells["steps_per_output_tile"] == 16
    assert cells["selected_cells"] == 10_240
    assert cells["confirmation_rows_observed"] == 0
    assert cells["sweep"] is False
    assert prereg["int4_quantizer"]["sweep"] is False
    assert layout["hybrid_down_payload_bytes"] == (
        layout["header_bytes"]
        + layout["cell_mode_bitmap_bytes"]
        + layout["unselected_cells"] * layout["cq3_bytes_per_cell"]
        + layout["selected_cells"] * layout["int4_bytes_per_cell"]
    )
    assert layout["hybrid_three_projection_mlp_payload_bytes"] == (
        2 * 23_674_880 + layout["hybrid_down_payload_bytes"]
    )
    observed_increase = (
        layout["hybrid_three_projection_mlp_payload_bytes"]
        - layout["baseline_three_cq3_mlp_payload_bytes"]
    ) / layout["baseline_three_cq3_mlp_payload_bytes"]
    assert observed_increase == layout["relative_full_mlp_payload_increase"]
    assert observed_increase < 0.04
    assert MODULE.HEADER.size == 64
