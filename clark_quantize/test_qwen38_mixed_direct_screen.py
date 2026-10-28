from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "run_qwen38_27b_mixed_direct_screen_v1.py"
)


def load_module():
    runtime = SCRIPT.parent
    sys.path.insert(0, str(runtime))
    try:
        spec = importlib.util.spec_from_file_location("mixed_direct_screen", SCRIPT)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(runtime))


def test_direct_screen_is_qwen38_only_and_covers_the_large_mixed_map() -> None:
    module = load_module()
    assert module.MODEL_ID == "Qwen/Qwen3.8-27B"
    assert module.TARGET_LAYERS == (0, 32, 63)
    for layer in module.TARGET_LAYERS:
        targets = module.mlp_targets(layer)
        assert len(targets) == 3
        assert sum(target["shape"][0] * target["shape"][1] for target in targets) == 267_386_880
    assert "Qwen3.5-4B" not in SCRIPT.read_text()


def test_direct_screen_computes_only_frozen_rows_in_the_cq3_mlp() -> None:
    module = load_module()
    codes_gate = torch.zeros(8, 256, dtype=torch.int8)
    scales_gate = torch.ones(8, 2, dtype=torch.float16)
    codes_down = torch.zeros(256, 8, dtype=torch.int8)
    # Use real modules with compatible dimensions through tiny stand-ins.
    class Linear(torch.nn.Module):
        def __init__(self, out_features: int):
            super().__init__()
            self.out_features = out_features

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.ones(*inputs.shape[:-1], self.out_features, dtype=inputs.dtype)

    selected = torch.tensor([0, 3, 7])
    mlp = module.SelectedRowsCq3Mlp(
        Linear(8), Linear(8), Linear(4), selected_rows=selected
    )
    output = mlp(torch.randn(2, 4, 256, dtype=torch.bfloat16)).reshape(8, 4)
    assert torch.count_nonzero(output[selected]).item() == 12
    unselected = torch.tensor([1, 2, 4, 5, 6])
    assert torch.count_nonzero(output[unselected]).item() == 0
    assert codes_gate.shape == (8, 256)
    assert scales_gate.shape == (8, 2)
    assert codes_down.shape == (256, 8)


def test_direct_screen_is_restartable_authenticated_and_sequentially_gated() -> None:
    source = SCRIPT.read_text()
    assert "validate_existing_layer(" in source
    assert "validate_or_write_direct_payload(" in source
    assert 'state.pop(target["state_name"])' in source
    assert "source_sha256" in source
    assert "payload did not round-trip" not in source
    assert "write_step_zero_cq3_tensor" in source
    assert '"calibration_rows_per_layer": 64' in source
    assert "maximum_direct_mixed_calibration_complete_block_nmse_per_layer" in source
    assert 'result.get("training_cache_sha256") != training_cache_sha256' in source
    assert "raise SystemExit(0 if gate[\"passed\"] else 2)" in source
    assert "fresh-confirmation score" in source
