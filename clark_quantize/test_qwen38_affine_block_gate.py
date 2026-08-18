from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime/run_qwen38_27b_affine_butterfly_block_gate_v1.py"
)
SPEC = importlib.util.spec_from_file_location("qwen38_affine_block_gate", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_site_layout_covers_every_selected_block_matrix() -> None:
    assert MODULE.site_layout("linear_attention") == {
        "attention_input": (
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
        ),
        "attention_output": ("linear_attn.out_proj.weight",),
        "mlp_input": ("mlp.gate_proj.weight", "mlp.up_proj.weight"),
        "mlp_down": ("mlp.down_proj.weight",),
    }
    full = MODULE.site_layout("full_attention")
    assert full["attention_input"] == (
        "self_attn.k_proj.weight",
        "self_attn.q_proj.weight",
        "self_attn.v_proj.weight",
    )
    assert full["attention_output"] == ("self_attn.o_proj.weight",)


def test_pairing_seed_is_site_specific_and_deterministic() -> None:
    first = MODULE.pairing_seed(20260818, 32, "mlp_input")
    assert first == MODULE.pairing_seed(20260818, 32, "mlp_input")
    assert first != MODULE.pairing_seed(20260818, 0, "mlp_input")
    assert first != MODULE.pairing_seed(20260818, 32, "attention_input")


def test_depthwise_affine_uses_each_rows_own_activation_metric() -> None:
    torch.manual_seed(113)
    weight = torch.randn(7, 4) * 0.2 + 0.1
    windows = torch.randn(7, 12, 4)
    fitted = MODULE.fit_depthwise_conv_affine(weight, windows)
    target = (windows * weight[:, None, :]).sum(-1)
    observed = (
        windows * fitted.reconstruct()[:, None, :]
    ).sum(-1)
    assert fitted.codes.shape == weight.shape
    assert fitted.mu.shape == (7, 1)
    assert fitted.alpha.shape == (7, 1)
    assert set(fitted.codes.unique().tolist()) <= {-1, 0, 1}
    assert torch.isfinite(observed).all()
    assert float((observed - target).square().sum()) < float(target.square().sum())


def test_validation_nmse_uses_only_frozen_validation_rows() -> None:
    parent = torch.ones(2, 3, 4)
    observed = parent.clone()
    observed.reshape(-1, 4)[1] += 2
    observed.reshape(-1, 4)[4] += 1
    assert MODULE.validation_nmse(observed, parent, torch.tensor([4])) == 1.0
    assert MODULE.validation_nmse(observed, parent, torch.tensor([1])) == 4.0
