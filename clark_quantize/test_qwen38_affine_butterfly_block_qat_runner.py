from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime/run_qwen38_27b_affine_butterfly_block_qat_v1.py"
)
SPEC = importlib.util.spec_from_file_location("qwen38_affine_block_qat", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_qat_runner_is_pinned_to_the_parent_and_qwen27b() -> None:
    assert MODULE.MODEL_ID == "Qwen/Qwen3.8-27B"
    assert MODULE.TARGET_LAYERS == (0, 32, 63)
    assert MODULE.PARENT_RUNNER_COMMIT == "1e524a9eb48e223b259a4d3714afc765bb107dcd"


def test_selected_nmse_uses_only_requested_rows() -> None:
    parent = torch.ones(2, 3, 4)
    observed = parent.clone()
    observed.reshape(-1, 4)[1] += 2
    observed.reshape(-1, 4)[4] += 1
    assert MODULE.selected_nmse(observed, parent, torch.tensor([4])) == 1.0
    assert MODULE.selected_nmse(observed, parent, torch.tensor([1])) == 4.0


def test_set_child_module_replaces_nested_and_root_children() -> None:
    root = torch.nn.Module()
    root.inner = torch.nn.Module()
    root.inner.first = torch.nn.Identity()
    root.second = torch.nn.Identity()
    nested = torch.nn.ReLU()
    direct = torch.nn.Sigmoid()
    MODULE.set_child_module(root, "inner.first", nested)
    MODULE.set_child_module(root, "second", direct)
    assert root.inner.first is nested
    assert root.second is direct


def test_qat_script_contains_no_four_b_model_identifier() -> None:
    text = SCRIPT.read_text()
    assert "Qwen3.8-27B" in text
    assert "Qwen3.5-4B" not in text
