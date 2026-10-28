from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
QWEN_DIR = Path(__file__).resolve().parent
if str(QWEN_DIR) not in sys.path:
    sys.path.insert(0, str(QWEN_DIR))

from fake_quant_policy import resolve_parameter_name
from weight_patch import apply_weight_patch


def main() -> None:
    parameter = torch.nn.Parameter(torch.ones(3))
    params = {"model.language_model.layers.3.input_layernorm.weight": parameter}
    assert resolve_parameter_name("model.language_model.layers.3.input_layernorm.weight", params) == next(iter(params))
    assert resolve_parameter_name("model.language_model.layers.3.layer.input_layernorm.weight", params) == next(iter(params))

    shortened = {"model.layers.3.input_layernorm.weight": parameter}
    assert resolve_parameter_name("model.language_model.layers.3.layer.input_layernorm.weight", shortened) == next(iter(shortened))

    ambiguous = {
        "model.layers.3.input_layernorm.weight": parameter,
        "model.language_model.layers.3.input_layernorm.weight": parameter,
    }
    try:
        resolve_parameter_name("model.language_model.layers.3.layer.input_layernorm.weight", ambiguous)
    except ValueError as exc:
        assert "ambiguous parameter mapping" in str(exc)
    else:
        raise AssertionError("ambiguous wrapper mapping must fail closed")

    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 2, bias=False))
    original_zero = model[0].weight.detach().clone()
    with tempfile.TemporaryDirectory() as temporary:
        patch = Path(temporary) / "patch.safetensors"
        save_file({"0.weight": torch.full((2, 2), 3.0), "1.weight": torch.full((2, 2), 4.0)}, patch)
        result = apply_weight_patch(model, patch, include_regex=r"^1\.weight$")
        assert result["applied_count"] == 1
        assert result["skipped_count"] == 1
        assert torch.equal(model[0].weight, original_zero)
        assert torch.equal(model[1].weight, torch.full_like(model[1].weight, 4.0))
    print("weight patch name tests passed")


if __name__ == "__main__":
    main()
