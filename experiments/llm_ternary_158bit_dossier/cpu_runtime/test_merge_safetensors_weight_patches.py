from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from experiments.llm_ternary_158bit_dossier.cpu_runtime.merge_safetensors_weight_patches import (
    merge_patches,
)


def test_merge_disjoint_patches(tmp_path: Path) -> None:
    base = tmp_path / "base.safetensors"
    addition = tmp_path / "addition.safetensors"
    out = tmp_path / "merged.safetensors"
    receipt = tmp_path / "merge.json"
    save_file({"layer.30.weight": torch.tensor([1.0, 2.0])}, str(base))
    save_file({"layer.29.weight": torch.tensor([3.0])}, str(addition))

    result = merge_patches(
        base_path=base,
        addition_path=addition,
        out_path=out,
        receipt_path=receipt,
        expected_tensor_count=2,
        required_key_regex=r"layer[.](29|30)[.]weight",
    )

    merged = load_file(str(out), device="cpu")
    assert sorted(merged) == ["layer.29.weight", "layer.30.weight"]
    assert torch.equal(merged["layer.29.weight"], torch.tensor([3.0]))
    assert torch.equal(merged["layer.30.weight"], torch.tensor([1.0, 2.0]))
    assert result["valid"] is True
    assert result["out"]["tensor_count"] == 2
    assert result["overlap"] == []
    assert receipt.is_file()


def test_merge_rejects_overlap(tmp_path: Path) -> None:
    base = tmp_path / "base.safetensors"
    addition = tmp_path / "addition.safetensors"
    save_file({"same": torch.tensor([1])}, str(base))
    save_file({"same": torch.tensor([2])}, str(addition))

    with pytest.raises(ValueError, match="overlap"):
        merge_patches(
            base_path=base,
            addition_path=addition,
            out_path=tmp_path / "out.safetensors",
            receipt_path=tmp_path / "receipt.json",
            expected_tensor_count=0,
            required_key_regex=None,
        )


def test_merge_rejects_out_of_scope_key(tmp_path: Path) -> None:
    base = tmp_path / "base.safetensors"
    addition = tmp_path / "addition.safetensors"
    save_file({"layer.30.weight": torch.tensor([1])}, str(base))
    save_file({"layer.28.weight": torch.tensor([2])}, str(addition))

    with pytest.raises(ValueError, match="out-of-scope"):
        merge_patches(
            base_path=base,
            addition_path=addition,
            out_path=tmp_path / "out.safetensors",
            receipt_path=tmp_path / "receipt.json",
            expected_tensor_count=2,
            required_key_regex=r"layer[.](29|30)[.]weight",
        )

