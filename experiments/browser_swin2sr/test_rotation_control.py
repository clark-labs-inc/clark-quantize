from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rotation_control import block_hadamard
from artifact_state import hard_state_dict
from train_swin2sr_qat import add_ternary_parametrizations


class ToyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(12, 7)

    def forward(self, values):
        return self.projection(values)


class ToyStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = ToyAttention()

    def forward(self, values):
        return self.attention(values)


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage = ToyStage()

    def forward(self, values):
        return self.stage(values)


class RotationControlTest(unittest.TestCase):
    def test_hadamard_is_its_own_inverse(self) -> None:
        values = torch.randn(3, 5, 12)
        restored = block_hadamard(block_hadamard(values, 4), 4)
        torch.testing.assert_close(restored, values)

    def test_rotating_inputs_and_weight_columns_preserves_linear(self) -> None:
        inputs = torch.randn(2, 7, 12)
        weight = torch.randn(9, 12)
        bias = torch.randn(9)
        expected = F.linear(inputs, weight, bias)
        actual = F.linear(
            block_hadamard(inputs, 4),
            block_hadamard(weight, 4),
            bias,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_rejects_non_power_of_two_blocks(self) -> None:
        with self.assertRaisesRegex(ValueError, "power of two"):
            block_hadamard(torch.ones(2, 12), 3)

    def test_rotated_checkpoint_exports_equivalent_dense_function(self) -> None:
        source = ToyModel()
        dense_reference = deepcopy(source)
        values = torch.randn(2, 5, 12)
        quantizers = add_ternary_parametrizations(
            source,
            "ternary",
            4,
            "optimal_mse",
            "hadamard",
            4,
        )

        torch.testing.assert_close(source(values), dense_reference(values))
        state = hard_state_dict(source, quantizers)
        rotated_hard = source(values)
        exported = ToyModel()
        exported.load_state_dict(state)
        torch.testing.assert_close(
            exported(values),
            rotated_hard,
            rtol=1e-5,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
