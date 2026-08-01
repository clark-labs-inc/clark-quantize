from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from verify_materialized_device_parity import compare, state_sha256


class IdentityReconstruction(torch.nn.Module):
    def forward(self, values):
        return SimpleNamespace(reconstruction=values)


class MaterializedDeviceParityTest(unittest.TestCase):
    def test_state_hash_is_independent_of_mapping_order(self) -> None:
        first = {
            "a": torch.tensor([1.0, 2.0]),
            "b": torch.tensor([[3.0]]),
        }
        second = {"b": first["b"], "a": first["a"]}
        self.assertEqual(state_sha256(first), state_sha256(second))

    def test_state_hash_changes_with_tensor_value(self) -> None:
        first = {"weight": torch.tensor([1.0, 2.0])}
        second = {"weight": torch.tensor([1.0, 3.0])}
        self.assertNotEqual(state_sha256(first), state_sha256(second))

    def test_same_materialized_state_matches_on_cpu(self) -> None:
        samples = [
            (
                torch.rand(1, 3, 16, 16),
                torch.rand(1, 3, 16, 16),
            )
        ]
        with patch(
            "verify_materialized_device_parity.pretrained",
            side_effect=lambda _device: IdentityReconstruction(),
        ):
            metrics = compare({}, samples, "cpu", "cpu")
        self.assertEqual(metrics["maximum_absolute_error"], 0.0)
        self.assertEqual(metrics["minimum_output_psnr_db"], 99.0)
        self.assertEqual(metrics["mean_gt_psnr_delta_db"], 0.0)


if __name__ == "__main__":
    unittest.main()
