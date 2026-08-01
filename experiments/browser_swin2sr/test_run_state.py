"""Tests for durable Swin2SR training state."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from run_state import (
    atomic_torch_save,
    capture_run_state,
    epoch_order,
    restore_run_state,
    save_training_checkpoint,
)


class RunStateTest(unittest.TestCase):
    def test_epoch_order_is_reproducible_and_epoch_specific(self) -> None:
        images = [Path(f"{index}.png") for index in range(20)]
        self.assertEqual(epoch_order(images, 42, 3), epoch_order(images, 42, 3))
        self.assertNotEqual(epoch_order(images, 42, 3), epoch_order(images, 42, 4))

    def test_capture_and_restore_replays_random_streams(self) -> None:
        augmentation_rng = random.Random(7)
        augmentation_rng.random()
        state = capture_run_state(augmentation_rng, 2, 17, 12.5)
        expected_python = augmentation_rng.random()
        expected_numpy = np.random.random()
        expected_torch = torch.rand(1).item()

        restored_rng = random.Random(999)
        restored = restore_run_state(
            {"run_state": state},
            restored_rng,
            examples_seen=0,
            image_count=100,
        )

        self.assertEqual(restored, (2, 17, 12.5, True))
        self.assertEqual(restored_rng.random(), expected_python)
        self.assertEqual(np.random.random(), expected_numpy)
        self.assertEqual(torch.rand(1).item(), expected_torch)

    def test_legacy_checkpoint_has_deterministic_cursor_fallback(self) -> None:
        restored = restore_run_state(
            {},
            random.Random(1),
            examples_seen=237,
            image_count=100,
        )
        self.assertEqual(restored, (2, 37, 0.0, False))

    def test_atomic_save_replaces_destination_and_cleans_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            atomic_torch_save({"step": 3}, path)
            atomic_torch_save({"step": 4}, path)
            self.assertEqual(torch.load(path)["step"], 4)
            self.assertEqual(list(Path(directory).glob(".*.tmp-*")), [])

    def test_checkpoint_records_optimizer_identity(self) -> None:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        quantizer = type("Quantizer", (), {
            "mode": "ternary",
            "method": "optimal_mse",
            "group_size": 60,
            "rotation": "none",
            "rotation_block": 4,
        })()
        config = {"name": "adamw", "learning_rate": 2e-5}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_training_checkpoint(
                path,
                model,
                optimizer,
                [("projection", quantizer, model)],
                7,
                {"psnr": 42.0},
                {},
                config,
            )
            checkpoint = torch.load(path)

        self.assertEqual(checkpoint["optimizer_config"], config)


if __name__ == "__main__":
    unittest.main()
