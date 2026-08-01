import unittest
from pathlib import Path

import torch

from fork_surface_group_map_checkpoint import fork_payload


class SurfaceGroupMapForkTests(unittest.TestCase):
    def payload(self):
        quantized = ["input", "output", "candidate"] + [
            f"surface_{index}" for index in range(201)
        ]
        return {
            "model": {"weight": torch.tensor([1.0])},
            "optimizer": {"state": {}, "param_groups": []},
            "run_state": {"cursor": 8},
            "surface_step": 26000,
            "quantized_modules": quantized,
            "quantization": {
                "scope": "all_matrix_and_conv",
                "surface_group_sizes": {"input": 2, "output": 2},
            },
            "surface_training": {
                "surface_group_sizes": {"input": 2, "output": 2},
            },
        }

    def test_fork_adds_map_without_changing_training_state(self):
        payload, receipt = fork_payload(
            self.payload(),
            parent_checkpoint=Path("parent.pt"),
            parent_sha256="abc",
            surface_map={"candidate": 3},
            expect_surface_step=26000,
        )

        self.assertEqual(
            payload["quantization"]["surface_group_sizes"],
            {"candidate": 3, "input": 2, "output": 2},
        )
        self.assertEqual(receipt["added_surface_group_sizes"], {"candidate": 3})

    def test_fork_rejects_replacing_existing_override(self):
        with self.assertRaisesRegex(ValueError, "replaces existing"):
            fork_payload(
                self.payload(),
                parent_checkpoint=Path("parent.pt"),
                parent_sha256="abc",
                surface_map={"input": 3},
                expect_surface_step=26000,
            )


if __name__ == "__main__":
    unittest.main()
