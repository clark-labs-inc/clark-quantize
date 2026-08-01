import json
import tempfile
import unittest
from pathlib import Path

import torch

from materialize_surface_states import load_materialized_state, state_receipt


class RetainedPrecisionRuntimeTests(unittest.TestCase):
    def test_generic_loader_accepts_versioned_retained_runtime_state(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            state = {"weight": torch.tensor([1.0, 0.0, -1.0])}
            state_path = root / "runtime-hard.pt"
            torch.save(state, state_path)
            manifest = root / "materialization.json"
            manifest.write_text(json.dumps({
                "format": "backdropkit-swin2sr-retained-precision-materialization-v1",
                "runtime_state": state_receipt(state_path, state),
            }))
            loaded_manifest, loaded = load_materialized_state(
                manifest, "runtime_state"
            )
            self.assertEqual(loaded_manifest["format"].split("-")[-1], "v1")
            self.assertTrue(torch.equal(loaded["weight"], state["weight"]))

    def test_generic_loader_accepts_mixed_fp16_island_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            state = {
                "trunk.weight": torch.tensor([1.0, 0.0, -1.0]),
                "input.weight": torch.tensor([0.125], dtype=torch.float32),
            }
            state_path = root / "runtime-hard.pt"
            torch.save(state, state_path)
            manifest = root / "materialization.json"
            manifest.write_text(json.dumps({
                "format": "backdropkit-swin2sr-retained-precision-materialization-v1",
                "retained_precision_island": {
                    "name": "image_io_fp16",
                    "precision": "fp16",
                    "weight_values": 1,
                    "weights": ["input.weight"],
                    "storage_dtype": "float16",
                },
                "runtime_state": state_receipt(state_path, state),
            }))
            loaded_manifest, loaded = load_materialized_state(
                manifest, "runtime_state"
            )
            self.assertEqual(
                loaded_manifest["retained_precision_island"]["precision"],
                "fp16",
            )
            self.assertTrue(torch.equal(loaded["input.weight"], state["input.weight"]))


if __name__ == "__main__":
    unittest.main()
