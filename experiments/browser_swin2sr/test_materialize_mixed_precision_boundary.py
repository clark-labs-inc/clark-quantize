import json
import tempfile
import unittest
from pathlib import Path
import torch

from materialize_mixed_precision_boundary import materialize
from materialize_surface_states import RETAINED_PRECISION_FORMAT, state_receipt


class MixedPrecisionMaterializationTests(unittest.TestCase):
    def test_replaces_only_declared_unrotated_rstb_surface(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_dense = {"rstb.weight": torch.tensor([0.0]), "core.weight": torch.tensor([1.0])}
            base_runtime = {"rstb.weight": torch.tensor([0.0]), "core.weight": torch.tensor([2.0])}
            mixed = {"rstb.weight": torch.tensor([0.5]), "core.weight": torch.tensor([1.0])}
            base_dense_path = root / "base-dense.pt"
            base_runtime_path = root / "base-runtime.pt"
            mixed_path = root / "mixed.pt"
            torch.save(base_dense, base_dense_path)
            torch.save(base_runtime, base_runtime_path)
            torch.save(mixed, mixed_path)
            base = {
                "format": RETAINED_PRECISION_FORMAT,
                "quantization": {"mode": "ternary"},
                "surface_training": {},
                "surfaces": {
                    "rstb.weight": {"category": "residual_body_conv", "rotation": "none"},
                    "core.weight": {"category": "core_linear", "rotation": "hadamard"},
                },
                "retained_precision_islands": [],
                "dense_state": state_receipt(base_dense_path, base_dense),
                "runtime_state": state_receipt(base_runtime_path, base_runtime),
            }
            base_path = root / "base.json"
            base_path.write_text(json.dumps(base))
            boundary = {
                "format": "backdropkit-swin2sr-rstb-precision-boundary-materialization-v1",
                "arm": "int7_per_output",
                "changed_tensors": ["rstb.weight"],
                "storage": {"data_bytes": 1, "incremental_data_bytes": 1},
                "dense_state": state_receipt(mixed_path, mixed),
            }
            boundary_path = root / "boundary.json"
            boundary_path.write_text(json.dumps(boundary))

            receipt = materialize(boundary_path, base_path, root / "out")

            runtime = torch.load(
                receipt["runtime_state"]["path"], map_location="cpu", weights_only=True
            )
            self.assertEqual(set(receipt["surfaces"]), {"core.weight"})
            self.assertEqual(runtime["rstb.weight"].item(), 0.5)
            self.assertEqual(runtime["core.weight"].item(), 2.0)
            self.assertEqual(
                receipt["retained_precision_islands"][-1]["precision"], "int7"
            )


if __name__ == "__main__":
    unittest.main()
