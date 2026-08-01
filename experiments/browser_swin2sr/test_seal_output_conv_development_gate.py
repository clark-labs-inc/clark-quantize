import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from full_surface_provenance import file_sha256
from seal_output_conv_development_gate import (
    build_manifest,
    matched_output_studies,
)


def write_study(
    root: Path,
    group: int,
    *,
    parent: str = "parent",
    input_group: int = 2,
    steps: int = 24000,
) -> Path:
    path = root / f"study-g{group}.json"
    path.write_text(json.dumps({
        "selection_source": "development",
        "final_heldout_locked": True,
        "bundle": str(root / "bundle.json"),
        "bundle_sha256": file_sha256(root / "bundle.json"),
        "q4_reference": str(root / "q4.json"),
        "q4_reference_sha256": file_sha256(root / "q4.json"),
        "parent_checkpoint_sha256": parent,
        "training_data": [{"path": "/data", "sha256": "data"}],
        "training_data_receipts": ["/data/receipt.json"],
        "training_state_sha256": {
            "model": "model",
            "optimizer_state": "optimizer",
            "run_state": "run",
        },
        "surface_group_sizes": {
            "swin2sr.first_convolution": input_group,
            "upsample.conv": group,
        },
        "arms": {
            f"output-g{group}": {
                "training_command": [
                    "python",
                    "train.py",
                    "--steps",
                    str(steps),
                    "--schedule-steps",
                    "24000",
                ],
            },
        },
    }))
    return path


class OutputConvGateTests(unittest.TestCase):
    def fixture(self, root: Path) -> list[Path]:
        (root / "bundle.json").write_text('{"bundle": true}\n')
        (root / "q4.json").write_text('{"q4": true}\n')
        return [write_study(root, group) for group in (9, 3, 2)]

    def test_requires_exact_matched_output_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.fixture(root)

            matched = matched_output_studies(
                paths,
                selection_step=24000,
            )

            self.assertEqual(
                set(matched["study_manifests_by_output_group"]),
                {"2", "3", "9"},
            )
            self.assertEqual(
                matched["non_output_surface_group_sizes"],
                {"swin2sr.first_convolution": 2},
            )

    def test_rejects_different_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.fixture(root)
            paths[-1] = write_study(root, 2, parent="different")

            with self.assertRaisesRegex(RuntimeError, "parent_checkpoint"):
                matched_output_studies(paths, selection_step=24000)

    def test_rejects_non_output_surface_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.fixture(root)
            paths[-1] = write_study(root, 2, input_group=3)

            with self.assertRaisesRegex(RuntimeError, "outside target"):
                matched_output_studies(paths, selection_step=24000)

    def test_rejects_wrong_selection_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.fixture(root)
            paths[-1] = write_study(root, 2, steps=23000)

            with self.assertRaisesRegex(RuntimeError, "selection step"):
                matched_output_studies(paths, selection_step=24000)

    def test_builds_locked_development_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.fixture(root)
            with patch(
                "seal_output_conv_development_gate.SOURCE_FILES",
                (),
            ):
                manifest = build_manifest(
                    paths,
                    repository=root,
                    selection_step=24000,
                    bootstrap_samples=10000,
                    seed=20260730,
                    source_margin_db=0.10,
                    mean_margin_db=0.10,
                )

            self.assertFalse(manifest["model_promotion_authorized"])
            self.assertTrue(manifest["final_heldout_locked"])
            self.assertEqual(manifest["control_group_size"], 9)
            self.assertEqual(manifest["candidate_group_sizes"], [2, 3])


if __name__ == "__main__":
    unittest.main()
