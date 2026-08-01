import json
import tempfile
import unittest
from pathlib import Path

from validate_sensitive_io_rescue_coverage import validate_coverage


TARGET = "swin2sr.first_convolution"


def write_receipt(
    root: Path,
    name: str,
    *,
    group: int,
    multiplier: float,
    parent: str = "parent",
) -> Path:
    path = root / f"{name}.json"
    path.write_text(json.dumps({
        "format": "backdropkit-swin2sr-hard-started-validation-v2",
        "diagnostic_only": True,
        "device": "mps",
        "crop_size": 128,
        "study_manifest": f"/{name}/study.json",
        "study_manifest_sha256": f"study-{name}",
        "validation_files": [
            {"path": "/sample.png", "sha256": "sample"},
        ],
        "checkpoints": [{
            "surface_step": 19000,
            "checkpoint": f"/{name}.pt",
            "checkpoint_sha256": f"checkpoint-{name}",
            "surface_group_sizes": {TARGET: group},
            "convolution_learning_rate_multiplier": multiplier,
            "fork_provenance": {
                "parent_checkpoint_sha256": parent,
            },
            "blended": {"psnr_vs_ground_truth": 30.0},
            "hard_started": {
                "psnr_vs_ground_truth": 30.0,
                "psnr_vs_teacher": 35.0,
            },
        }],
    }))
    return path


class SensitiveIoCoverageTests(unittest.TestCase):
    def fixture(self, root: Path):
        control = write_receipt(
            root,
            "control",
            group=9,
            multiplier=0.25,
        )
        arms = [
            write_receipt(
                root,
                f"arm-{multiplier}",
                group=2,
                multiplier=multiplier,
            )
            for multiplier in (0.25, 1.0, 4.0)
        ]
        return control, arms

    def test_accepts_exact_preregistered_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            control, arms = self.fixture(Path(temporary))

            result = validate_coverage(control, arms, step=19000)

            self.assertTrue(result["exact_coverage_verified"])
            self.assertEqual(
                [
                    arm["convolution_learning_rate_multiplier"]
                    for arm in result["arms"]
                ],
                [0.25, 1.0, 4.0],
            )
            self.assertTrue(result["final_heldout_locked"])

    def test_rejects_missing_or_duplicate_arm(self):
        with tempfile.TemporaryDirectory() as temporary:
            control, arms = self.fixture(Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "exactly three"):
                validate_coverage(control, arms[:2], step=19000)
            arms[-1] = write_receipt(
                Path(temporary),
                "duplicate",
                group=2,
                multiplier=1.0,
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate LR"):
                validate_coverage(control, arms, step=19000)

    def test_rejects_wrong_control_or_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control, arms = self.fixture(root)
            control = write_receipt(
                root,
                "wrong-control",
                group=9,
                multiplier=1.0,
            )
            with self.assertRaisesRegex(RuntimeError, "matched 0.25"):
                validate_coverage(control, arms, step=19000)
            arms[-1] = write_receipt(
                root,
                "wrong-parent",
                group=2,
                multiplier=4.0,
                parent="other",
            )
            control = write_receipt(
                root,
                "control-correct",
                group=9,
                multiplier=0.25,
            )
            with self.assertRaisesRegex(RuntimeError, "share one parent"):
                validate_coverage(control, arms, step=19000)


if __name__ == "__main__":
    unittest.main()
