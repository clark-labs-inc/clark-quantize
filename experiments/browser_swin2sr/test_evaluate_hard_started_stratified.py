import json
import tempfile
import unittest
from pathlib import Path

from evaluate_hard_started_stratified import verify_development_inputs
from full_surface_provenance import file_sha256


class HardStartedStratifiedTests(unittest.TestCase):
    def test_development_inputs_are_bound_to_study_and_q4(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_path = root / "bundle.json"
            bundle_path.write_text('{"bundle":"sealed"}')
            q4_path = root / "q4.json"
            q4_path.write_text('{"q4":"sealed"}')
            bundle = {
                "manifest_sha256": "manifest",
                "samples": 2,
            }
            q4 = {
                "bundle_manifest_sha256": "manifest",
                "device": "mps",
                "samples": [{}, {}],
            }
            study = {
                "selection_source": "development",
                "bundle_sha256": file_sha256(bundle_path),
                "q4_reference_sha256": file_sha256(q4_path),
            }

            verify_development_inputs(
                study,
                bundle_path,
                bundle,
                q4_path,
                q4,
                "mps",
            )
            q4["device"] = "cpu"
            with self.assertRaisesRegex(RuntimeError, "device"):
                verify_development_inputs(
                    study,
                    bundle_path,
                    bundle,
                    q4_path,
                    q4,
                    "mps",
                )

    def test_non_development_study_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_path = root / "bundle.json"
            q4_path = root / "q4.json"
            bundle_path.write_text("{}")
            q4_path.write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "development"):
                verify_development_inputs(
                    {
                        "selection_source": "final",
                        "bundle_sha256": file_sha256(bundle_path),
                        "q4_reference_sha256": file_sha256(q4_path),
                    },
                    bundle_path,
                    {"manifest_sha256": "same", "samples": 0},
                    q4_path,
                    {
                        "bundle_manifest_sha256": "same",
                        "device": "mps",
                        "samples": [],
                    },
                    "mps",
                )


if __name__ == "__main__":
    unittest.main()
