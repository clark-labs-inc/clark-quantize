import json
import tempfile
import unittest
from pathlib import Path

from seal_sensitive_io_development_gate import shared_study_inputs


class SealSensitiveIoDevelopmentGateTests(unittest.TestCase):
    def test_matched_studies_share_development_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index in range(2):
                path = root / f"study-{index}.json"
                path.write_text(json.dumps({
                    "selection_source": "development",
                    "bundle": "/bundle.json",
                    "bundle_sha256": "bundle",
                    "q4_reference": "/q4.json",
                    "q4_reference_sha256": "q4",
                    "parent_checkpoint_sha256": "parent",
                }))
                paths.append(path)

            shared = shared_study_inputs(paths)

            self.assertEqual(shared["parent_checkpoint_sha256"], "parent")

    def test_mismatched_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index, parent in enumerate(("first", "second")):
                path = root / f"study-{index}.json"
                path.write_text(json.dumps({
                    "selection_source": "development",
                    "bundle": "/bundle.json",
                    "bundle_sha256": "bundle",
                    "q4_reference": "/q4.json",
                    "q4_reference_sha256": "q4",
                    "parent_checkpoint_sha256": parent,
                }))
                paths.append(path)

            with self.assertRaisesRegex(RuntimeError, "parent_checkpoint"):
                shared_study_inputs(paths)


if __name__ == "__main__":
    unittest.main()
