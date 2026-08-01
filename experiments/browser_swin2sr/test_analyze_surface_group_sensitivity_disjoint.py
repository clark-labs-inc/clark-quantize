import json
import tempfile
import unittest
from pathlib import Path

from analyze_surface_group_sensitivity_disjoint import load_exclusions


class DisjointSurfaceGroupSensitivityTests(unittest.TestCase):
    def test_exclusions_deduplicate_across_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(json.dumps({
                "samples": [{"sha256": "a"}, {"sha256": "b"}],
            }))
            second.write_text(json.dumps({
                "samples": [{"sha256": "b"}, {"sha256": "c"}],
            }))

            identities, receipts = load_exclusions([first, second])

            self.assertEqual(identities, {"a", "b", "c"})
            self.assertEqual(len(receipts), 2)


if __name__ == "__main__":
    unittest.main()
