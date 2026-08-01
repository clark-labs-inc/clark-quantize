import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from merge_runtime_receipts import merge


class MergeRuntimeReceiptsTests(unittest.TestCase):
    def test_merges_unique_surfaces_for_one_candidate(self):
        merged = merge([
            {
                "model_sha256": "candidate",
                "fixture_sha256": "fixture",
                "fixture_hard_state_sha256": "hard",
                "receipts": [
                    {"surface": "chromium-webgpu", "passed": True},
                    {"surface": "chromium-wasm", "passed": True},
                ],
            },
            {
                "model_sha256": "candidate",
                "fixture_sha256": "fixture",
                "fixture_hard_state_sha256": "hard",
                "receipts": [{"surface": "safari-wasm", "passed": True}],
            },
        ])

        self.assertEqual(merged["model_sha256"], "candidate")
        self.assertEqual(merged["fixture_sha256"], "fixture")
        self.assertEqual(merged["fixture_hard_state_sha256"], "hard")
        self.assertEqual(
            [receipt["surface"] for receipt in merged["receipts"]],
            ["chromium-wasm", "chromium-webgpu", "safari-wasm"],
        )

    def test_rejects_mixed_candidates(self):
        with self.assertRaisesRegex(ValueError, "one candidate"):
            merge([
                {
                    "model_sha256": "a",
                    "fixture_sha256": "fixture",
                    "fixture_hard_state_sha256": "hard",
                    "receipts": [],
                },
                {
                    "model_sha256": "b",
                    "fixture_sha256": "fixture",
                    "fixture_hard_state_sha256": "hard",
                    "receipts": [],
                },
            ])

    def test_rejects_duplicate_surface(self):
        payload = {
            "model_sha256": "candidate",
            "fixture_sha256": "fixture",
            "fixture_hard_state_sha256": "hard",
            "receipts": [{"surface": "chromium-wasm", "passed": True}],
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            merge([payload, payload])

    def test_rejects_mixed_fixtures(self):
        first = {
            "model_sha256": "candidate",
            "fixture_sha256": "fixture-a",
            "fixture_hard_state_sha256": "hard",
            "receipts": [],
        }
        second = {
            **first,
            "fixture_sha256": "fixture-b",
        }
        with self.assertRaisesRegex(ValueError, "one fixture"):
            merge([first, second])


if __name__ == "__main__":
    unittest.main()
