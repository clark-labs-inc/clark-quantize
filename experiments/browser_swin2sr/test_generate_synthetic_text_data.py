import json
import tempfile
import unittest
from pathlib import Path

from generate_synthetic_text_data import (
    load_excluded_hashes,
    sanitize_text,
    select_disjoint_records,
)


class SyntheticTextDataTests(unittest.TestCase):
    def test_sanitize_text_repairs_cp1252_and_removes_controls(self):
        self.assertEqual(
            sanitize_text("1469\u00961539\u0000"),
            "1469–1539",
        )

    def test_load_excluded_hashes_requires_and_collects_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            manifest.write_text(
                json.dumps({"sha256": "train-a"}) + "\n"
                + json.dumps({"sha256": "train-b"}) + "\n"
            )
            self.assertEqual(
                load_excluded_hashes([manifest]),
                {"train-a", "train-b"},
            )
            manifest.write_text(json.dumps({"language": "eng_Latn"}) + "\n")
            with self.assertRaisesRegex(ValueError, "without sha256"):
                load_excluded_hashes([manifest])

    def test_select_disjoint_records_backfills_each_language(self):
        records = [
            {"language": "eng_Latn", "sha256": "train"},
            {"language": "eng_Latn", "sha256": "eng-a"},
            {"language": "eng_Latn", "sha256": "eng-b"},
            {"language": "fra_Latn", "sha256": "eng-a"},
            {"language": "fra_Latn", "sha256": "fra-a"},
            {"language": "fra_Latn", "sha256": "fra-b"},
        ]
        selected = select_disjoint_records(records, 2, {"train"})
        self.assertEqual(
            [(row["language"], row["sha256"]) for row in selected],
            [
                ("eng_Latn", "eng-a"),
                ("eng_Latn", "eng-b"),
                ("fra_Latn", "fra-a"),
                ("fra_Latn", "fra-b"),
            ],
        )

    def test_select_disjoint_records_fails_without_enough_reserve(self):
        records = [
            {"language": "eng_Latn", "sha256": "train"},
            {"language": "eng_Latn", "sha256": "eng-a"},
        ]
        with self.assertRaisesRegex(ValueError, "only has 1 disjoint"):
            select_disjoint_records(records, 2, {"train"})


if __name__ == "__main__":
    unittest.main()
