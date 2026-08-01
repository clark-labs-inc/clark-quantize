import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from prepare_disjoint_eval_manifest import build_disjoint_manifest


def write_manifest(root: Path, name: str, rows: list[tuple[str, str, bytes]]) -> Path:
    directory = root / name
    directory.mkdir()
    records = []
    for index, (language, filename, data) in enumerate(rows):
        path = directory / filename
        path.write_bytes(data)
        records.append({
            "language": language,
            "record_id": index,
            "local_path": filename,
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    manifest = directory / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    return manifest


class DisjointEvalManifestTests(unittest.TestCase):
    def test_filters_training_overlap_and_backfills_language(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = write_manifest(
                root,
                "training",
                [("eng_Latn", "train.png", b"overlap")],
            )
            original = write_manifest(
                root,
                "original",
                [
                    ("eng_Latn", "overlap.png", b"overlap"),
                    ("fra_Latn", "fra.png", b"fra"),
                ],
            )
            replacements = write_manifest(
                root,
                "replacements",
                [("eng_Latn", "eng.png", b"eng")],
            )
            out = root / "disjoint" / "manifest.jsonl"
            receipt = build_disjoint_manifest(
                [training],
                [original, replacements],
                out,
                per_language=1,
                expected_languages=2,
            )
            self.assertEqual(receipt["samples"], 2)
            self.assertEqual(receipt["rejected_training_overlap"], 1)
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(
                {row["language"] for row in rows},
                {"eng_Latn", "fra_Latn"},
            )
            self.assertTrue(all(Path(row["local_path"]).is_absolute() for row in rows))

    def test_fails_when_language_balance_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = write_manifest(root, "training", [])
            candidates = write_manifest(
                root,
                "candidates",
                [("eng_Latn", "eng.png", b"eng")],
            )
            with self.assertRaisesRegex(ValueError, "expected 2 languages"):
                build_disjoint_manifest(
                    [training],
                    [candidates],
                    root / "out" / "manifest.jsonl",
                    per_language=1,
                    expected_languages=2,
                )


if __name__ == "__main__":
    unittest.main()
