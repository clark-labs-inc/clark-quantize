import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from evaluate_stratified import load_bundle_records
from prepare_eval_bundle import prepare_bundle


class EvalBundleTests(unittest.TestCase):
    def test_materializes_hashed_deterministic_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "source.png"
            Image.new("RGB", (192, 160), (20, 80, 140)).save(image)
            source_sha256 = hashlib.sha256(image.read_bytes()).hexdigest()
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({
                "local_path": image.name,
                "sha256": source_sha256,
                "source": "fixture",
                "split": "test",
            }) + "\n")
            first = root / "first"
            second = root / "second"

            first_metadata = prepare_bundle([manifest], first, 128, None, 7)
            second_metadata = prepare_bundle([manifest], second, 128, None, 7)

            self.assertEqual(first_metadata["samples"], 1)
            first_record = json.loads((first / "manifest.jsonl").read_text())
            second_record = json.loads((second / "manifest.jsonl").read_text())
            self.assertEqual(first_record["source_sha256"], source_sha256)
            self.assertEqual(first_record["low_sha256"], second_record["low_sha256"])
            self.assertEqual(first_record["high_sha256"], second_record["high_sha256"])
            with Image.open(first / first_record["low_path"]) as low:
                self.assertEqual(low.size, (64, 64))
            with Image.open(first / first_record["high_path"]) as high:
                self.assertEqual(high.size, (128, 128))
            loaded_metadata, loaded_records = load_bundle_records(first / "bundle.json")
            self.assertEqual(loaded_metadata["manifest_sha256"], first_metadata["manifest_sha256"])
            self.assertEqual(loaded_records[0]["sha256"], source_sha256)
            self.assertTrue(loaded_records[0]["sealed_low_path"].endswith(".png"))

    def test_rejects_non_swin_crop_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "divisible by 16"):
                prepare_bundle([], Path(temporary), 127, None, 7)


if __name__ == "__main__":
    unittest.main()
