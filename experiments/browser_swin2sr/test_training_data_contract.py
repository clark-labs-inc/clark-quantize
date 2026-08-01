import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from training_data_contract import dataset_fingerprints, validation_receipts


class TrainingDataContractTests(unittest.TestCase):
    def test_fingerprints_data_and_validated_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.json"
            manifest = root / "manifest.jsonl"
            summary.write_text('{"images":2,"source":"fixture"}\n')
            manifest.write_text('{"sha256":"a"}\n{"sha256":"b"}\n')
            datasets = dataset_fingerprints([root])
            receipt_path = root / "validation.json"
            receipt_path.write_text(json.dumps({
                "format": "backdropkit-swin2sr-training-corpus-receipt-v1",
                "root": str(root),
                "summary_sha256": hashlib.sha256(
                    summary.read_bytes()
                ).hexdigest(),
                "manifest_sha256": hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest(),
                "images": 2,
                "languages": 2,
                "heldout_hash_overlap": 0,
            }))

            receipts = validation_receipts([receipt_path], datasets)

            self.assertEqual(datasets[0]["images"], 2)
            self.assertEqual(receipts[0]["heldout_hash_overlap"], 0)

    def test_rejects_receipt_for_changed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text('{"images":1}\n')
            (root / "manifest.jsonl").write_text('{"sha256":"a"}\n')
            datasets = dataset_fingerprints([root])
            receipt = root / "validation.json"
            receipt.write_text(json.dumps({
                "format": "backdropkit-swin2sr-training-corpus-receipt-v1",
                "root": str(root),
                "summary_sha256": datasets[0]["summary_sha256"],
                "manifest_sha256": "changed",
            }))

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validation_receipts([receipt], datasets)


if __name__ == "__main__":
    unittest.main()
