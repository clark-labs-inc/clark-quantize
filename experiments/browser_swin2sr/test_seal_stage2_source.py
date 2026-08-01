import json
import tempfile
import unittest
from pathlib import Path

from seal_stage2_source import build_gate


class StageTwoSourceGateTests(unittest.TestCase):
    def inputs(self, root: Path, completed: int = 24):
        manifest = root / "stage1-manifest.json"
        summary = root / "stage1-summary.json"
        training = root / "training-receipt.json"
        manifest.write_text(json.dumps({
            "format": "backdropkit-swin2sr-recovery-factorial-v1",
            "stage": "stage1",
            "config": {"data": ["/old/ui", "/old/text"]},
        }))
        summary.write_text(json.dumps({
            "completed": completed,
            "failed": 0,
            "ranking": [{} for _ in range(completed)],
        }))
        training.write_text(json.dumps({
            "format": "backdropkit-swin2sr-training-corpus-receipt-v1",
            "root": "/glyph-safe",
            "manifest_sha256": "glyph-safe-manifest",
            "images": 102000,
            "languages": 204,
            "heldout_hash_overlap": 0,
        }))
        return manifest, summary, training

    def test_gate_requires_explicit_reset_and_seals_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, summary, training = self.inputs(Path(directory))

            gate = build_gate(manifest, summary, training)

            self.assertTrue(gate["reset_data_stream_required"])
            self.assertEqual(gate["stage2_training_root"], "/glyph-safe")
            self.assertEqual(len(gate["source_files"]), 9)

    def test_incomplete_stage_one_cannot_open_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, summary, training = self.inputs(Path(directory), 23)

            with self.assertRaisesRegex(ValueError, "not complete"):
                build_gate(manifest, summary, training)


if __name__ == "__main__":
    unittest.main()
