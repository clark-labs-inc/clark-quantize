from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluate_hard_started_checkpoints import (
    checkpoint_fork_provenance,
    restore_checkpoint_schedule,
    surface_amount_summary,
    verify_study_sources,
)
from full_surface_provenance import file_sha256


def surface(category: str, amount: float):
    return SimpleNamespace(
        category=category,
        quantizer=SimpleNamespace(amount=torch.tensor(amount)),
    )


class FakeQuantizer:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end
        self.amount = torch.tensor(0.0)

    def set_progress(self, progress: float):
        amount = (progress - self.start) / (self.end - self.start)
        self.amount.fill_(min(1.0, max(0.0, amount)))


class EvaluateHardStartedCheckpointsTest(unittest.TestCase):
    def test_surface_amount_summary_separates_stages(self):
        summary = surface_amount_summary([
            surface("attention", 1.0),
            surface("attention", 0.5),
            surface("output", 0.0),
        ])
        self.assertEqual(summary["surfaces"], 3)
        self.assertEqual(summary["started_surfaces"], 2)
        self.assertEqual(summary["transitioning_surfaces"], 1)
        self.assertEqual(summary["fully_hard_surfaces"], 1)
        self.assertEqual(
            summary["categories"]["attention"],
            {
                "surfaces": 2,
                "started": 2,
                "transitioning": 1,
                "fully_hard": 1,
            },
        )

    def test_checkpoint_schedule_is_reconstructed_from_surface_step(self):
        surfaces = [
            SimpleNamespace(
                category="core_linear",
                quantizer=FakeQuantizer(0.03, 0.15),
            ),
            SimpleNamespace(
                category="image_io_conv",
                quantizer=FakeQuantizer(0.80, 0.98),
            ),
        ]
        progress = restore_checkpoint_schedule(
            surfaces,
            {
                "surface_step": 5000,
                "surface_training": {
                    "schedule_steps": 24000,
                    "hard_core_from_start": False,
                },
            },
        )
        self.assertAlmostEqual(progress, 5000 / 24000)
        self.assertEqual(float(surfaces[0].quantizer.amount), 1.0)
        self.assertEqual(float(surfaces[1].quantizer.amount), 0.0)

    def test_retained_precision_checkpoint_restores_fully_hard_schedule(self):
        surfaces = [
            SimpleNamespace(
                category="core_linear",
                quantizer=FakeQuantizer(0.03, 0.15),
            ),
            SimpleNamespace(
                category="image_io_conv",
                quantizer=FakeQuantizer(0.80, 0.98),
            ),
        ]
        progress = restore_checkpoint_schedule(
            surfaces,
            {
                "retained_fp16_training": {
                    "island": "image_io_fp16",
                    "precision": "fp16",
                },
                "surface_training": {
                    "schedule_steps": 24000,
                    "hard_core_from_start": False,
                },
            },
        )
        self.assertEqual(progress, 1.0)
        self.assertTrue(all(float(item.quantizer.amount) == 1.0 for item in surfaces))

    def test_sealed_source_verification_rejects_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.py"
            source.write_text("original\n")
            study = {"source_files": {"source.py": file_sha256(source)}}
            self.assertEqual(
                verify_study_sources(study, root),
                {"source.py": file_sha256(source)},
            )
            source.write_text("changed\n")
            with self.assertRaisesRegex(RuntimeError, "sealed source changed"):
                verify_study_sources(study, root)

    def test_sealed_snapshot_can_verify_source_after_live_code_moves_on(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.py"
            source.write_text("original\n")
            study = {"source_files": {"source.py": file_sha256(source)}}
            sealed_root = root / "sealed"
            sealed_root.mkdir()
            sealed_source = sealed_root / "source.py"
            sealed_source.write_text(source.read_text())
            source.write_text("changed\n")
            self.assertEqual(
                verify_study_sources(study, root, sealed_root),
                {"source.py": file_sha256(sealed_source)},
            )

    def test_fork_lineage_is_recovered_from_sealed_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_hash = "a" * 64
            fork = root / "fork.pt"
            fork.write_bytes(b"fork")
            receipt = root / "fork-receipt.json"
            receipt.write_text(
                '{"parent_checkpoint_sha256":"' + parent_hash + '"}'
            )
            study = {
                "fork_checkpoint": str(fork),
                "fork_checkpoint_sha256": file_sha256(fork),
                "fork_receipt": str(receipt),
                "fork_receipt_sha256": file_sha256(receipt),
                "parent_checkpoint_sha256": parent_hash,
                "training_state_sha256": "b" * 64,
            }
            payload = {
                "surface_training": {
                    "resume_checkpoint": str(fork),
                },
            }

            provenance = checkpoint_fork_provenance(
                payload,
                study,
                root,
            )

            self.assertEqual(
                provenance["parent_checkpoint_sha256"],
                parent_hash,
            )
            self.assertTrue(
                provenance["recovered_from_sealed_study_manifest"]
            )

    def test_fork_lineage_rejects_a_different_resume_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fork = root / "fork.pt"
            fork.write_bytes(b"fork")
            other = root / "other.pt"
            other.write_bytes(b"other")
            receipt = root / "fork-receipt.json"
            receipt.write_text('{"parent_checkpoint_sha256":"parent"}')
            study = {
                "fork_checkpoint": str(fork),
                "fork_checkpoint_sha256": file_sha256(fork),
                "fork_receipt": str(receipt),
                "fork_receipt_sha256": file_sha256(receipt),
                "parent_checkpoint_sha256": "parent",
            }
            payload = {
                "surface_training": {
                    "resume_checkpoint": str(other),
                },
            }
            with self.assertRaisesRegex(RuntimeError, "does not descend"):
                checkpoint_fork_provenance(payload, study, root)


if __name__ == "__main__":
    unittest.main()
