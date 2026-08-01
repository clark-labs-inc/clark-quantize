import json
import tempfile
import unittest
from pathlib import Path

from full_surface_provenance import file_sha256
from run_sealed_sensitive_resume import (
    lineage_preserving_save,
    validate_trainer_arguments,
    verify_checkpoint,
    verify_sealed_sources,
)


class SealedSensitiveResumeTests(unittest.TestCase):
    def test_verifies_exact_checkpoint_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({
                "format": "backdropkit-swin2sr-hard-started-validation-v2",
                "checkpoints": [{
                    "surface_step": 17000,
                    "checkpoint_sha256": file_sha256(checkpoint),
                }],
            }))

            record = verify_checkpoint(
                checkpoint,
                receipt,
                surface_step=17000,
            )

            self.assertEqual(record["surface_step"], 17000)

    def test_verifies_sealed_source_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            sealed = repository / "sealed"
            source = sealed / "source.py"
            source.parent.mkdir()
            source.write_text("value = 1\n")
            manifest = repository / "study.json"
            study = {
                "selection_source": "development",
                "final_heldout_locked": True,
                "sealed_source_root": str(sealed),
                "source_files": {"source.py": file_sha256(source)},
            }
            manifest.write_text(json.dumps(study))

            actual = verify_sealed_sources(
                study,
                manifest,
                repository,
            )

            self.assertEqual(actual, sealed)
            source.write_text("value = 2\n")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                verify_sealed_sources(study, manifest, repository)

    def test_requires_exact_resume_command(self):
        checkpoint = Path("/study/checkpoint.pt")
        arguments = [
            "--base-checkpoint",
            str(checkpoint),
            "--steps",
            "19000",
            "--schedule-steps",
            "24000",
            "--device",
            "mps",
        ]

        target = validate_trainer_arguments(
            arguments,
            checkpoint,
            surface_step=17000,
        )

        self.assertEqual(target, 19000)
        arguments[arguments.index("--schedule-steps") + 1] = "23000"
        with self.assertRaisesRegex(RuntimeError, "hard schedule"):
            validate_trainer_arguments(
                arguments,
                checkpoint,
                surface_step=17000,
            )

    def test_checkpoint_metadata_preserves_direct_fork_and_resume_chain(self):
        captured = {}

        def original(path, *positional, **keywords):
            captured.update(keywords["config"])

        save = lineage_preserving_save(
            original,
            original_fork="/study/fork-step14000.pt",
            checkpoint=Path("/study/checkpoint-step17000.pt"),
            checkpoint_sha256="step17",
        )
        save(
            Path("/study/checkpoint-step18000.pt"),
            config={"resume_checkpoint": "/study/checkpoint-step17000.pt"},
        )

        self.assertEqual(
            captured["resume_checkpoint"],
            "/study/fork-step14000.pt",
        )
        self.assertEqual(captured["resume_chain"], [{
            "checkpoint": "/study/checkpoint-step17000.pt",
            "checkpoint_sha256": "step17",
        }])


if __name__ == "__main__":
    unittest.main()
