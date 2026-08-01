import json
import tempfile
import unittest
from pathlib import Path

from verify_sealed_sensitive_resume import latest_resume_start


class VerifySealedSensitiveResumeTests(unittest.TestCase):
    def test_finds_exact_resume_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "train.jsonl"
            log.write_text("\n".join((
                json.dumps({
                    "event": "start",
                    "start_surface_step": 14000,
                }),
                json.dumps({
                    "event": "validation",
                    "surface_step": 17000,
                }),
                json.dumps({
                    "event": "start",
                    "start_surface_step": 17000,
                    "exact_data_resume": True,
                }),
            )) + "\n")

            record = latest_resume_start(log, surface_step=17000)

            self.assertTrue(record["exact_data_resume"])

    def test_rejects_ambiguous_resume_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "train.jsonl"
            record = json.dumps({
                "event": "start",
                "start_surface_step": 17000,
            })
            log.write_text(record + "\n" + record + "\n")

            with self.assertRaisesRegex(RuntimeError, "unique"):
                latest_resume_start(log, surface_step=17000)


if __name__ == "__main__":
    unittest.main()
