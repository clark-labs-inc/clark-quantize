import tempfile
import unittest
from pathlib import Path

from evaluate_output_conv_stratified import (
    GATE_FORMAT,
    verify_output_gate_sources,
)
from full_surface_provenance import file_sha256


class EvaluateOutputConvStratifiedTests(unittest.TestCase):
    def test_verifies_output_gate_format_and_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.py"
            source.write_text("value = 1\n")
            gate = {
                "format": GATE_FORMAT,
                "selection_source": "development",
                "model_promotion_authorized": False,
                "final_heldout_locked": True,
                "source_files": {"source.py": file_sha256(source)},
            }

            self.assertEqual(
                verify_output_gate_sources(gate, root),
                {"source.py": file_sha256(source)},
            )

    def test_rejects_unlocked_or_changed_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.py"
            source.write_text("value = 1\n")
            gate = {
                "format": GATE_FORMAT,
                "selection_source": "development",
                "model_promotion_authorized": False,
                "final_heldout_locked": False,
                "source_files": {"source.py": file_sha256(source)},
            }
            with self.assertRaisesRegex(RuntimeError, "lock"):
                verify_output_gate_sources(gate, root)

            gate["final_heldout_locked"] = True
            source.write_text("value = 2\n")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                verify_output_gate_sources(gate, root)


if __name__ == "__main__":
    unittest.main()
