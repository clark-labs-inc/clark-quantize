from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/qwen38_27b_spot/qat_teacher_cache.py"
SPEC = importlib.util.spec_from_file_location("qwen38_qat_teacher_cache", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class QATTeacherCacheTests(unittest.TestCase):
    def row(self, tokens: int = 4, top_k: int = 3):
        return {
            "input_ids": torch.arange(tokens),
            "top_values": torch.zeros(tokens, top_k),
            "top_indices": torch.zeros(tokens, top_k, dtype=torch.int32),
            "teacher_logsumexp": torch.ones(tokens),
        }

    def test_accepts_complete_finite_rows(self) -> None:
        rows = [self.row(), self.row()]
        self.assertTrue(MODULE.validate_cache_rows(rows, expected_rows=2, top_k=3))

    def test_rejects_missing_shape_and_nonfinite_rows(self) -> None:
        self.assertFalse(MODULE.validate_cache_rows([self.row()], expected_rows=2, top_k=3))
        bad_shape = self.row()
        bad_shape["top_indices"] = torch.zeros(2, 3)
        self.assertFalse(MODULE.validate_cache_rows([bad_shape], expected_rows=1, top_k=3))
        nonfinite = self.row()
        nonfinite["top_values"][0, 0] = float("nan")
        self.assertFalse(MODULE.validate_cache_rows([nonfinite], expected_rows=1, top_k=3))

    def test_teacher_cache_texts_and_dimensions_are_sealed(self) -> None:
        prereg = json.loads((Path(__file__).resolve().parents[1] / "experiments/qwen38_27b_spot/qat_preregistration.json").read_text())
        MODULE.validate_preregistration(prereg, top_k=64, max_length=128)
        changed = json.loads(json.dumps(prereg))
        changed["first_rung"]["calibration_texts"][0] += " changed"
        with self.assertRaisesRegex(ValueError, "differ"):
            MODULE.validate_preregistration(changed, top_k=64, max_length=128)


if __name__ == "__main__":
    unittest.main()
