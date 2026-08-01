from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from rstb_bias_recovery import (
    BIAS_NAMES,
    bias_delta_receipt,
    clamp_bias_deltas,
    load_bias_student,
)
from surface_model_state import pretrained


class RstbBiasRecoveryTests(unittest.TestCase):
    def test_load_freezes_everything_except_four_biases(self) -> None:
        with TemporaryDirectory() as root:
            state = Path(root) / "state.pt"
            torch.save(pretrained("cpu").state_dict(), state)
            student, parameters, origins = load_bias_student(state, "cpu")
        trainable = {name for name, value in student.named_parameters() if value.requires_grad}
        self.assertEqual(trainable, set(BIAS_NAMES))
        self.assertEqual(sum(value.numel() for value in parameters), 240)
        self.assertEqual(set(origins), set(BIAS_NAMES))

    def test_clamp_and_receipt_cover_deployment_roundtrip(self) -> None:
        with TemporaryDirectory() as root:
            state = Path(root) / "state.pt"
            torch.save(pretrained("cpu").state_dict(), state)
            student, parameters, origins = load_bias_student(state, "cpu")
        with torch.no_grad():
            parameters[0].add_(0.2)
        clamp_bias_deltas(student, origins)
        receipt = bias_delta_receipt(student, origins)
        self.assertEqual(receipt["parameters"], 240)
        self.assertLessEqual(receipt["max_abs_delta"], 0.050001)
        self.assertGreaterEqual(receipt["fp16_roundtrip_max_error"], 0)


if __name__ == "__main__":
    unittest.main()
