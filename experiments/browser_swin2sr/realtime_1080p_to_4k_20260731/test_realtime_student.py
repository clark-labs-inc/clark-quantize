from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from realtime_student import (
    RealtimeLumaStudent,
    RealtimeStudentConfig,
    export_wgsl_weights,
    initialize_rgb_from_luma,
    initialize_wider_rgb_from_rgb,
    wgsl_weight_arrays,
)


class RealtimeStudentTest(unittest.TestCase):
    def test_wider_rgb_initialization_preserves_the_selected_student(self) -> None:
        torch.manual_seed(11)
        narrow = RealtimeLumaStudent(RealtimeStudentConfig(
            channels=12,
            blocks=3,
            color_mode="rgb",
        ))
        wider = RealtimeLumaStudent(RealtimeStudentConfig(
            channels=14,
            blocks=3,
            color_mode="rgb",
        ))
        initialize_wider_rgb_from_rgb(wider, narrow.state_dict())
        source = torch.rand(2, 3, 64, 96)
        self.assertEqual(wider.parameter_count(), 11_424)
        self.assertLess(float((wider(source) - narrow(source)).abs().max()), 2e-6)
        extra_input = wider.input_projection.weight[12:]
        extra_output = wider.output_projection.weight[:, 12:]
        self.assertGreater(float(extra_input.abs().sum()), 0)
        self.assertEqual(float(extra_output.abs().sum()), 0)

    def test_rgb_initialization_exactly_reproduces_luma_student(self) -> None:
        torch.manual_seed(7)
        luma = RealtimeLumaStudent(RealtimeStudentConfig(channels=12, blocks=3))
        rgb = RealtimeLumaStudent(RealtimeStudentConfig(
            channels=12,
            blocks=3,
            color_mode="rgb",
        ))
        initialize_rgb_from_luma(rgb, luma.state_dict())
        source = torch.rand(2, 3, 64, 96)
        self.assertEqual(rgb.parameter_count(), 8496)
        self.assertLess(float((rgb(source) - luma(source)).abs().max()), 2e-6)

    def test_shape_parameter_count_and_baseline(self) -> None:
        model = RealtimeLumaStudent(RealtimeStudentConfig(channels=12, blocks=3))
        for parameter in model.parameters():
            parameter.data.zero_()
        source = torch.rand(2, 3, 64, 96)
        output = model(source)
        expected = torch.nn.functional.interpolate(
            source, scale_factor=2, mode="bilinear", align_corners=False
        )
        self.assertEqual(output.shape, (2, 3, 128, 192))
        self.assertEqual(model.parameter_count(), 8_016)
        torch.testing.assert_close(output, expected)

    def test_runtime_layout_and_payload_size(self) -> None:
        model = RealtimeLumaStudent(RealtimeStudentConfig(channels=12, blocks=3))
        arrays = wgsl_weight_arrays(model)
        self.assertEqual(arrays["input_projection"].shape, (12, 4))
        self.assertEqual(arrays["block_0"].shape, (2 * 9 * 12 * 12,))
        self.assertEqual(arrays["output_projection"].shape, (16, 12))
        self.assertTrue(all(values.dtype == np.float16 for values in arrays.values()))
        with TemporaryDirectory() as root:
            output = Path(root) / "model.bin"
            ledger = export_wgsl_weights(model, output)
            self.assertEqual(ledger["parameters"], 8_016)
            self.assertEqual(ledger["bytes"], 16_032)
            self.assertEqual(output.stat().st_size, 16_032)

    def test_rejects_odd_input_dimensions(self) -> None:
        model = RealtimeLumaStudent()
        with self.assertRaisesRegex(ValueError, "must be even"):
            model(torch.rand(1, 3, 63, 64))


if __name__ == "__main__":
    unittest.main()
