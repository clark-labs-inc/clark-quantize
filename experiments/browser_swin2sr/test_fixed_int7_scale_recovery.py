import unittest
from types import SimpleNamespace

import torch
from torch.nn.utils import parametrize

from fixed_int7_scale_recovery import (
    FixedInt7ScaleProjection,
    clamp_scale_deltas,
    configure_fixed_int7_scales,
    scale_receipt,
)
from retained_precision_islands import RSTB_FUSION_FP16, symmetric_int7_per_output


class Identity(torch.nn.Module):
    def forward(self, weight):
        return weight


class FixedInt7ScaleRecoveryTests(unittest.TestCase):
    def test_projection_starts_exactly_and_only_scales_receive_gradients(self):
        torch.manual_seed(17)
        weight = torch.randn(4, 5, 3, 3) * 0.1
        projection = FixedInt7ScaleProjection(weight, "mse_iterative")
        actual = projection(weight)
        self.assertTrue(torch.equal(
            actual,
            symmetric_int7_per_output(weight, "mse_iterative"),
        ))
        actual.sum().backward()
        self.assertIsNotNone(projection.log_scale_delta.grad)

    def test_four_layer_configuration_exposes_only_240_values(self):
        model = torch.nn.Module()
        surfaces = []
        for stage, (name, _values) in enumerate(RSTB_FUSION_FP16.modules):
            module = torch.nn.Conv2d(60, 60, 3, bias=False)
            model.add_module(f"stage_{stage}", module)
            parametrize.register_parametrization(module, "weight", Identity())
            surfaces.append(SimpleNamespace(name=name, module=module))
        fixed = configure_fixed_int7_scales(model, surfaces, 0b0111)
        trainable = [value for value in model.parameters() if value.requires_grad]
        self.assertEqual(sum(value.numel() for value in trainable), 240)
        for stage, surface in enumerate(fixed):
            expected = "mse_iterative" if stage < 3 else "max"
            self.assertTrue(torch.equal(
                surface.projection(surface.projection.codes.new_zeros(
                    (60, 60, 3, 3), dtype=torch.float32
                )),
                surfaces[stage].module.weight,
            ))
            self.assertEqual(
                expected == "mse_iterative",
                bool(0b0111 & (1 << stage)),
            )
        fixed[0].projection.log_scale_delta.data.fill_(1.0)
        clamp_scale_deltas(fixed)
        self.assertEqual(scale_receipt(fixed)["at_bound"], 60)


if __name__ == "__main__":
    unittest.main()
