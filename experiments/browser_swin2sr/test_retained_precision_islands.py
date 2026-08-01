import unittest
from types import SimpleNamespace

import torch
from torch.nn.utils import parametrize

from retained_precision_islands import (
    IMAGE_IO_FP16,
    RSTB_FUSION_FP16,
    boost_int7_boundary_gradients,
    configure_fp16_island_for_loading,
    configure_int7_island_for_loading,
    fp16_roundtrip,
    initialize_trainable_fp16_island,
    initialize_trainable_int7_island,
    initialize_trainable_ternary_island,
    materialize_fp16_island,
    packed_trit_bytes,
    retained_precision_configs,
    symmetric_int7_per_output,
    symmetric_int7_per_output_components,
)


class Identity(torch.nn.Module):
    def forward(self, weight):
        return weight


def surface(name, category, module, group_size):
    parametrize.register_parametrization(module, "weight", Identity())
    values = torch.linspace(
        -0.113,
        0.127,
        module.weight.numel(),
        dtype=torch.float32,
    ).reshape(module.weight.shape)
    module.parametrizations.weight.original.data.copy_(values)
    return SimpleNamespace(
        name=name,
        category=category,
        module=module,
        quantizer=SimpleNamespace(
            group_size=group_size,
            rotation="none",
            amount=torch.tensor(0.0),
        ),
    )


class RetainedPrecisionIslandTests(unittest.TestCase):
    def test_specs_are_exact_and_disjoint(self):
        self.assertEqual(IMAGE_IO_FP16.weight_values, 8_100)
        self.assertEqual(RSTB_FUSION_FP16.weight_values, 129_600)
        self.assertTrue(
            set(name for name, _ in IMAGE_IO_FP16.modules).isdisjoint(
                name for name, _ in RSTB_FUSION_FP16.modules
            )
        )

    def test_packed_trit_storage_uses_twenty_values_per_u32(self):
        self.assertEqual(packed_trit_bytes(0), 0)
        self.assertEqual(packed_trit_bytes(20), 4)
        self.assertEqual(packed_trit_bytes(21), 8)

    def test_int7_boundary_boost_targets_code_transitions(self):
        parameter = torch.nn.Parameter(torch.zeros(1, 1, 3, 3))
        parameter.data[0, 0, 0, 0] = 63.0
        parameter.data[0, 0, 0, 1] = 10.5
        parameter.data[0, 0, 0, 2] = 10.0
        parameter.grad = torch.ones_like(parameter)

        receipt = boost_int7_boundary_gradients(
            [parameter],
            boost=3.0,
            temperature=0.1,
        )

        self.assertEqual(receipt["parameters"], 9)
        self.assertGreater(
            parameter.grad[0, 0, 0, 1],
            parameter.grad[0, 0, 0, 2],
        )
        self.assertLessEqual(receipt["max_gain"], 4.0)

    def test_iterative_int7_scale_never_worsens_weight_mse(self):
        torch.manual_seed(7)
        weight = torch.randn(4, 6, 3, 3) * 0.13
        weight[0, 0, 0, 0] = 1.0

        maximum = symmetric_int7_per_output(weight, "max")
        iterative = symmetric_int7_per_output(weight, "mse_iterative")
        blended_zero = symmetric_int7_per_output(
            weight,
            "mse_iterative",
            0.0,
        )
        blended_full = symmetric_int7_per_output(
            weight,
            "mse_iterative",
            1.0,
        )

        self.assertTrue(torch.equal(blended_zero, maximum))
        self.assertTrue(torch.equal(blended_full, iterative))
        self.assertLessEqual(
            (weight - iterative).square().sum().item(),
            (weight - maximum).square().sum().item() + 1e-8,
        )
        rows = iterative.reshape(iterative.shape[0], -1)
        scales = rows.abs().amax(dim=1) / 63
        recovered_codes = torch.round(rows / scales[:, None])
        self.assertTrue(torch.equal(rows, recovered_codes * scales[:, None]))

    def test_int7_components_reconstruct_the_deployment_tensor(self):
        torch.manual_seed(11)
        weight = torch.randn(3, 4, 3, 3) * 0.2
        codes, scales = symmetric_int7_per_output_components(
            weight,
            "mse_iterative",
        )
        reconstructed = (
            codes.to(weight.dtype) * scales[:, None]
        ).reshape_as(weight)
        self.assertEqual(codes.dtype, torch.int8)
        self.assertTrue(torch.equal(
            reconstructed,
            symmetric_int7_per_output(weight, "mse_iterative"),
        ))

    def test_materialization_changes_only_declared_fp16_weights(self):
        surfaces = [
            surface(
                "swin2sr.first_convolution",
                "image_io_conv",
                torch.nn.Conv2d(3, 60, 3, bias=False),
                2,
            ),
            surface(
                "upsample.conv",
                "image_io_conv",
                torch.nn.Conv2d(60, 12, 3, bias=False),
                2,
            ),
        ]
        baseline = {
            f"{item.name}.weight": torch.zeros_like(item.module.weight)
            for item in surfaces
        }
        baseline["unrelated"] = torch.tensor([3.25])
        state, receipt = materialize_fp16_island(
            baseline,
            surfaces,
            IMAGE_IO_FP16,
        )
        self.assertTrue(torch.equal(state["unrelated"], baseline["unrelated"]))
        for item in surfaces:
            expected = fp16_roundtrip(
                item.module.parametrizations.weight.original
            )
            self.assertTrue(torch.equal(state[f"{item.name}.weight"], expected))
        self.assertEqual(receipt["current_ternary_code_bytes"], 1_620)
        self.assertEqual(receipt["current_fp16_scale_bytes"], 8_100)
        self.assertEqual(receipt["retained_fp16_weight_bytes"], 16_200)
        self.assertEqual(receipt["projected_data_byte_delta"], 6_480)
        self.assertEqual(receipt["retained_int7_data_bytes"], 7_232)
        self.assertEqual(receipt["projected_int7_data_byte_delta"], -2_488)

    def test_trainable_island_starts_from_hard_fp16_not_latent_master(self):
        first = torch.nn.Conv2d(3, 60, 3, bias=False)
        output = torch.nn.Conv2d(60, 12, 3, bias=False)
        model = torch.nn.Module()
        model.add_module("first", first)
        model.add_module("output", output)
        surfaces = [
            surface("swin2sr.first_convolution", "image_io_conv", first, 2),
            surface("upsample.conv", "image_io_conv", output, 2),
        ]
        expected = {
            item.name: item.module.weight.detach().to(torch.float16).to(torch.float32)
            for item in surfaces
        }
        receipt = initialize_trainable_fp16_island(
            model,
            surfaces,
            IMAGE_IO_FP16,
        )
        self.assertEqual(receipt["trainable_values"], 8_100)
        self.assertEqual(len(receipt["parameters"]), 2)
        for item in surfaces:
            self.assertTrue(torch.equal(item.module.weight, expected[item.name]))
            self.assertTrue(
                item.module.parametrizations.weight.original.requires_grad
            )

    def test_ternary_control_starts_from_same_hard_weight(self):
        first = torch.nn.Conv2d(3, 60, 3, bias=False)
        output = torch.nn.Conv2d(60, 12, 3, bias=False)
        model = torch.nn.Module()
        model.add_module("first", first)
        model.add_module("output", output)
        surfaces = [
            surface("swin2sr.first_convolution", "image_io_conv", first, 2),
            surface("upsample.conv", "image_io_conv", output, 2),
        ]
        expected = {item.name: item.module.weight.detach().clone() for item in surfaces}
        receipt = initialize_trainable_ternary_island(
            model,
            surfaces,
            IMAGE_IO_FP16,
        )
        self.assertEqual(receipt["precision"], "ternary")
        self.assertEqual(receipt["trainable_values"], 8_100)
        for item in surfaces:
            self.assertTrue(torch.equal(item.module.weight, expected[item.name]))

    def test_int7_island_starts_at_exact_per_output_boundary(self):
        modules = [
            torch.nn.Conv2d(60, 60, 3, bias=False)
            for _stage in range(4)
        ]
        model = torch.nn.Module()
        surfaces = []
        for stage, module in enumerate(modules):
            model.add_module(f"stage_{stage}", module)
            surfaces.append(surface(
                f"swin2sr.encoder.stages.{stage}.conv",
                "residual_body_conv",
                module,
                9,
            ))
        parents = {
            item.name: item.module.weight.detach().clone()
            for item in surfaces
        }

        receipt = initialize_trainable_int7_island(
            model,
            surfaces,
            RSTB_FUSION_FP16,
        )

        self.assertEqual(receipt["precision"], "int7")
        self.assertEqual(receipt["trainable_values"], 129_600)
        for item in surfaces:
            expected = symmetric_int7_per_output(parents[item.name])
            self.assertTrue(torch.equal(item.module.weight, expected))
            item.module.weight.sum().backward()
            self.assertIsNotNone(
                item.module.parametrizations.weight.original.grad
            )

    def test_fp16_island_can_be_recreated_before_checkpoint_load(self):
        def fixture():
            first = torch.nn.Conv2d(3, 60, 3, bias=False)
            output = torch.nn.Conv2d(60, 12, 3, bias=False)
            model = torch.nn.Module()
            model.add_module("first", first)
            model.add_module("output", output)
            return model, [
                surface("swin2sr.first_convolution", "image_io_conv", first, 2),
                surface("upsample.conv", "image_io_conv", output, 2),
            ]

        source_model, source_surfaces = fixture()
        initialize_trainable_fp16_island(
            source_model,
            source_surfaces,
            IMAGE_IO_FP16,
        )
        source_state = source_model.state_dict()
        target_model, target_surfaces = fixture()
        receipt = configure_fp16_island_for_loading(
            target_surfaces,
            IMAGE_IO_FP16,
        )
        target_model.load_state_dict(source_state, strict=True)
        self.assertEqual(receipt["weight_values"], 8_100)
        for source_item, target_item in zip(source_surfaces, target_surfaces):
            torch.testing.assert_close(
                target_item.module.weight,
                source_item.module.weight,
            )

    def test_int7_island_can_be_recreated_before_checkpoint_load(self):
        def fixture():
            modules = [
                torch.nn.Conv2d(60, 60, 3, bias=False)
                for _stage in range(4)
            ]
            model = torch.nn.Module()
            surfaces = []
            for stage, module in enumerate(modules):
                model.add_module(f"stage_{stage}", module)
                surfaces.append(surface(
                    f"swin2sr.encoder.stages.{stage}.conv",
                    "residual_body_conv",
                    module,
                    9,
                ))
            return model, surfaces

        source_model, source_surfaces = fixture()
        initialize_trainable_int7_island(
            source_model,
            source_surfaces,
            RSTB_FUSION_FP16,
        )
        source_state = source_model.state_dict()
        target_model, target_surfaces = fixture()
        receipt = configure_int7_island_for_loading(
            target_surfaces,
            RSTB_FUSION_FP16,
        )
        target_model.load_state_dict(source_state, strict=True)
        self.assertEqual(receipt["weight_values"], 129_600)
        for source_item, target_item in zip(source_surfaces, target_surfaces):
            torch.testing.assert_close(
                target_item.module.weight,
                source_item.module.weight,
            )

    def test_precision_history_keeps_one_effective_config_per_island(self):
        image = {"island": "image_io_fp16", "precision": "fp16", "seed": 1}
        earlier = {"island": "rstb_fusion_fp16", "precision": "ternary", "seed": 2}
        current = {"island": "rstb_fusion_fp16", "precision": "fp16", "seed": 3}
        payload = {
            "retained_precision_history": [image, earlier],
            "retained_fp16_training": current,
        }
        self.assertEqual(retained_precision_configs(payload), [image, current])

    def test_precision_history_accepts_int7(self):
        int7 = {
            "island": "rstb_fusion_fp16",
            "precision": "int7",
            "seed": 4,
        }
        self.assertEqual(
            retained_precision_configs({"retained_fp16_training": int7}),
            [int7],
        )


if __name__ == "__main__":
    unittest.main()
