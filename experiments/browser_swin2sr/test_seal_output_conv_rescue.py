import unittest

from seal_output_conv_rescue import verify_selected_parent


def decision(selected: str = "selected") -> dict:
    return {
        "format": "backdropkit-swin2sr-sensitive-io-development-decision-v1",
        "sensitive_recipe_selected": True,
        "selected_checkpoint_sha256": selected,
        "model_promotion_authorized": False,
        "final_heldout_locked": True,
    }


def receipt(
    *,
    group: int = 3,
    parent: str = "selected",
    surface: str = "upsample.conv",
    step: int = 19000,
) -> dict:
    return {
        "parent_checkpoint_sha256": parent,
        "parent_surface_step": step,
        "surface_group_sizes": {
            "swin2sr.first_convolution": 2,
            "upsample.conv": group,
        },
        "change": {
            "surface": surface,
            "previous_group_size": 9,
            "group_size": group,
            "convolution_learning_rate": None,
        },
    }


class OutputConvRescueSealTests(unittest.TestCase):
    def test_accepts_exact_selected_parent_and_group(self):
        group, non_output = verify_selected_parent(
            receipt(),
            decision(),
            arm="output-g3",
        )

        self.assertEqual(group, 3)
        self.assertEqual(
            non_output,
            {"swin2sr.first_convolution": 2},
        )

    def test_rejects_unselected_parent(self):
        with self.assertRaisesRegex(RuntimeError, "selected input"):
            verify_selected_parent(
                receipt(parent="other"),
                decision(),
                arm="output-g3",
            )

    def test_rejects_wrong_surface_or_group(self):
        with self.assertRaisesRegex(RuntimeError, "output convolution"):
            verify_selected_parent(
                receipt(surface="swin2sr.first_convolution"),
                decision(),
                arm="output-g3",
            )
        with self.assertRaisesRegex(RuntimeError, "match arm"):
            verify_selected_parent(
                receipt(group=2),
                decision(),
                arm="output-g3",
            )

    def test_rejects_wrong_parent_step(self):
        with self.assertRaisesRegex(RuntimeError, "step 19000"):
            verify_selected_parent(
                receipt(step=18000),
                decision(),
                arm="output-g3",
            )


if __name__ == "__main__":
    unittest.main()
