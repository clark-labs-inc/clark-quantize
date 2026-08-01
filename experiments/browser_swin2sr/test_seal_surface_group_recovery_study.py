import unittest

from seal_surface_group_recovery_study import selected_parent


class SurfaceGroupRecoveryStudyTests(unittest.TestCase):
    def test_optimizer_parent_has_priority_when_selected(self):
        parent = selected_parent(
            {
                "optimizer_recipe_selected": True,
                "selected_checkpoint": "adamw.pt",
                "selected_checkpoint_sha256": "adamw-sha",
                "selected_optimizer": "adamw",
            },
            {
                "selected_pixel_objective": "charbonnier",
                "receipts": {
                    "charbonnier": {
                        "checkpoint": "objective.pt",
                        "checkpoint_sha256": "objective-sha",
                    }
                },
            },
        )

        self.assertEqual(parent["checkpoint"], "adamw.pt")
        self.assertEqual(parent["selection"], "optimizer_transfer")

    def test_objective_parent_is_fallback(self):
        parent = selected_parent(
            {"optimizer_recipe_selected": False},
            {
                "selected_pixel_objective": "psnr_aligned",
                "receipts": {
                    "psnr_aligned": {
                        "checkpoint": "psnr.pt",
                        "checkpoint_sha256": "psnr-sha",
                    }
                },
            },
        )

        self.assertEqual(parent["checkpoint"], "psnr.pt")
        self.assertEqual(parent["selection"], "pixel_objective")


if __name__ == "__main__":
    unittest.main()
