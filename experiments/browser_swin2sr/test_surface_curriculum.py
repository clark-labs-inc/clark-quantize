import unittest
from types import SimpleNamespace

import torch

from surface_curriculum import (
    enter_hard_shadow,
    hard_shadow_plan,
    restore_live_amounts,
)


def surface(amount: float):
    quantizer = SimpleNamespace(amount=torch.tensor(amount))
    return SimpleNamespace(quantizer=quantizer)


class SurfaceCurriculumTests(unittest.TestCase):
    def test_shadow_hardens_started_surfaces_and_preserves_future_surfaces(self):
        surfaces = [surface(1.0), surface(0.4), surface(0.0)]

        live = enter_hard_shadow(surfaces)

        self.assertEqual(live[0], 1.0)
        self.assertAlmostEqual(live[1], 0.4)
        self.assertEqual(live[2], 0.0)
        self.assertEqual(
            [item.quantizer.amount.item() for item in surfaces],
            [1.0, 1.0, 0.0],
        )
        restore_live_amounts(surfaces, live)
        self.assertAlmostEqual(surfaces[1].quantizer.amount.item(), 0.4)

    def test_shadow_weight_tracks_active_transition(self):
        plan = hard_shadow_plan(
            [surface(1.0), surface(0.5), surface(0.0)],
            weight_start=0.05,
            weight_end=0.25,
        )

        self.assertAlmostEqual(plan.weight, 0.15)
        self.assertEqual(plan.active_surfaces, 1)
        self.assertEqual(plan.hard_surfaces, 2)

    def test_shadow_is_disabled_outside_a_transition(self):
        plan = hard_shadow_plan(
            [surface(1.0), surface(0.0)],
            weight_start=0.05,
            weight_end=0.25,
        )

        self.assertEqual(plan.weight, 0.0)
        self.assertEqual(plan.active_surfaces, 0)

    def test_invalid_weight_range_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "0 <= start <= end"):
            hard_shadow_plan(
                [surface(0.5)],
                weight_start=0.25,
                weight_end=0.05,
            )


if __name__ == "__main__":
    unittest.main()
