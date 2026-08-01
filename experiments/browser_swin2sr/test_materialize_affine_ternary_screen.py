import unittest
from types import SimpleNamespace

from materialize_affine_ternary_screen import is_eligible


def surface(category: str, group_size: int):
    return SimpleNamespace(
        category=category,
        quantizer=SimpleNamespace(group_size=group_size),
    )


class AffineScreenMaterializationTests(unittest.TestCase):
    def test_only_group60_linear_categories_are_eligible(self):
        self.assertTrue(is_eligible(surface("core_linear", 60)))
        self.assertTrue(is_eligible(surface("position_bias_linear", 60)))
        self.assertFalse(is_eligible(surface("core_linear", 3)))
        self.assertFalse(is_eligible(surface("position_bias_linear", 2)))
        self.assertFalse(is_eligible(surface("residual_body_conv", 60)))
        self.assertFalse(is_eligible(surface("image_io_conv", 60)))
