import unittest

from seal_broad_surface_group_confirmation import (
    choose_with_budget,
    screen_options,
)


def row(name, group, gt, edge, teacher, cost):
    return {
        "name": name,
        "category": "core_linear",
        "delta_vs_hard": {
            "psnr_vs_ground_truth": gt,
            "edge_psnr_vs_ground_truth": edge,
            "psnr_vs_teacher": teacher,
        },
        "storage": {"scale_bytes_delta": cost},
        "candidate_group_size": group,
    }


class BroadSurfaceGroupConfirmationTests(unittest.TestCase):
    def test_screen_picks_one_efficient_option_per_surface(self):
        reports = {
            3: {"surfaces": [row("a", 3, 0.2, 0.1, 0.0, 2000)]},
            6: {"surfaces": [
                row("a", 6, 0.15, 0.1, 0.0, 500),
                row("existing", 6, 1.0, 1.0, 1.0, 100),
            ]},
        }

        options = screen_options(reports, {"existing"})

        self.assertEqual(len(options), 1)
        self.assertEqual(options[0]["group_size"], 6)

    def test_budget_includes_existing_map_cost(self):
        selected, consumed = choose_with_budget([
            {"name": "a", "scale_bytes_delta": 1000},
            {"name": "b", "scale_bytes_delta": 2000},
        ], 4000, 5500)

        self.assertEqual([row["name"] for row in selected], ["a"])
        self.assertEqual(consumed, 5000)


if __name__ == "__main__":
    unittest.main()
