import unittest

from seal_surface_group_confirmation import select_candidates


def surface(name, ground_truth, edge, teacher, scale_bytes):
    return {
        "name": name,
        "category": "core_linear",
        "delta_vs_hard": {
            "psnr_vs_ground_truth": ground_truth,
            "edge_psnr_vs_ground_truth": edge,
            "psnr_vs_teacher": teacher,
        },
        "storage": {"scale_bytes_delta": scale_bytes},
    }


class SealSurfaceGroupConfirmationTests(unittest.TestCase):
    def test_selection_enforces_quality_and_byte_budget(self):
        selected, consumed = select_candidates([
            surface("efficient", 0.1, 0.1, 0.0, 1000),
            surface("larger", 0.15, 0.1, 0.0, 2000),
            surface("teacher-regression", 1.0, 1.0, -0.2, 100),
            surface("edge-regression", 1.0, -0.1, 0.0, 100),
        ], 2500)

        self.assertEqual([row["name"] for row in selected], ["efficient"])
        self.assertEqual(consumed, 1000)

    def test_selection_breaks_equal_scores_by_name(self):
        selected, _consumed = select_candidates([
            surface("zeta", 0.1, 0.1, 0.0, 1000),
            surface("alpha", 0.1, 0.1, 0.0, 1000),
        ], 2000)

        self.assertEqual(
            [row["name"] for row in selected],
            ["alpha", "zeta"],
        )


if __name__ == "__main__":
    unittest.main()
