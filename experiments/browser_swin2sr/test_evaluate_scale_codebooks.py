from evaluate_scale_codebooks import gates, select_bits


def _evidence(ground=-0.01, edge=-0.01, teacher=-0.01, source=-0.01):
    return {
        "paired": {
            "psnr_vs_ground_truth": {"lower_95": ground},
            "edge_psnr_vs_ground_truth": {"mean": edge},
            "psnr_vs_teacher": {"lower_95": teacher},
        },
        "ground_truth_by_source": {"a": source, "b": source},
    }


def _rule():
    return {
        "ground_truth_lower_95_min_db": -0.02,
        "edge_mean_min_db": -0.02,
        "teacher_lower_95_min_db": -0.03,
        "every_source_ground_truth_mean_min_db": -0.03,
        "minimum_scale_storage_reduction_fraction": 0.5,
    }


def test_gate_requires_quality_and_material_storage_reduction():
    baseline = {"scale_storage_bytes": 100}
    assert all(gates(_evidence(), {"scale_storage_bytes": 40}, baseline, _rule()).values())
    failed = gates(_evidence(ground=-0.021), {"scale_storage_bytes": 40}, baseline, _rule())
    assert not failed["ground_truth_noninferior"]
    failed = gates(_evidence(), {"scale_storage_bytes": 60}, baseline, _rule())
    assert not failed["scale_storage_reduction"]


def test_selection_prefers_smallest_passing_storage_not_quality_mean():
    screen = {
        1: {"bits": 1, "storage": {"packed_trunk_bits_per_quantized_weight": 1.64}, "gates": {"x": False}},
        2: {"bits": 2, "storage": {"packed_trunk_bits_per_quantized_weight": 1.69}, "gates": {"x": True}},
        3: {"bits": 3, "storage": {"packed_trunk_bits_per_quantized_weight": 1.74}, "gates": {"x": True}},
    }
    assert select_bits(screen) == 2
    for row in screen.values():
        row["gates"] = {"x": False}
    assert select_bits(screen) == 16
