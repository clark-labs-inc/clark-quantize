import torch

from mixed_scale_codebook import mixed_scale_quantized_dense_state


def fixture():
    runtime = {
        "core.weight": torch.tensor([[-1.0, 0.0], [2.0, 0.0]]),
        "io.weight": torch.tensor([[-0.25, 0.0], [1.0, 0.0], [4.0, 0.0]]),
    }
    materialization = {
        "surfaces": {
            "core.weight": {
                "category": "core_linear",
                "group_size": 2,
                "rotation": "none",
                "rotation_block": 4,
            },
            "io.weight": {
                "category": "image_io_conv",
                "group_size": 2,
                "rotation": "none",
                "rotation_block": 4,
            },
        }
    }
    return materialization, runtime, {name: value.clone() for name, value in runtime.items()}


def test_sensitive_category_bits_reduce_weight_error_and_report_storage():
    materialization, runtime, dense = fixture()
    _uniform, q1 = mixed_scale_quantized_dense_state(
        materialization,
        runtime,
        dense,
        {"core_linear": 1, "image_io_conv": 1},
    )
    candidate, mixed = mixed_scale_quantized_dense_state(
        materialization,
        runtime,
        dense,
        {"core_linear": 1, "image_io_conv": 2},
    )
    assert mixed["relative_weight_rmse"] < q1["relative_weight_rmse"]
    assert torch.equal(candidate["io.weight"], dense["io.weight"])
    assert mixed["by_category"]["image_io_conv"]["requested_bits"] == 2
    assert mixed["packed_trunk_bits_per_quantized_weight"] > 0


def test_mixed_map_requires_every_category():
    materialization, runtime, dense = fixture()
    try:
        mixed_scale_quantized_dense_state(
            materialization, runtime, dense, {"core_linear": 1}
        )
    except ValueError as error:
        assert "every surface category" in str(error)
    else:
        raise AssertionError("incomplete mixed scale map must be rejected")
