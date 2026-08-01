import numpy as np
import pytest

from scale_codebook import (
    aligned_bytes,
    fp16_scale_dense_state,
    group_nonzero_counts,
    quantize_scale_codebook,
)


def test_codebook_is_deterministic_and_uses_fp16_centers():
    scales = np.asarray([0.1, 0.11, 0.5, 0.52, 1.0, 1.1], dtype=np.float32)
    weights = np.asarray([4, 3, 2, 5, 1, 4], dtype=np.float64)
    first = quantize_scale_codebook(scales, weights, 2)
    second = quantize_scale_codebook(scales, weights, 2)
    assert first.centers.dtype == np.float16
    assert np.array_equal(first.centers, second.centers)
    assert np.array_equal(first.indices, second.indices)
    assert np.array_equal(first.decoded, first.centers.astype(np.float32)[first.indices])
    assert first.index_bytes == 2


def test_exact_values_collapse_without_wasting_index_bits():
    scales = np.asarray([0.25, 0.25, 0.25], dtype=np.float32)
    result = quantize_scale_codebook(scales, np.ones(3), 4)
    assert result.bit_width == 0
    assert result.index_bytes == 0
    assert result.codebook_bytes == 2
    assert np.all(result.decoded == np.float32(0.25))


def test_group_counts_and_alignment_are_explicit():
    assert group_nonzero_counts(np.asarray([1, 0, -1, 0, 1]), 2).tolist() == [1, 1, 1]
    assert aligned_bytes([6, 3]) == (11, 2)
    with pytest.raises(ValueError):
        aligned_bytes([-1])


def test_rejects_invalid_scale_inputs():
    with pytest.raises(ValueError):
        quantize_scale_codebook(np.asarray([-1.0]), np.ones(1), 2)
    with pytest.raises(ValueError):
        quantize_scale_codebook(np.ones(2), np.ones(1), 2)
    with pytest.raises(ValueError):
        quantize_scale_codebook(np.ones(2), np.ones(2), 0)


def test_fp16_baseline_rebuilds_a_plain_unrotated_surface():
    state = {"layer.weight": __import__("torch").tensor([[0.5, 0.0, -0.5]])}
    materialization = {
        "surfaces": {
            "layer.weight": {
                "group_size": 3,
                "rotation": "none",
                "rotation_block": 4,
            }
        }
    }
    rebuilt, receipt = fp16_scale_dense_state(materialization, state, state)
    assert np.array_equal(rebuilt["layer.weight"].numpy(), state["layer.weight"].numpy())
    assert receipt["scale_storage_bytes"] == 2
    assert receipt["quantized_values"] == 3
