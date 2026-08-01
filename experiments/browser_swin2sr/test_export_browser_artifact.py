import numpy as np

from export_browser_artifact import pack_unsigned, packed_scale_codebook


def unpack_unsigned(payload: bytes, count: int, bit_width: int) -> list[int]:
    values = []
    for index in range(count):
        bit_offset = index * bit_width
        value = 0
        for bit in range(bit_width):
            value |= (
                (payload[(bit_offset + bit) // 8] >> ((bit_offset + bit) % 8)) & 1
            ) << bit
        values.append(value)
    return values


def test_unsigned_bitpacking_round_trips_unaligned_counts():
    for bit_width in (1, 2, 3, 6, 8):
        values = np.asarray(
            [index % (1 << bit_width) for index in range(19)],
            dtype=np.uint8,
        )
        payload = pack_unsigned(values, bit_width)
        assert len(payload) == (values.size * bit_width + 7) // 8
        assert unpack_unsigned(payload, values.size, bit_width) == values.tolist()


def test_scale_codebook_uses_fp16_centers_and_target_width():
    scales = np.asarray([0.125, 0.5, 0.125, 0.5, 0.5], dtype=np.float32)
    centers, payload = packed_scale_codebook(scales, 1)
    assert centers.dtype == np.float16
    assert centers.tolist() == [0.125, 0.5]
    assert unpack_unsigned(payload, scales.size, 1) == [0, 1, 0, 1, 1]


def test_scale_codebook_rejects_more_levels_than_target_width():
    scales = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    try:
        packed_scale_codebook(scales, 1)
    except ValueError as error:
        assert "exceeding q1" in str(error)
    else:
        raise AssertionError("q1 must reject three distinct stored scale levels")
