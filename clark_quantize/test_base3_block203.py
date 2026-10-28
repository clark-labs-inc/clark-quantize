from __future__ import annotations

import hashlib
import unittest

import numpy as np

from clark_quantize.base3_block203 import (
    GROUP_BITS,
    GROUP_TRITS,
    bits_for_trits,
    metadata,
    matrix_payload_bits,
    matrix_payload_bytes,
    pack_row,
    pack_rows,
    row_payload_bits,
    row_payload_bytes,
    unpack_row,
    unpack_rows,
)


class Base3Block203Tests(unittest.TestCase):
    def test_group_width_is_minimal(self) -> None:
        self.assertLess(3**GROUP_TRITS, 2**GROUP_BITS)
        self.assertGreaterEqual(3**GROUP_TRITS, 2 ** (GROUP_BITS - 1))
        self.assertEqual(bits_for_trits(128), 203)

    def test_adversarial_and_tail_sizes_round_trip(self) -> None:
        generator = np.random.default_rng(20260817)
        for count in (0, 1, 2, 5, 127, 128, 129, 255, 256, 4097, 5120):
            cases = [
                np.full(count, -1, dtype=np.int8),
                np.zeros(count, dtype=np.int8),
                np.ones(count, dtype=np.int8),
                generator.integers(-1, 2, size=count, dtype=np.int8),
            ]
            for codes in cases:
                payload = pack_row(codes)
                self.assertEqual(len(payload), row_payload_bytes(count))
                np.testing.assert_array_equal(unpack_row(payload, count), codes)

    def test_encoding_is_deterministic(self) -> None:
        codes = (np.arange(5120, dtype=np.int64) % 3 - 1).astype(np.int8)
        first = pack_row(codes)
        self.assertEqual(first, pack_row(codes.copy()))
        self.assertEqual(
            hashlib.sha256(first).hexdigest(),
            "eb84f163a53f78a948d735a9ed36d80af535acf0665d651669bbd703f784c579",
        )

    def test_noncanonical_codeword_is_rejected(self) -> None:
        value = 3**128
        payload = value.to_bytes(row_payload_bytes(128), "little")
        with self.assertRaisesRegex(ValueError, "non-canonical"):
            unpack_row(payload, 128)

    def test_nonzero_padding_bits_are_rejected(self) -> None:
        payload = bytearray(pack_row(np.zeros(1, dtype=np.int8)))
        payload[-1] |= 0x80
        with self.assertRaisesRegex(ValueError, "padding"):
            unpack_row(payload, 1)

    def test_invalid_codes_and_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pack_row(np.asarray([-2], dtype=np.int8))
        with self.assertRaises(ValueError):
            pack_row(np.asarray([255], dtype=np.uint8))
        with self.assertRaises(ValueError):
            pack_row(np.asarray([0.5], dtype=np.float32))
        with self.assertRaises(ValueError):
            unpack_row(b"", 1)
        with self.assertRaises(ValueError):
            row_payload_bits(-1)

    def test_metadata_counts_exact_payload(self) -> None:
        result = metadata(5120)
        self.assertEqual(result["groups"], 40)
        self.assertEqual(result["payload_bits"], 8120)
        self.assertEqual(result["payload_bytes"], 1015)
        self.assertEqual(result["payload_bits_per_trit"], 203 / 128)

    def test_matrix_rows_reset_groups_without_interrow_byte_padding(self) -> None:
        generator = np.random.default_rng(20260817)
        for rows, columns in ((3, 1), (5, 2), (3, 5), (2, 127), (3, 128),
                              (4, 129), (2, 255), (3, 256), (2, 4097)):
            codes = generator.integers(-1, 2, size=(rows, columns), dtype=np.int8)
            payload = pack_rows(codes)
            self.assertEqual(len(payload), matrix_payload_bytes(rows, columns))
            self.assertEqual(
                matrix_payload_bits(rows, columns), rows * row_payload_bits(columns),
            )
            np.testing.assert_array_equal(unpack_rows(payload, rows, columns), codes)
        self.assertEqual(len(pack_rows(np.zeros((3, 1), dtype=np.int8))), 1)

    def test_matrix_padding_and_dimensions_are_canonical(self) -> None:
        payload = bytearray(pack_rows(np.zeros((3, 1), dtype=np.int8)))
        payload[0] |= 0x80
        with self.assertRaisesRegex(ValueError, "padding"):
            unpack_rows(payload, 3, 1)
        with self.assertRaises(ValueError):
            pack_rows(np.zeros(3, dtype=np.int8))
        with self.assertRaises(ValueError):
            matrix_payload_bits(1, 0)


if __name__ == "__main__":
    unittest.main()
