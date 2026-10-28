from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

from clark_quantize import base3
from clark_quantize.base3 import frame_metadata, pack_base3, unpack_base3


class Base3PackingTests(unittest.TestCase):
    def test_exhaustive_short_sequences(self) -> None:
        for length in range(8):
            for values in itertools.product((-1, 0, 1), repeat=length):
                source = np.asarray(values, dtype=np.int8)
                np.testing.assert_array_equal(unpack_base3(pack_base3(source)), source)

    def test_deterministic_random_roundtrip_and_rate(self) -> None:
        rng = np.random.default_rng(20260814)
        source = rng.integers(-1, 2, size=100_000, dtype=np.int8)
        first = pack_base3(source)
        second = pack_base3(source.copy())
        self.assertEqual(first, second)
        np.testing.assert_array_equal(unpack_base3(first), source)
        metadata = frame_metadata(first)
        self.assertLessEqual(metadata["payload_bits"] / source.size, math.log2(3) + 0.001)

    def test_corruption_is_rejected(self) -> None:
        frame = bytearray(pack_base3(np.asarray([-1, 0, 1] * 100, dtype=np.int8)))
        frame[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            unpack_base3(bytes(frame))

    def test_invalid_code_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ternary"):
            pack_base3(np.asarray([0, 2], dtype=np.int8))

    @unittest.skipUnless(base3.native_decoder_available(), "native decoder is not built")
    def test_native_decoder_matches_python_exactly(self) -> None:
        generator = np.random.default_rng(20260816)
        for count in (0, 1, 2, 31, 128, 1025, 65537):
            codes = generator.integers(-1, 2, size=count, dtype=np.int8)
            payload, bit_length = base3.encode_payload(codes)
            native = base3.decode_payload(payload, bit_length, count)
            python = base3._decode_payload_python(payload, bit_length, count)
            self.assertTrue(np.array_equal(native, python))
            self.assertTrue(np.array_equal(native, codes))


if __name__ == "__main__":
    unittest.main()
