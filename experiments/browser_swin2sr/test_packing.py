"""Round-trip tests for the browser Swin2SR packing format."""

from __future__ import annotations

import unittest

import numpy as np

from clark_air.quant.packing import (
    pack_trits_20,
    pack_trits_40,
    unpack_trits_20,
    unpack_trits_40,
)


class BrowserPackingTest(unittest.TestCase):
    def test_pack20_round_trip_across_word_boundary(self) -> None:
        codes = np.resize(np.array([-1, 0, 1], dtype=np.int8), (7, 13))
        packed, shape, pad = pack_trits_20(codes)

        self.assertEqual(packed.dtype, np.uint32)
        np.testing.assert_array_equal(unpack_trits_20(packed, shape, pad), codes)

    def test_pack20_matches_pack40_decoding(self) -> None:
        random = np.random.default_rng(42)
        codes = random.integers(-1, 2, size=(61, 60), dtype=np.int8)
        packed20, shape20, pad20 = pack_trits_20(codes)
        packed40, shape40, pad40 = pack_trits_40(codes)

        np.testing.assert_array_equal(
            unpack_trits_20(packed20, shape20, pad20),
            unpack_trits_40(packed40, shape40, pad40),
        )

    def test_pack20_rejects_non_ternary_codes(self) -> None:
        with self.assertRaises(ValueError):
            pack_trits_20(np.array([-1, 0, 2], dtype=np.int8))


if __name__ == "__main__":
    unittest.main()
