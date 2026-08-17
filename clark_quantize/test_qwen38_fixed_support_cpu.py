from __future__ import annotations

import ctypes
from dataclasses import replace
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from clark_quantize.qwen38_fixed_support_codec import (
    FixedSupportPayload,
    fixed_support_state_count,
    pack_fixed_support_matrix,
)
from clark_quantize.qwen38_fixed_support_cpu import (
    FixedSupportCPULibrary,
    load_fixed_support_cpu_library,
    native_pack_fixed_support_matrix,
    native_unpack_fixed_support_matrix,
)


def _random_fixed_support_matrix(
    generator: np.random.Generator,
    *,
    rows: int,
    columns: int,
    support: int,
) -> np.ndarray:
    codes = np.zeros((rows, columns), dtype=np.int8)
    for group in codes.reshape(-1, 128):
        positions = generator.choice(128, size=support, replace=False)
        group[positions] = generator.choice(
            np.array([-1, 1], dtype=np.int8), size=support
        )
    return codes


@unittest.skipUnless(shutil.which("cc"), "a C11 compiler is unavailable")
class Qwen38FixedSupportCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        repo = Path(__file__).resolve().parents[1]
        source = repo / "clark_quantize/qwen38_fixed_support_cpu.c"
        extension = ".dylib" if platform.system() == "Darwin" else ".so"
        output = Path(cls.temporary.name) / f"fixed-support{extension}"
        shared_flags = (
            ["-dynamiclib"] if platform.system() == "Darwin"
            else ["-shared", "-fPIC"]
        )
        subprocess.run(
            [
                "cc", "-std=c11", "-O3", "-Wall", "-Wextra", "-Werror",
                *shared_flags, str(source), "-o", str(output),
            ],
            check=True,
        )
        library = load_fixed_support_cpu_library(output)
        if library is None:
            raise RuntimeError("fresh fixed-support library did not load")
        cls.native: FixedSupportCPULibrary = library

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_qwen_widths_and_invalid_zero_support(self) -> None:
        for support, expected in ((54, 176), (60, 184), (64, 189)):
            width = np.zeros(1, dtype=np.uint32)
            status = self.native.library.cq3_hfs_group_width(
                support,
                width.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            )
            self.assertEqual(status, 0)
            self.assertEqual(int(width[0]), expected)
        codes = np.zeros((1, 128), dtype=np.int8)
        with self.assertRaisesRegex(RuntimeError, "invalid fixed-support"):
            native_pack_fixed_support_matrix(
                codes, nonzero_per_group=0, library=self.native
            )

    def test_native_payload_is_byte_identical_to_frozen_python_codec(self) -> None:
        generator = np.random.default_rng(158)
        for support in (54, 60, 64):
            codes = _random_fixed_support_matrix(
                generator, rows=7, columns=384, support=support
            )
            expected = pack_fixed_support_matrix(
                codes, nonzero_per_group=support
            )
            observed = native_pack_fixed_support_matrix(
                codes, nonzero_per_group=support, library=self.native
            )
            self.assertEqual(observed, expected)
            np.testing.assert_array_equal(
                native_unpack_fixed_support_matrix(observed, library=self.native),
                codes,
            )

    def test_boundary_support_ranks_match_python(self) -> None:
        for support in (54, 60, 64):
            codes = np.zeros((2, 128), dtype=np.int8)
            codes[0, :support] = np.where(np.arange(support) & 1, 1, -1)
            codes[1, -support:] = np.where(np.arange(support) & 1, -1, 1)
            expected = pack_fixed_support_matrix(
                codes, nonzero_per_group=support
            )
            observed = native_pack_fixed_support_matrix(
                codes, nonzero_per_group=support, library=self.native
            )
            self.assertEqual(observed, expected)
            np.testing.assert_array_equal(
                native_unpack_fixed_support_matrix(observed, library=self.native),
                codes,
            )

    def test_bad_codes_support_and_noncanonical_states_are_rejected(self) -> None:
        codes = np.zeros((1, 128), dtype=np.int8)
        codes[0, :54] = 1
        codes[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "lie in"):
            native_pack_fixed_support_matrix(
                codes, nonzero_per_group=54, library=self.native
            )
        codes[0, 0] = 0
        with self.assertRaisesRegex(RuntimeError, "non-canonical"):
            native_pack_fixed_support_matrix(
                codes, nonzero_per_group=54, library=self.native
            )

        support = 64
        invalid = fixed_support_state_count(128, support)
        bit_length = 189
        encoded = FixedSupportPayload(
            payload=invalid.to_bytes((bit_length + 7) // 8, "little"),
            bit_length=bit_length,
            rows=1,
            columns=128,
            group_size=128,
            nonzero_per_group=support,
            group_width_bits=bit_length,
        )
        with self.assertRaisesRegex(RuntimeError, "non-canonical"):
            native_unpack_fixed_support_matrix(encoded, library=self.native)

    def test_padding_and_metadata_are_rejected_before_decode(self) -> None:
        codes = np.zeros((1, 128), dtype=np.int8)
        codes[0, :64] = 1
        encoded = native_pack_fixed_support_matrix(
            codes, nonzero_per_group=64, library=self.native
        )
        bad_padding = replace(
            encoded,
            payload=encoded.payload[:-1] + bytes([encoded.payload[-1] | 0x80]),
        )
        with self.assertRaisesRegex(RuntimeError, "non-canonical"):
            native_unpack_fixed_support_matrix(bad_padding, library=self.native)
        for corrupted in (
            replace(encoded, group_size=64),
            replace(encoded, group_width_bits=188),
            replace(encoded, bit_length=188),
            replace(encoded, payload=encoded.payload[:-1]),
        ):
            with self.assertRaisesRegex(ValueError, "group size|metadata"):
                native_unpack_fixed_support_matrix(corrupted, library=self.native)


if __name__ == "__main__":
    unittest.main()
