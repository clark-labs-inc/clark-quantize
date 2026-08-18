from __future__ import annotations

from dataclasses import replace
import hashlib
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
from clark_quantize.qwen38_fixed_support_mask64_cpu import (
    load_mask64_cpu_library,
    native_pack_mask64_matrix,
)
from clark_quantize.qwen38_gain_hash_layer_transcode import (
    load_gain_hash_layer_transcode_cpu_library,
    native_transcode_fixed_support_mask64,
)


def random_fixed_support(
    generator: np.random.Generator, *, rows: int, columns: int
) -> np.ndarray:
    codes = np.zeros((rows, columns), dtype=np.int8)
    for group in codes.reshape(-1, 128):
        support = generator.choice(128, size=64, replace=False)
        group[support] = generator.choice(
            np.array([-1, 1], dtype=np.int8), size=64
        )
    return codes


@unittest.skipUnless(shutil.which("cc"), "a C11 compiler is unavailable")
class Qwen38GainHashLayerTranscodeV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        repo = Path(__file__).resolve().parents[1]
        extension = ".dylib" if platform.system() == "Darwin" else ".so"
        output = Path(cls.temporary.name) / f"gain-layer-transcode-v2{extension}"
        shared_flags = (
            ["-dynamiclib"] if platform.system() == "Darwin" else ["-shared", "-fPIC"]
        )
        subprocess.run(
            [
                "cc",
                "-std=c11",
                "-O3",
                "-Wall",
                "-Wextra",
                "-Werror",
                *shared_flags,
                str(repo / "clark_quantize/qwen38_gain_hash_layer_transcode_v2.c"),
                str(repo / "clark_quantize/qwen38_fixed_support_mask64_cpu.c"),
                "-o",
                str(output),
            ],
            check=True,
        )
        cls.transcode = load_gain_hash_layer_transcode_cpu_library(output)
        cls.mask64 = load_mask64_cpu_library(output)
        if cls.transcode is None or cls.mask64 is None:
            raise RuntimeError("fresh optimized transcode library did not load")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_100_random_streams_match_dense_control_exactly(self) -> None:
        generator = np.random.default_rng(38621)
        observed_hashes = set()
        for _ in range(100):
            rows = int(generator.integers(1, 10))
            columns = int(generator.integers(1, 7)) * 128
            codes = random_fixed_support(generator, rows=rows, columns=columns)
            encoded = pack_fixed_support_matrix(
                codes, group_size=128, nonzero_per_group=64
            )
            observed = native_transcode_fixed_support_mask64(
                encoded, library=self.transcode
            )
            expected = native_pack_mask64_matrix(codes, library=self.mask64)
            self.assertEqual(observed, expected)
            observed_hashes.add(hashlib.sha256(observed.payload).hexdigest())
        self.assertEqual(len(observed_hashes), 100)

    def test_canonicality_checks_survive_windowed_reader(self) -> None:
        invalid_rank = FixedSupportPayload(
            payload=fixed_support_state_count(128, 64).to_bytes(24, "little"),
            bit_length=189,
            rows=1,
            columns=128,
            group_size=128,
            nonzero_per_group=64,
            group_width_bits=189,
        )
        with self.assertRaisesRegex(RuntimeError, "non-canonical"):
            native_transcode_fixed_support_mask64(
                invalid_rank, library=self.transcode
            )

        codes = np.zeros((1, 128), dtype=np.int8)
        codes[0, :64] = 1
        encoded = pack_fixed_support_matrix(
            codes, group_size=128, nonzero_per_group=64
        )
        bad_padding = replace(
            encoded,
            payload=encoded.payload[:-1]
            + bytes((encoded.payload[-1] | 0x80,)),
        )
        with self.assertRaisesRegex(RuntimeError, "non-canonical"):
            native_transcode_fixed_support_mask64(
                bad_padding, library=self.transcode
            )


if __name__ == "__main__":
    unittest.main()
