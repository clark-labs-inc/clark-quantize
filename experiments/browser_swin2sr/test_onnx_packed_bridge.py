import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from onnx_packed_bridge import build_packed_linear_model, pack_trits_5, trit_lookup


class PackedOnnxBridgeTests(unittest.TestCase):
    def test_five_trits_round_trip_through_byte_lookup(self):
        codes = np.asarray([-1, 0, 1, 1, -1, 0, 1], dtype=np.int8)

        packed, pad = pack_trits_5(codes)
        decoded = trit_lookup()[packed].reshape(-1)[: codes.size]

        np.testing.assert_array_equal(decoded, codes)
        self.assertEqual(pad, 3)

    def test_rejects_non_ternary_values(self):
        with self.assertRaisesRegex(ValueError, "ternary codes"):
            pack_trits_5(np.asarray([-1, 0, 2], dtype=np.int8))

    def test_builds_compact_standard_onnx_graph(self):
        rng = np.random.default_rng(4)
        out_features, in_features, group_size = 60, 120, 60
        values = out_features * in_features
        codes = rng.integers(-1, 2, size=values, dtype=np.int8)
        scales = np.ones(values // group_size, dtype=np.float32) * 0.025
        bias = np.zeros(out_features, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "packed.onnx"
            receipt = build_packed_linear_model(
                codes=codes,
                scales=scales,
                bias=bias,
                out_features=out_features,
                in_features=in_features,
                group_size=group_size,
                output=output,
            )
            model = onnx.load_model(output)

        self.assertEqual(receipt["packed_code_bytes"], 1_440)
        self.assertEqual(receipt["scale_bytes"], 240)
        self.assertLess(receipt["model_bytes"], 10_000)
        self.assertLess(receipt["weight_bits_per_value"], 2.0)
        self.assertEqual(model.graph.name, "packed_ternary_linear")


if __name__ == "__main__":
    unittest.main()
