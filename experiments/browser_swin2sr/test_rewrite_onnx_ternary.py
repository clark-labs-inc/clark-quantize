import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rewrite_onnx_ternary import recover_ternary_groups, rewrite_ternary_initializers


def small_model(weight: np.ndarray, opset: int = 18) -> onnx.ModelProto:
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "small",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, weight.shape[0]])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, weight.shape[1]])],
        [numpy_helper.from_array(weight, "weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 9
    return model


class RewriteOnnxTernaryTests(unittest.TestCase):
    def test_recovers_group_codes_and_scales(self):
        values = np.asarray(
            [-0.5, 0.0, 0.5, -0.5, 0.25, 0.0, -0.25, 0.25],
            dtype=np.float32,
        )

        codes, scales, error = recover_ternary_groups(values, 4)

        np.testing.assert_array_equal(codes, [-1, 0, 1, -1, 1, 0, -1, 1])
        np.testing.assert_array_equal(scales, [0.5, 0.25])
        self.assertEqual(error, 0.0)

    def test_rejects_non_ternary_group(self):
        with self.assertRaisesRegex(ValueError, "not hard ternary"):
            recover_ternary_groups(np.asarray([0.0, 0.25, 0.4]), 3)

    def test_rewrites_dense_initializer_without_leaving_dense_payload(self):
        weight = np.asarray(
            [
                [-0.5, 0.0, 0.5],
                [-0.5, 0.25, 0.0],
                [-0.25, 0.25, -0.75],
                [0.0, 0.75, -0.75],
            ],
            dtype=np.float32,
        )
        model = small_model(weight)

        ledger = rewrite_ternary_initializers(
            model,
            {"weight"},
            {"weight": 4},
        )

        initializer_names = {item.name for item in model.graph.initializer}
        node_outputs = {name for node in model.graph.node for name in node.output}
        self.assertNotIn("weight", initializer_names)
        self.assertIn("weight", node_outputs)
        self.assertEqual(ledger["tensor_count"], 1)
        self.assertEqual(ledger["values"], 12)
        self.assertEqual(ledger["packed_code_bytes"], 3)
        input_values = np.asarray([[1.0, -0.5, 0.25, 2.0]], dtype=np.float32)
        actual = ReferenceEvaluator(model).run(None, {"input": input_values})[0]
        expected = input_values @ weight
        np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=5e-4)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rewritten.onnx"
            onnx.save_model(model, output)
            self.assertLess(output.stat().st_size, 10_000)

    def test_rewrites_opset_11_model_with_attribute_unsqueeze(self):
        weight = np.asarray(
            [
                [-0.5, 0.0],
                [0.5, -0.5],
                [0.25, 0.0],
                [-0.25, 0.25],
            ],
            dtype=np.float32,
        )
        model = small_model(weight, opset=11)

        ledger = rewrite_ternary_initializers(
            model,
            {"weight"},
            {"weight": 4},
        )
        input_values = np.asarray([[1.0, -0.5, 0.25, 2.0]], dtype=np.float32)
        actual = ReferenceEvaluator(model).run(None, {"input": input_values})[0]

        np.testing.assert_allclose(actual, input_values @ weight, atol=5e-4, rtol=5e-4)
        self.assertEqual(ledger["opset_version"], 11)

    def test_preserves_source_grouping_before_export_transpose(self):
        source = np.asarray(
            [
                [-0.5, 0.0, 0.5, -0.5],
                [0.25, 0.0, -0.25, 0.25],
            ],
            dtype=np.float32,
        )
        model = small_model(source.T)

        ledger = rewrite_ternary_initializers(
            model,
            {"weight"},
            {"weight": 4},
            source_values={"weight": source},
            transpose_outputs={"weight"},
        )
        input_values = np.asarray([[1.0, -0.5, 0.25, 2.0]], dtype=np.float32)
        actual = ReferenceEvaluator(model).run(None, {"input": input_values})[0]

        np.testing.assert_allclose(actual, input_values @ source.T, atol=5e-4, rtol=5e-4)
        self.assertEqual(ledger["tensors"][0]["output_transform"], "transpose")

    def test_requires_an_explicit_group_size_for_every_tensor(self):
        model = small_model(np.eye(4, dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "exactly match"):
            rewrite_ternary_initializers(model, {"weight"}, {})


if __name__ == "__main__":
    unittest.main()
