import unittest

from analyze_layer_sensitivity import select_balanced, summarize


class LayerSensitivityTest(unittest.TestCase):
    def test_balanced_selection_is_deterministic(self):
        records = [
            {"source": source, "record_id": index}
            for source in ("ui", "text")
            for index in range(10)
        ]

        first = select_balanced(records, 4, 17)
        second = select_balanced(records, 4, 17)

        self.assertEqual(first, second)
        self.assertEqual(
            {source: sum(row["source"] == source for row in first) for source in ("ui", "text")},
            {"ui": 4, "text": 4},
        )

    def test_summary_ranks_by_worst_source_relative_error(self):
        accumulator = {
            "safe": {
                "ui": {
                    "squared_error": 1.0,
                    "squared_signal": 10.0,
                    "values": 2,
                },
                "text": {
                    "squared_error": 1.0,
                    "squared_signal": 10.0,
                    "values": 2,
                },
            },
            "ui_sensitive": {
                "ui": {
                    "squared_error": 4.0,
                    "squared_signal": 10.0,
                    "values": 2,
                },
                "text": {
                    "squared_error": 0.5,
                    "squared_signal": 10.0,
                    "values": 2,
                },
            },
        }

        rows = summarize(accumulator, {"safe": 10, "ui_sensitive": 20})

        self.assertEqual(rows[0]["name"], "ui_sensitive")
        self.assertAlmostEqual(rows[0]["worst_source_relative_mse"], 0.4)


if __name__ == "__main__":
    unittest.main()
