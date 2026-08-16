from types import SimpleNamespace

from train_block_survival_qat import selected_rows


def test_selected_rows_interleaves_tasks_after_balancing() -> None:
    cache = {
        "rows": [
            {"id": f"a:{index}", "task": "a", "tokens": 10}
            for index in range(5)
        ]
        + [
            {"id": f"b:{index}", "task": "b", "tokens": 10}
            for index in range(2)
        ]
        + [
            {"id": f"c:{index}", "task": "c", "tokens": 10}
            for index in range(3)
        ]
    }
    args = SimpleNamespace(
        max_train_tokens=20,
        failures=None,
        shuffle_rows=False,
        row_seed=1,
        max_rows_per_task=4,
        max_train_rows=None,
        interleave_tasks=True,
    )

    rows = selected_rows(cache, args)

    assert [row["task"] for row in rows] == ["a", "b", "c", "a", "b", "c", "a", "c", "a"]
