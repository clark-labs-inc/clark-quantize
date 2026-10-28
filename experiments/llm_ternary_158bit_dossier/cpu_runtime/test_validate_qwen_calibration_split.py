from pathlib import Path

from validate_qwen_calibration_split import validate


def test_frozen_v4_calibration_split_is_balanced_disjoint_and_eval_clean() -> None:
    runtime_dir = Path(__file__).resolve().parent
    result = validate(runtime_dir / "qwen35_4b_cpu_layer30_split_v4.json")

    assert result["valid"] is True
    assert result["rows"] == 48
    assert result["exact_prompt_overlap_ids"] == []
    assert result["splits"]["train"]["rows"] == 24
    assert result["splits"]["validation_selection"]["rows"] == 12
    assert result["splits"]["independent_test"]["rows"] == 12
