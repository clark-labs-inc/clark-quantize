from pathlib import Path

from seal_staged_scale_codebook_study import (
    ARMS,
    FINAL_BITS,
    SOURCE_FILES,
    sealed_command,
)


def test_staged_arms_isolate_schedule_then_shadow_pressure():
    assert set(ARMS) == {"late_io", "gentle_io", "gentle_io_strong_shadow"}
    late = ARMS["late_io"]
    gentle = ARMS["gentle_io"]
    strong = ARMS["gentle_io_strong_shadow"]
    assert late["schedule_profile"] != gentle["schedule_profile"]
    assert late["hard_shadow_weight_start"] == gentle["hard_shadow_weight_start"]
    assert late["hard_shadow_weight_end"] == gentle["hard_shadow_weight_end"]
    assert gentle["schedule_profile"] == strong["schedule_profile"]
    assert gentle["hard_shadow_weight_start"] != strong["hard_shadow_weight_start"]
    assert gentle["hard_shadow_weight_end"] != strong["hard_shadow_weight_end"]


def test_staged_endpoint_and_source_closure_are_explicit():
    assert FINAL_BITS == {
        "core_linear": 1,
        "image_io_conv": 1,
        "position_bias_linear": 1,
        "residual_body_conv": 1,
    }
    assert len(SOURCE_FILES) == len(set(SOURCE_FILES))
    repository = Path(__file__).resolve().parents[2]
    assert all((repository / relative).is_file() for relative in SOURCE_FILES)


def test_staged_commands_bind_pythonpath_to_the_sealed_tree(tmp_path):
    command = sealed_command(tmp_path, "experiments/browser_swin2sr/train.py", "--x", "1")
    assert command[0] == "/usr/bin/env"
    assert command[1] == (
        f"PYTHONPATH={tmp_path / 'experiments/browser_swin2sr'}:{tmp_path}"
    )
    assert command[-2:] == ["--x", "1"]
