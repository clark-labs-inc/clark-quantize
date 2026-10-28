import ast
from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
SCRIPT = RUNTIME / "run_qwen38_27b_aff3_llamacpp_response_gate_v52.sh"
PREREG = RUNTIME / "qwen38_27b_aff3_llamacpp_v52_response_gate_preregistration.json"
V51_RECEIPT = RUNTIME / (
    "receipts/qwen38_27b_aff3_llamacpp_v51/actual-model-smoke.json"
)


def test_v52_captures_the_response_separately_from_bounded_ui_transport() -> None:
    source = SCRIPT.read_text()

    assert 'response_path="$state_dir/assistant.response"' in source
    assert '--output "$response_path.tmp"' in source
    assert "--no-show-timings" in source
    assert 'install -m 0644 /dev/null "$response_path.tmp"' in source
    assert "for path in (stdout_path, stderr_path, response_path)" in source
    assert '"assistant_response": {' in source
    assert "bool(response_bytes.strip())" in source


def test_v52_keeps_the_v51_hard_bounds_and_requires_natural_exit() -> None:
    source = SCRIPT.read_text()

    assert "--property=RuntimeMaxSec=300" in source
    assert "--property=LimitFSIZE=1048576" in source
    assert "--property=MemoryMax=30000000000" in source
    assert "--property=MemorySwapMax=0" in source
    assert 'service_result == "success"' in source
    assert "int(service_exec_status or -1) == 0" in source
    assert "ui_prompt_count <= 1" in source
    assert "ui_exit_count == 1" in source
    assert "not performance_timing_ran" in source
    assert "--pipe" not in source


def test_v52_is_preregistered_to_unchanged_v50_numerics_and_v51_failure() -> None:
    source = SCRIPT.read_text()
    prereg = PREREG.read_text()

    assert "828611badba04de446907c69f50bd1959f9623511e695536ca69e09c47baa168" in source
    assert "6953d033c467029db896efa528b6261eb04999b94d3e731952b75f4ffa5e36d5" in source
    assert "a4cec7372ac21a0093d047c1452277508583ebca1e3f4f66c73a8a74142ccf48" in source
    assert "3bad2da6f1ed34b15f094eb1f3394475d0e46204940c10b1c2d3f50d9a368556" in prereg
    assert V51_RECEIPT.is_file()


def test_v52_embedded_python_is_syntactically_valid() -> None:
    programs = re.findall(r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(), flags=re.DOTALL)

    assert len(programs) == 3
    for program in programs:
        ast.parse(program)
