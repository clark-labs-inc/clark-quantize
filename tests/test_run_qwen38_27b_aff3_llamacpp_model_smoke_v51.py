import ast
from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
SCRIPT = RUNTIME / "run_qwen38_27b_aff3_llamacpp_model_smoke_v51.sh"
WORKER = RUNTIME / "run_qwen38_27b_aff3_llamacpp_model_worker_v51.sh"
PREREG = RUNTIME / "qwen38_27b_aff3_llamacpp_v51_model_gate_preregistration.json"
INCIDENT = RUNTIME / (
    "receipts/qwen38_27b_aff3_llamacpp_v50/actual-model-smoke-incident.json"
)


def test_v51_forces_one_shot_execution_and_hard_resource_bounds() -> None:
    source = SCRIPT.read_text()

    assert "--no-conversation" in source
    assert "--single-turn" in source
    assert "--property=RuntimeMaxSec=300" in source
    assert "--property=LimitFSIZE=1048576" in source
    assert "--property=MemoryMax=30000000000" in source
    assert "--property=MemorySwapMax=0" in source
    assert "--property=AllowedCPUs=0,2,4,6" in source
    assert "--pipe" not in source
    assert 'install -m 0644 /dev/null "$stdout_path.tmp"' in source
    assert 'install -m 0644 /dev/null "$stderr_path.tmp"' in source


def test_v51_rejects_the_observed_v50_false_pass_modes() -> None:
    source = SCRIPT.read_text()

    assert 'service_result == "success"' in source
    assert "int(service_exec_status or -1) == 0" in source
    assert 'interactive_markers = ("available commands:", "\\n> ", "\\n>\\n")' in source
    assert "stream_bounds_respected" in source
    assert "not interactive_mode_ran" in source
    assert "not performance_timing_ran" in source
    assert '"terminal": True' in source
    assert "v50_harness_incident_preserved" in source


def test_v51_is_bound_to_unchanged_runtime_and_preregistered_inputs() -> None:
    source = SCRIPT.read_text()

    assert "e68b0f8f1f6a4bed156e55f4a9107299410048d60198d8bff3d40f5b00e0766c" in source
    assert "6953d033c467029db896efa528b6261eb04999b94d3e731952b75f4ffa5e36d5" in source
    assert "e55a49bfc03e7ae6346a7f3e8240a18cddd64a176a8cc6d61e9ab7f1a38ff92c" in source
    assert "a4cec7372ac21a0093d047c1452277508583ebca1e3f4f66c73a8a74142ccf48" in source
    assert "cb04554a9316af13d59f970701ed42d8c9c1c2cdaa6382a4e26cdc54eacddc42" in PREREG.read_text()
    assert INCIDENT.is_file()


def test_v51_embedded_python_is_syntactically_valid() -> None:
    programs = re.findall(r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(), flags=re.DOTALL)

    assert len(programs) == 3
    for program in programs:
        ast.parse(program)


def test_v51_worker_only_redirects_then_execs_the_bound_command() -> None:
    source = WORKER.read_text()

    assert "set -euo pipefail" in source
    assert 'exec "$@" >"$stdout_path" 2>"$stderr_path"' in source
    assert "llama" not in source.lower()
