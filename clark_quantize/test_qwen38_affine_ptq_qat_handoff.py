from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WATCHER = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
    / "watch_qwen38_27b_affine_ptq_to_qat_v1.sh"
)


def test_handoff_is_qwen38_only_and_pins_both_research_identities() -> None:
    source = WATCHER.read_text()
    assert 'readonly model="Qwen/Qwen3.8-27B"' in source
    assert 'readonly qat_repo_commit="1343f9616fc145ec9dbfb6716f6f8c9e21006cca"' in source
    assert 'readonly parent_runner_commit="1e524a9eb48e223b259a4d3714afc765bb107dcd"' in source
    assert 'readonly qat_prereg_sha256="28d7ef3ca0eca806fc25e8a0b7de315a82e9e295fb2302210ad9720a1fdd0a26"' in source


def test_handoff_requires_a_terminal_parent_pass_before_qat() -> None:
    source = WATCHER.read_text()
    receipt_check = source.index('.complete == true and .terminal == true')
    pass_check = source.index(".gate.passed == true")
    launch = source.index("systemd-run")
    assert receipt_check < pass_check < launch
    assert 'echo "parent PTQ gate rejected; QAT remains unrun"' in source
    assert 'systemctl stop "$parent_unit"' in source


def test_handoff_prevents_scientific_rejection_restart_loops() -> None:
    source = WATCHER.read_text()
    assert "--property=Restart=on-failure" in source
    assert "--property=RestartPreventExitStatus=2" in source
    assert 'if [[ -f "$qat_receipt" ]]' in source
    assert 'systemctl is-active --quiet "$qat_unit"' in source


def test_handoff_times_the_frozen_kernel_while_idle_before_qat() -> None:
    source = WATCHER.read_text()
    stop_parent = source.index('systemctl stop "$parent_unit"')
    benchmark = source.index("--mode benchmark")
    launch_qat = source.rindex("systemd-run")
    assert stop_parent < benchmark < launch_qat
    assert 'readonly kernel_receipt="/workspace/quant-state/qwen38-affine-avx2-x16-benchmark-v17.json"' in source
    assert "taskset -c 0" in source
    assert 'kernel_status" -ne 0 && "$kernel_status" -ne 2' in source
    assert 'echo "AFF3_X16 benchmark was infrastructure-invalid"' in source
