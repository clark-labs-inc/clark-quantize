from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = (
    REPO
    / "experiments/llm_ternary_158bit_dossier/cpu_runtime/receipts"
    / "qwen38_27b_llamacpp_ternary_controls_v1"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_terminal_decision_authenticates_complete_control_receipts() -> None:
    decision = json.loads((ROOT / "terminal-decision.json").read_text())
    assert decision["logical_scope"]["parameters"] == 26_895_998_464
    assert decision["service_terminal_observation"]["result"] == "success"
    assert decision["service_terminal_observation"]["exec_main_status"] == 0
    for descriptor in decision["receipts"].values():
        path = ROOT / descriptor["path"]
        assert path.stat().st_size == descriptor["bytes"]
        assert sha256(path) == descriptor["sha256"]
    controls = json.loads((ROOT / "state/controls/receipt.json").read_text())
    assert controls["complete"] is True
    assert controls["terminal"] is True


def test_both_controls_are_full_denominator_catastrophic_rejections() -> None:
    decision = json.loads((ROOT / "terminal-decision.json").read_text())
    for label in ("TQ1_0", "TQ2_0"):
        control = decision["controls"][label]
        assert control["honest_file_bpw"] > 3.0
        assert control["cute"] == {"correct": 0, "attempted": 20}
        assert control["math500"] == {"correct": 0, "attempted": 8}
        assert control["code_trace"] == {"correct": 0, "attempted": 6}
        assert control["wikitext_ratio_vs_q4"] > 300_000
        assert control["mmlu"]["launched"] is False
        summary_path = ROOT / control["summary"]["path"]
        assert sha256(summary_path) == control["summary"]["sha256"]
    assert decision["decision"]["rescue_tuning_authorized"] is False


def test_peg_scoring_ledger_keeps_every_attempt_in_denominator() -> None:
    decision = json.loads((ROOT / "terminal-decision.json").read_text())
    ledger = ROOT / "state/peg-resume/peg-native-scoring-events.jsonl"
    events = [json.loads(line) for line in ledger.read_text().splitlines() if line]
    assert len(events) == decision["scoring_continuation"]["peg_native_http_500_events"] == 64
    assert all(event["scored_as_incorrect"] is True for event in events)
    assert all(event["upstream_http_status"] == 500 for event in events)
    assert {event["url"].rsplit(":", 1)[-1].split("/", 1)[0] for event in events} == {
        "8110",
        "8111",
    }
