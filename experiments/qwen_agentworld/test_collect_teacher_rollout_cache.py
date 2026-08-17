import inspect
from pathlib import Path
from types import SimpleNamespace

import torch

from collect_teacher_rollout_cache import (
    alignment_counts,
    causal_label_sequence,
    checkpoint_contract,
    extract_text,
    load_model,
    teacher_cache_receipt,
    validate_checkpoint_rows,
)


def test_causal_label_sequence_predicts_every_completion_token() -> None:
    prompt = torch.tensor([10, 11, 12])
    completion = torch.tensor([20, 21, 22])

    inputs, keep = causal_label_sequence(prompt, completion)

    assert keep == 3
    assert inputs.tolist() == [10, 11, 12, 20, 21]
    assert inputs[-keep:].tolist() == [12, 20, 21]
    assert completion.tolist() == [20, 21, 22]


def test_extract_text_prefers_nonempty_supported_field() -> None:
    assert extract_text({"text": "  answer me  ", "prompt": "ignored"}) == "answer me"


def test_alignment_counts_distinguishes_greedy_from_topk() -> None:
    top_indices = torch.tensor([[4, 5, 6], [8, 7, 9]])
    targets = torch.tensor([4, 7])

    assert alignment_counts(top_indices, targets) == (1, 2)


def test_collector_loader_accepts_explicit_cpu_device() -> None:
    assert "device" in inspect.signature(load_model).parameters


def test_checkpoint_contract_pins_every_generation_input(tmp_path) -> None:
    args = SimpleNamespace(
        model_dir=tmp_path / "model",
        input=tmp_path / "rows.jsonl",
        max_rows=48,
        max_model_len=256,
        generate_tokens=24,
        top_k=128,
        projection_dim=64,
        projection_seed=20260701,
        seed=20260817,
        device="cpu",
        torch_threads=4,
    )
    contract = checkpoint_contract(
        args,
        input_sha256="source-hash",
        hidden_layers=[30],
        source_rows=48,
    )

    assert contract["model_dir"] == str(Path(args.model_dir).resolve())
    assert contract["input_sha256"] == "source-hash"
    assert contract["source_rows"] == 48
    assert contract["hidden_layers"] == [30]
    assert contract["generate_tokens"] == 24


def test_checkpoint_rows_must_match_frozen_source_prefix() -> None:
    sources = [
        {"id": "knowledge-01", "source": "knowledge"},
        {"id": "math-01", "source": "math"},
    ]
    completed = [
        {"index": 0, "id": "knowledge-01", "task": "knowledge"},
    ]
    validate_checkpoint_rows(completed, sources)

    completed[0]["id"] = "wrong"
    try:
        validate_checkpoint_rows(completed, sources)
    except ValueError as error:
        assert "source id" in str(error)
    else:
        raise AssertionError("mismatched checkpoint source rows must fail")


def test_receipt_can_be_rebuilt_from_a_complete_final_cache(tmp_path) -> None:
    target = tmp_path / "teacher.pt"
    payload = {
        "format": "qwen_agentworld_teacher_behavior_v1",
        "label_semantics": "teacher_completion_next_token_v1",
        "hidden_layers": [30],
        "hidden_projection_dim": 64,
        "runtime_device": "cpu",
        "torch_threads": 4,
        "resumed_from_rows": 2,
        "checkpoint_every": 2,
        "rows": [
            {
                "index": 0,
                "id": "math-1",
                "task": "math",
                "tokens": 5,
                "generated_tokens": 2,
                "greedy_alignment_matches": 1,
                "topk_alignment_matches": 2,
                "completion_text": "42",
            }
        ],
    }
    torch.save(payload, target)

    receipt = teacher_cache_receipt(target, payload)

    assert receipt["rows"] == 1
    assert receipt["topk_alignment_matches"] == 2
    assert receipt["topk_alignment_total"] == 2
    assert receipt["rows_summary"][0]["completion_preview"] == "42"
