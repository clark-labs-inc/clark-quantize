import inspect

import torch

from collect_teacher_rollout_cache import alignment_counts, causal_label_sequence, extract_text, load_model


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
