from pathlib import Path

from eval_llama_cpp_mmlu import (
    extract_choice_probabilities,
    format_prompt,
    validate_completed_prefix,
)


def test_extract_choice_probabilities_requires_exact_four_token_softmax() -> None:
    response = {
        "choices": [
            {
                "message": {"content": "B"},
                "logprobs": {
                    "content": [
                        {
                            "id": 20,
                            "token": " B",
                            "prob": 0.6,
                            "top_probs": [
                                {"id": 10, "prob": 0.1},
                                {"id": 20, "prob": 0.6},
                                {"id": 30, "prob": 0.2},
                                {"id": 40, "prob": 0.1},
                            ],
                        }
                    ]
                },
            }
        ]
    }

    prediction, probabilities, sampled = extract_choice_probabilities(
        response, {"A": 10, "B": 20, "C": 30, "D": 40}
    )

    assert prediction == "B"
    assert probabilities == {"A": 0.1, "B": 0.6, "C": 0.2, "D": 0.1}
    assert sampled["id"] == 20


def test_extract_choice_probabilities_rejects_missing_label() -> None:
    response = {
        "choices": [
            {
                "message": {"content": "B"},
                "logprobs": {
                    "content": [
                        {
                            "id": 20,
                            "token": " B",
                            "prob": 0.7,
                            "top_probs": [
                                {"id": 10, "prob": 0.1},
                                {"id": 20, "prob": 0.7},
                                {"id": 30, "prob": 0.1},
                                {"id": 50, "prob": 0.1},
                            ],
                        }
                    ]
                },
            }
        ]
    }

    try:
        extract_choice_probabilities(
            response, {"A": 10, "B": 20, "C": 30, "D": 40}
        )
    except ValueError as error:
        assert "differs from A/B/C/D" in str(error)
    else:
        raise AssertionError("missing D token must fail closed")


def test_prompt_matches_five_shot_mmlu_contract() -> None:
    support = [
        {
            "question": f"Support {index}?",
            "choices": ["a", "b", "c", "d"],
            "answer": index % 4,
            "subject": "abstract_algebra",
        }
        for index in range(5)
    ]
    row = {
        "question": "Target?",
        "choices": ["one", "two", "three", "four"],
        "answer": 2,
        "subject": "abstract_algebra",
    }

    prompt = format_prompt(row, support, 5)

    assert prompt.startswith(
        "The following are multiple choice questions (with answers) about abstract algebra."
    )
    assert prompt.count("Answer:") == 6
    assert prompt.endswith("D. four\nAnswer:")


def test_partial_records_must_match_frozen_prefix() -> None:
    selected = [
        ({}, {"sample_id": "mmlu:1"}),
        ({}, {"sample_id": "mmlu:2"}),
    ]
    validate_completed_prefix(
        [{"position": 0, "sample_id": "mmlu:1"}], selected
    )
    try:
        validate_completed_prefix(
            [{"position": 0, "sample_id": "mmlu:2"}], selected
        )
    except ValueError as error:
        assert "frozen row order" in str(error)
    else:
        raise AssertionError("wrong partial row order must fail closed")
