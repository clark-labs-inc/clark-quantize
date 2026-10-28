import hashlib
import json
from pathlib import Path

import torch

from summarize_qwen_cpu_caches import canonical_digest, summarize


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_summarize_validates_complete_paired_caches(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    source_rows = [
        {"id": "train-1", "source": "math", "text": "one"},
        {"id": "select-1", "source": "code", "text": "two"},
        {"id": "test-1", "source": "knowledge", "text": "three"},
    ]
    source_path.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    manifest = {
        "source": {
            "path": source_path.name,
            "sha256": sha256_file(source_path),
            "rows": 3,
            "task_counts": {"math": 1, "code": 1, "knowledge": 1},
        },
        "train": {"indices": [0]},
        "validation_selection": {"indices": [1]},
        "independent_test": {"indices": [2]},
    }
    manifest["content_sha256"] = canonical_digest(manifest)
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps(manifest))

    teacher_rows = []
    activation_rows = []
    for index, source in enumerate(source_rows):
        generated = index + 1
        teacher_rows.append(
            {
                "index": index,
                "id": source["id"],
                "task": source["source"],
                "generated_tokens": generated,
                "greedy_alignment_matches": generated,
                "topk_alignment_matches": generated,
            }
        )
        activation_rows.append(
            {
                "id": source["id"],
                "task": source["source"],
                "positions": generated,
            }
        )
    teacher_path = tmp_path / "teacher.pt"
    torch.save(
        {"format": "qwen_agentworld_teacher_behavior_v1", "rows": teacher_rows},
        teacher_path,
    )
    activation_path = tmp_path / "activation.pt"
    torch.save(
        {
            "format": "qwen_mlp_activation_cache_v1",
            "rows": activation_rows,
            "layers": {30: {"x": torch.ones(6, 4), "y": torch.ones(6, 4)}},
        },
        activation_path,
    )

    result = summarize(
        teacher_path,
        activation_path,
        manifest_path,
        layer=30,
        positions_per_row=24,
    )

    assert result["valid"] is True
    assert result["rows"] == 3
    assert result["topk_alignment_matches"] == 6
    assert result["activation_shape"] == [6, 4]
    assert result["splits"]["independent_test"]["positions"] == 3


def test_summarize_rejects_noncausal_alignment(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(json.dumps({"id": "row-1", "source": "math", "text": "x"}) + "\n")
    manifest = {
        "source": {
            "path": source_path.name,
            "sha256": sha256_file(source_path),
            "task_counts": {"math": 1},
        },
        "train": {"indices": [0]},
        "validation_selection": {"indices": []},
        "independent_test": {"indices": []},
    }
    manifest["content_sha256"] = canonical_digest(manifest)
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps(manifest))
    teacher_path = tmp_path / "teacher.pt"
    torch.save(
        {
            "format": "qwen_agentworld_teacher_behavior_v1",
            "rows": [{
                "index": 0,
                "id": "row-1",
                "task": "math",
                "generated_tokens": 1,
                "greedy_alignment_matches": 0,
                "topk_alignment_matches": 0,
            }],
        },
        teacher_path,
    )
    activation_path = tmp_path / "activation.pt"
    torch.save(
        {
            "format": "qwen_mlp_activation_cache_v1",
            "rows": [{"id": "row-1", "task": "math", "positions": 1}],
            "layers": {30: {"x": torch.ones(1, 4), "y": torch.ones(1, 4)}},
        },
        activation_path,
    )

    try:
        summarize(
            teacher_path,
            activation_path,
            manifest_path,
            layer=30,
            positions_per_row=24,
        )
    except ValueError as error:
        assert "causal top-k alignment" in str(error)
    else:
        raise AssertionError("an incomplete causal alignment must fail validation")
