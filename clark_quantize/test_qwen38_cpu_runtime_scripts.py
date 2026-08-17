"""Static safety checks for the authoritative Qwen3.8-27B CPU runners."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
RUNTIME = (
    REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
)


def load_cute_module():
    name = "qwen38_aligned2_cute_test_module"
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(RUNTIME))
    spec = importlib.util.spec_from_file_location(
        name, RUNTIME / "eval_qwen38_27b_aligned2_cute_v1.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_qwen38_python_runners_do_not_use_json_literals_as_names() -> None:
    invalid: list[str] = []
    for path in sorted(RUNTIME.glob("*qwen38_27b*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in {"true", "false", "null"}:
                    invalid.append(f"{path.name}:{node.lineno}:{node.id}")
    assert not invalid, "JSON literals used as Python names: " + ", ".join(invalid)


def test_qwen38_cute_generation_advances_prompt_and_stops_on_eos() -> None:
    module = load_cute_module()

    class State:
        position = 0

    class Model:
        def __init__(self) -> None:
            self.outputs = iter((21, 22, 99))

        def allocate_state(self, capacity: int):
            assert capacity == 6
            return State()

        def forward_token_topk(self, token_id, state, count, **kwargs):
            assert count == 1
            state.position += 1
            return np.array([0.0], dtype=np.float32), np.array(
                [next(self.outputs)], dtype=np.int64
            )

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [10, 11]

        def decode(self, values, **kwargs):
            return ",".join(str(value) for value in values)

    result = module.generate_one(
        Model(),
        Tokenizer(),
        "prompt",
        max_tokens=4,
        threads=1,
        executor=None,
        cpu_ids=[0],
        row_chunk=4096,
        eos_token_ids={99},
    )
    assert result["output"] == "22"
    assert result["prompt_tokens"] == 2
    assert result["completion_tokens"] == 1
    assert result["forward_tokens"] == 3
    assert result["final_state_position"] == 3
    assert result["finish_reason"] == "stop"
