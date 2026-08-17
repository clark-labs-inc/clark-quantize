"""Static safety checks for the authoritative Qwen3.8-27B CPU runners."""
from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = (
    REPO / "experiments/llm_ternary_158bit_dossier/cpu_runtime"
)


def test_qwen38_python_runners_do_not_use_json_literals_as_names() -> None:
    invalid: list[str] = []
    for path in sorted(RUNTIME.glob("*qwen38_27b*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in {"true", "false", "null"}:
                    invalid.append(f"{path.name}:{node.lineno}:{node.id}")
    assert not invalid, "JSON literals used as Python names: " + ", ".join(invalid)
