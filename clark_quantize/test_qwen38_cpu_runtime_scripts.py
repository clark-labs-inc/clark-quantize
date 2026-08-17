"""Static safety checks for the authoritative Qwen3.8-27B CPU runners."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import threading
from unittest import mock

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


def load_persistent_affinity_module():
    name = "qwen38_persistent_affinity_test_module"
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(RUNTIME))
    spec = importlib.util.spec_from_file_location(
        name, RUNTIME / "benchmark_qwen38_27b_persistent_affinity_v1.py"
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


def test_qwen38_persistent_executor_pins_each_worker_once() -> None:
    module = load_persistent_affinity_module()
    local = threading.local()
    set_calls: list[tuple[int, frozenset[int]]] = []
    lock = threading.Lock()

    def set_affinity(pid: int, values: set[int]) -> None:
        affinity = frozenset(values)
        local.affinity = affinity
        with lock:
            set_calls.append((threading.get_native_id(), affinity))

    def get_affinity(pid: int) -> set[int]:
        return set(getattr(local, "affinity", frozenset({0, 2, 4, 6})))

    with mock.patch.object(
        module.os, "sched_setaffinity", set_affinity, create=True
    ), mock.patch.object(
        module.os, "sched_getaffinity", get_affinity, create=True
    ):
        with module.persistent_pinned_executor([0, 2, 4, 6]) as (
            executor,
            records,
        ):
            observed = [
                executor.submit(module.os.sched_getaffinity, 0).result()
                for _ in range(8)
            ]

    assert len(records) == 4
    assert sorted(record["affinity"] for record in records) == [
        [0],
        [2],
        [4],
        [6],
    ]
    assert len(set(record["native_thread_id"] for record in records)) == 4
    assert len(set_calls) == 4
    assert all(len(value) == 1 for value in observed)


def test_qwen38_service_commit_chain_is_monotonic() -> None:
    if not (REPO / ".git").exists():
        return
    units = (
        "quant-qwen38-27b-layer0-mlp-activation-qat-watch-v1.service",
        "quant-qwen38-27b-layer0-mlp-activation-qat-paired-watch-v1.service",
        "quant-qwen38-27b-row-normalized-scale-codebook4-watch-v1.service",
        "quant-qwen38-27b-full-row-normalized-scale-codebook4-watch-v1.service",
        "quant-qwen38-27b-row-scale4-kernel-watch-v1.service",
        "quant-qwen38-27b-block203-row-scale4-direct-gemv-watch-v1.service",
        "quant-qwen38-27b-block203-row-scale4-constdiv-watch-v1.service",
        "quant-qwen38-27b-block203-row-scale4-chunk16-watch-v1.service",
        "quant-qwen38-27b-cq3-row-normalized-export-watch-v1.service",
        "quant-qwen38-27b-cq3-runtime-mmap-watch-v1.service",
        "quant-qwen38-27b-cq3-layer0-mlp-watch-v1.service",
        "quant-qwen38-27b-aligned2-cache-watch-v1.service",
        "quant-qwen38-27b-aligned2-full-cache-watch-v1.service",
        "quant-qwen38-27b-recurrent-delta-avx2-watch-v2.service",
        "quant-qwen38-27b-aligned2-real-layers-watch-v1.service",
        "quant-qwen38-27b-aligned2-full-text-watch-v1.service",
        "quant-qwen38-27b-aligned2-cute-watch-v1.service",
        "quant-qwen38-27b-aligned2-math500-watch-v1.service",
        "quant-qwen38-27b-aligned2-prefill-skip-lm-head-watch-v1.service",
        "quant-qwen38-27b-persistent-affinity-watch-v1.service",
    )
    commits = []
    for unit in units:
        content = (RUNTIME / unit).read_text()
        match = re.search(r"^Environment=EXPECTED_COMMIT=([0-9a-f]{40})$", content, re.M)
        assert match is not None, f"{unit} does not pin a full commit"
        commits.append(match.group(1))
    for parent, child, unit in zip(commits, commits[1:], units[1:]):
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", parent, child],
            cwd=REPO,
            check=False,
        )
        assert result.returncode == 0, (
            f"{unit} moves Qwen3.8 service checkout backward: {parent} -> {child}"
        )
