from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "triton_awq_dequantize",
    "triton_awq_gemm",
    "triton_flash_prefill_attention",
    "triton_matmul_persistent",
    "triton_topk_log_softmax",
)


def _load_runner(task: str):
    path = REPO_ROOT / f"tasks/triton2triton/vllm/{task}/scripts/task_runner.py"
    spec = importlib.util.spec_from_file_location(f"integrity_{task}", path)
    module = importlib.util.module_from_spec(spec)
    previous_cwd = os.getcwd()
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous_cwd)
    return module


def _snapshot(runner, torch, paths):
    runner_globals = {
        name: runner._evaluator_value_token(value)
        for name, value in vars(runner).items()
        if not name.startswith("__")
    }
    torch_attributes = {
        path: runner._resolve_attribute(torch, path) for path in paths
    }
    return runner_globals, torch_attributes


@pytest.mark.parametrize("task", TASKS)
def test_runner_integrity_guard_detects_torch_and_checker_global_mutation(task) -> None:
    runner = _load_runner(task)
    allclose = object()
    event = object()
    torch = SimpleNamespace(
        allclose=allclose,
        cuda=SimpleNamespace(Event=event),
    )
    paths = ("allclose", "cuda.Event")
    assert set(paths) <= set(runner._TRUSTED_TORCH_PATHS)
    snapshot = _snapshot(runner, torch, paths)
    assert runner._verify_evaluator_integrity(torch, snapshot) is None

    torch.allclose = object()
    assert "trusted torch primitive allclose" in runner._verify_evaluator_integrity(
        torch, snapshot
    )
    torch.allclose = allclose

    torch.cuda.Event = object()
    assert "trusted torch primitive cuda.Event" in runner._verify_evaluator_integrity(
        torch, snapshot
    )
    torch.cuda.Event = event

    runner.candidate_injected_checker_global = object()
    try:
        assert "changed evaluator globals" in runner._verify_evaluator_integrity(
            torch, snapshot
        )
    finally:
        del runner.candidate_injected_checker_global


@pytest.mark.parametrize(
    ("task", "primitive"),
    [
        ("triton_awq_dequantize", "equal"),
        ("triton_awq_dequantize", "isfinite"),
        ("triton_awq_gemm", "equal"),
        ("triton_awq_gemm", "isfinite"),
        ("triton_flash_prefill_attention", "equal"),
        ("triton_flash_prefill_attention", "isfinite"),
        ("triton_matmul_persistent", "equal"),
        ("triton_matmul_persistent", "isfinite"),
        ("triton_topk_log_softmax", "equal"),
    ],
)
def test_runner_integrity_guard_covers_exact_trusted_comparison_primitive(
    task, primitive
) -> None:
    runner = _load_runner(task)
    original = object()
    torch = SimpleNamespace(**{primitive: original})
    snapshot = _snapshot(runner, torch, (primitive,))

    setattr(torch, primitive, object())

    assert f"trusted torch primitive {primitive}" in (
        runner._verify_evaluator_integrity(torch, snapshot)
    )
