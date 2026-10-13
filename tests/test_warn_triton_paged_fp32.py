"""The paged-prefix public FP32 branch has a real unscored correctness gate."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


TASK = (Path(__file__).resolve().parents[1] / "tasks/triton2triton/vllm/"
        "triton_paged_prefix_prefill_alibi")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = os.getcwd()
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous)
    return module


def test_fp32_control_is_correctness_only_and_crosses_public_tile_boundary():
    data = json.loads((TASK / "workloads.json").read_text())
    original = data["cases"][:5]
    assert [case["test_case_id"] for case in original] == [f"perf{i}" for i in range(1, 6)]
    assert all(case["checks"] == ["correctness", "performance"] for case in original)
    row = next(case for case in data["cases"] if case["test_case_id"] == "float32_branch")
    assert row["checks"] == ["correctness"]
    assert row["params"]["dtypes"]["data"] == "float32"
    assert row["params"]["query_lengths"] == [65, 3]
    assert row["params"]["input_shapes"]["q"] == [68, 8, 64]


def test_fp32_only_wrong_candidate_is_rejected_by_public_wrapper_control(monkeypatch):
    checks = load(TASK / "_arena_checks.py", "_paged_fp32_checks")
    harness = load(TASK / "scripts/task_runner.py", "_paged_fp32_harness")
    # Keep this CPU negative control focused on numerical acceptance. The GPU
    # validator separately checks that the real candidate launches its Triton JIT.
    monkeypatch.setattr(checks, "checked_candidate_call", lambda module, fn, *a, **k: fn(*a, **k))

    def candidate(*args, **kwargs):
        output = args[3]
        if args[0].dtype == torch.float32:
            output.zero_()
        else:
            output.copy_(checks.expected_outputs(harness, dict(enumerate(args)) | kwargs)[0])

    module = SimpleNamespace(context_attention_fwd_alibi=candidate)
    harness.load_module = lambda: module
    for name in ("ragged_permuted_alibi_scale", "float32_branch"):
        args, kwargs = checks.control_inputs(harness, name, "cpu")
        assert args[0].dtype == (torch.float32 if name == "float32_branch" else torch.float16)
        assert kwargs["max_input_len"] == (65 if name == "float32_branch" else 5)
        with checks.checked_modules(harness):
            checked = harness.load_module().context_attention_fwd_alibi
            if name == "float32_branch":
                with pytest.raises(AssertionError):
                    checked(*args, **kwargs)
            else:
                checked(*args, **kwargs)
        assert module.context_attention_fwd_alibi is candidate
