"""The six scored FlyDSL Event paths check every measured invocation."""

import ast
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1] / "tasks/torch2flydsl"
TASKS = (
    "batched_gemm_a8w8_kernel",
    "gemm_a16w8_blockscale_kernel",
    "gemm_a16wfp4_kernel",
    "gemm_a8w8_per_token_scale_kernel",
    "gemm_afp4wfp4_kernel",
    "gemm_afp8wfp8_kernel",
)


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("task,behavior,provided_baseline", [
    (task, behavior, False)
    for task in TASKS
    for behavior in (
        "correct", "wrong_middle", "warmup_cache", "warmup_mutates",
        "measured_mutates", "weight_mutates",
    )
] + [("gemm_a16wfp4_kernel", "correct", True)])
def test_actual_arena_benchmark_checks_all_samples_and_inputs(
    task, behavior, provided_baseline, tmp_path, monkeypatch,
):
    directory = ROOT / task
    controls = _module(directory / "scripts/sample_controls.py", task + "_controls")
    replay = _module(directory / "scripts/replay_checks.py", task + "_replay")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setitem(sys.modules, "aiter", types.ModuleType("aiter"))
    batched = task.startswith("batched_")

    def fresh_inputs():
        generator = torch.Generator().manual_seed(20260401)
        if batched:
            return (
                torch.randn((2, 4, 32), dtype=torch.bfloat16, generator=generator),
                torch.randn((2, 8, 32), dtype=torch.bfloat16, generator=generator),
            )
        return (
            torch.randn((4, 32), dtype=torch.bfloat16, generator=generator),
            torch.randn((8, 32), dtype=torch.bfloat16, generator=generator),
        )

    def reference(left, right):
        if batched:
            return torch.bmm(left.float(), right.float().transpose(1, 2)).to(torch.bfloat16)
        return (left.float() @ right.float().T).to(torch.bfloat16)

    prepared = [False]
    prepared_calls = []

    def prepare_model(device):
        prepared[0] = True
        prepared_calls.append(device)

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, left, right):
            if task == "gemm_a16wfp4_kernel":
                assert prepared[0], "Fresh FP4 decode table was not prepared before reference"
            return reference(left, right)

    class TimedRun:
        def __init__(self):
            self.outputs = None
            self.after_sample = None
            self._rerun = None

        @property
        def bound(self):
            return self._rerun is not None

        def rerun(self):
            self.outputs = self._rerun()
            return self.outputs

    measured = []

    def benchmark(fn, *, warmup, repetition, use_cuda_graph,
                  fallback_reason, timed_run=None, prepare_fn=None):
        assert warmup == 0 and repetition == 100 and use_cuda_graph is False
        output = None
        for _ in range(repetition):
            if prepare_fn is not None:
                prepare_fn()
            output = fn()
            if timed_run is not None:
                timed_run.after_sample(output)
        if timed_run is not None:
            measured.append(timed_run)
            timed_run.outputs = output
            timed_run._rerun = lambda: (prepare_fn(), fn())[1]
        return 1.0, {"benchmark_method": "cuda_event_fallback",
                     "benchmark_fallback_reason": fallback_reason,
                     "benchmark_samples": repetition}

    call_count = 0
    warmup_output = None

    def compute(left, right):
        nonlocal call_count, warmup_output
        call_count += 1
        output = reference(left, right)
        if warmup_output is None:
            warmup_output = output
        if behavior == "warmup_mutates" and call_count == 2:
            left.add_(1)
        if behavior == "measured_mutates" and call_count == 62:
            left.add_(1)
        if behavior == "weight_mutates" and call_count == 62:
            right.add_(1)
        if behavior == "warmup_cache":
            return warmup_output
        if behavior == "wrong_middle" and call_count == 62:
            return torch.zeros_like(output)
        return output

    candidate_name = "flydsl_" + task.removesuffix("_kernel")
    created_inputs = []

    def make_inputs(*args):
        pair = fresh_inputs()
        created_inputs.append(pair)
        return pair

    def load_module(directory, filename, alias):
        if filename == "model.py":
            return types.SimpleNamespace(Model=Model,
                                         prepare_mxfp4_values=prepare_model)
        if provided_baseline:
            return None
        return types.SimpleNamespace(**{candidate_name: compute})

    source = ast.parse((directory / "test_kernel_harness.py").read_text())
    names = {"_norm_worst", "_checked_batched_output", "_compare_batched_output",
             "_checked_quant_gemm_output", "_compare_quant_gemm_output",
             "_gemm_replay_validator", "arena_benchmark"}
    functions = [node for node in source.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "torch": torch, "math": math, "json": json, "Path": Path,
        "TimedRun": TimedRun, "MeasuredInputStream": controls.MeasuredInputStream,
        "benchmark_cuda_graph_or_events": benchmark,
        "require_tensor_contract": replay.require_tensor_contract,
        "require_unchanged": replay.require_unchanged,
        "verify_timed_run": replay.verify_timed_run,
        "_KERNEL_DIR": str(tmp_path), "KERNEL_FILE": "kernel.py",
        "MODEL_FILE": "model.py", "KERNEL_ENTRY": candidate_name,
        "SHAPES": [{"name": "cpu_control", "b": 2, "m": 4, "n": 8, "k": 32}],
        "SEED": 20260401, "TOL": 1e-2,
        "_make_inputs": make_inputs,
        "_load_module": load_module,
        "_aiter_ground_truth": lambda module, left, right: compute(left, right),
        "_retry": lambda fn, **kwargs: fn(),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]),
                 str(directory / "test_kernel_harness.py"), "exec"), namespace)
    if behavior == "correct":
        result = namespace["arena_benchmark"](verbose=False)
        assert result[0]["measured_sample_outputs_checked"] == 100
        assert result[0]["timed_output_checked"] is True
        assert len(measured) == 1
        if task == "gemm_a16wfp4_kernel":
            assert prepared_calls == [torch.device("cpu")]
    else:
        with pytest.raises(AssertionError):
            namespace["arena_benchmark"](verbose=False)
    original_left, original_right = fresh_inputs()
    assert torch.equal(created_inputs[-1][0], original_left)
    assert torch.equal(created_inputs[-1][1], original_right)
