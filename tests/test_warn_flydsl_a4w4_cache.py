"""The a4w4 score checks every measured output under distinct valid inputs."""

import ast
import importlib.util
import json
import math
import types
from pathlib import Path

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / "tasks/torch2flydsl/gemm_a4w4_kernel"
spec = importlib.util.spec_from_file_location(
    "a4w4_sample_controls", TASK / "scripts/sample_controls.py")
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


def _inputs():
    generator = torch.Generator().manual_seed(20260401)
    a = torch.randn((4, 32), dtype=torch.bfloat16, generator=generator)
    w = torch.randn((8, 32), dtype=torch.bfloat16, generator=generator)
    return a, w


def _reference(a, w):
    return (a.float() @ w.float().T).to(torch.bfloat16)


def _compare(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def _measured_run(candidate, *, samples=100):
    a, w = _inputs()
    stream = controls.MeasuredInputStream(
        a, w, seed=20260401, case_index=0, samples=samples)
    for _ in range(10):
        candidate(a, w)
    for _ in range(samples):
        stream.prepare()
        stream.observe(candidate(a, w))
    assert stream.prepared == len(stream.outputs) == samples
    return stream, a, w


def test_every_reported_sample_is_checked_and_the_first_is_original():
    seen = []
    def candidate(a, w):
        seen.append((a.clone(), w.clone()))
        return _reference(a, w)

    stream, a, w = _measured_run(candidate)
    assert torch.equal(seen[10][0], stream.original_a)
    assert torch.equal(seen[10][1], stream.original_w)
    assert all(not torch.equal(seen[i][0], seen[j][0])
               for i in range(10, 110) for j in range(i + 1, 110))
    last = stream.validate(lambda: _reference(a, w), _compare)
    assert torch.equal(last, _reference(*seen[-1]))
    assert stream.measuring is False
    a.neg_()
    stream.prepare()  # The changed-input replay must not be reset.
    assert torch.equal(a, -seen[-1][0])


def test_wrong_middle_measured_output_fails_even_if_last_output_is_correct():
    calls = 0
    def candidate(a, w):
        nonlocal calls
        calls += 1
        result = _reference(a, w)
        return torch.zeros_like(result) if calls == 61 else result

    stream, a, w = _measured_run(candidate)
    assert torch.equal(stream.outputs[-1], _reference(a, w))
    with pytest.raises(AssertionError):
        stream.validate(lambda: _reference(a, w), _compare)


def test_warmup_output_cache_fails_on_actual_measured_sample():
    cached = None
    original_a, original_w = _inputs()
    def candidate(a, w):
        nonlocal cached
        if cached is None:
            cached = _reference(a, w)
        return cached  # A scored-sample shortcut with no actual recomputation.

    stream, a, w = _measured_run(candidate)
    assert torch.equal(stream.outputs[-1], cached)
    with pytest.raises(AssertionError):
        stream.validate(lambda: _reference(a, w), _compare)
    assert torch.equal(stream.original_a, original_a)
    assert torch.equal(stream.original_w, original_w)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "missing"])
def test_observer_rejects_invalid_measured_output_before_host_copy(invalid):
    a, w = _inputs()
    stream = controls.MeasuredInputStream(
        a, w, seed=20260401, case_index=0, samples=1)
    stream.prepare()
    output = _reference(a, w)
    if invalid == "shape":
        output = output[:, :1]
    elif invalid == "dtype":
        output = output.float()
    else:
        output = None
    with pytest.raises(AssertionError):
        stream.observe(output)
    assert stream.outputs == []


@pytest.mark.parametrize("key_kind", ["version", "content"])
def test_exact_input_caches_cannot_reuse_work_across_distinct_samples(key_kind):
    cache = {}
    hits = 0
    full_calls = 0
    def candidate(a, w):
        nonlocal hits, full_calls
        if key_kind == "version":
            key = (a._version, w._version)
        else:
            key = (a.view(torch.uint16).numpy().tobytes(),
                   w.view(torch.uint16).numpy().tobytes())
        if key in cache:
            hits += 1
            return cache[key]
        full_calls += 1
        result = _reference(a, w)
        cache[key] = result
        return result

    stream, a, w = _measured_run(candidate)
    stream.validate(lambda: _reference(a, w), _compare)
    # Content cache may reuse the original first sample from external warmup;
    # none of the 99 new samples can reuse a warmup or another measured result.
    assert (hits, full_calls) == ((9, 101) if key_kind == "version" else (10, 100))


@pytest.mark.parametrize("entrypoint", ["run_benchmark", "arena_benchmark"])
@pytest.mark.parametrize("behavior", ["correct", "warmup_cache", "wrong_middle",
                                     "warmup_mutates", "measured_mutates"])
def test_actual_benchmark_entrypoints_check_scored_outputs(
    entrypoint, behavior, monkeypatch, tmp_path,
):
    """CPU Event simulator invokes the harness's real preparation and observer."""
    replay_spec = importlib.util.spec_from_file_location(
        "a4w4_replay_checks", TASK / "scripts/replay_checks.py")
    replay = importlib.util.module_from_spec(replay_spec)
    replay_spec.loader.exec_module(replay)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

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

    measured_callbacks = []
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
            measured_callbacks.append((prepare_fn, timed_run.after_sample))
            timed_run.outputs = output
            timed_run._rerun = lambda: (prepare_fn(), fn())[1]
        return 1.0, {"benchmark_method": "cuda_event_fallback",
                     "benchmark_fallback_reason": fallback_reason,
                     "benchmark_samples": repetition}

    call_count = 0
    warmup_output = None
    def compute(a, w):
        nonlocal call_count, warmup_output
        call_count += 1
        out = _reference(a, w)
        if warmup_output is None:
            warmup_output = out
        if behavior == "warmup_mutates" and call_count == 2:
            a.add_(1)
        if behavior == "measured_mutates" and call_count == 62:
            a.add_(1)
        if behavior == "warmup_cache":
            return warmup_output
        if behavior == "wrong_middle" and call_count == 62:
            return torch.zeros_like(out)
        return out

    class Model:
        def to(self, device):
            return self
        def eval(self):
            return self
        def __call__(self, a, w):
            return _reference(a, w)

    model_module = types.SimpleNamespace(Model=Model)
    candidate_module = types.SimpleNamespace(flydsl_gemm_a4w4=compute)
    created_inputs = []
    def make_inputs(*args):
        pair = _inputs()
        created_inputs.append(pair)
        return pair
    source = ast.parse((TASK / "test_kernel_harness.py").read_text())
    names = {"_norm_worst", "_checked_quant_gemm_output",
             "_compare_quant_gemm_output", "_gemm_replay_validator",
             "_measured_stream", entrypoint}
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
        "MODEL_FILE": "model.py", "KERNEL_ENTRY": "flydsl_gemm_a4w4",
        "SHAPES": [{"name": "cpu_control", "m": 4, "n": 8, "k": 32}],
        "SEED": 20260401, "TOL": 1e-2,
        "_make_inputs": make_inputs,
        "_load_module": lambda directory, filename, alias:
            model_module if filename == "model.py" else candidate_module,
        "_retry": lambda fn, **kwargs: fn(),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]),
                 str(TASK / "test_kernel_harness.py"), "exec"), namespace)
    if behavior == "correct":
        result = namespace[entrypoint](verbose=False)
        if entrypoint == "run_benchmark":
            result = json.loads((tmp_path / "build/performance_report.json").read_text())
        assert result[0]["measured_sample_outputs_checked"] == 100
        assert result[0]["timed_output_checked"] is True
        assert len(measured_callbacks) == 1
    else:
        with pytest.raises(AssertionError):
            namespace[entrypoint](verbose=False)
        assert len(measured_callbacks) == (
            0 if behavior in {"warmup_mutates", "measured_mutates"} else 1)
    original_a, original_w = _inputs()
    assert torch.equal(created_inputs[-1][0], original_a)
    assert torch.equal(created_inputs[-1][1], original_w)
