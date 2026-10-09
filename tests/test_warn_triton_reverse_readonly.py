"""Reverse-range adapters must reject input corruption even with a correct output."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1] / "tasks"
TASKS = (
    "instruction2triton/rocmbench/test_reverse_range",
    "triton2triton/rocmbench/easy/test_reverse_range",
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("unused_index", (0, 515))
def test_reference_rejects_mutated_unread_input_with_correct_output(task, dtype, unused_index):
    reference = _load(ROOT / task / "_arena_reference.py", "reverse_readonly_reference")
    data = torch.arange(516, dtype=torch.float32).remainder_(17).to(dtype)
    output = torch.flip(data[1:513], [0])
    context = {"data_perf": data, "res_perf_buffer": output.clone()}
    check = reference.prepare(context, None)
    check(output)
    data[unused_index].add_(1)
    with pytest.raises(AssertionError, match="Input argument data_perf was modified"):
        check(output)


class _TimedRun:
    def __init__(self):
        self.outputs = None
        self.after_sample = None
        self.bound = False
        self._replay = None

    def rerun(self):
        self.outputs = self._replay()
        return self.outputs


def _measure_times(*args, **kwargs):
    raise AssertionError("Adapter failed to bind the canonical timed observer")


class _Base:
    def __init__(self, *, op_callable, config):
        self.op_callable = op_callable
        self.config = config
        self.prepare_fn = None

    def run_benchmark(self, *, baseline_callable=None):
        assert baseline_callable is None
        samples, metadata = _measure_times(
            self.op_callable, self.config, prepare_fn=self.prepare_fn)
        return {"timing_ms": {"mean": sum(samples) / len(samples)}, **metadata}


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("bad_call", (None, 1, 2, 3, 5, 7))
def test_adapter_checks_initial_middle_and_changed_replay_input(monkeypatch, task, bad_call):
    reference = _load(ROOT / task / "_arena_reference.py", "reverse_readonly_runtime_reference")
    monkeypatch.setitem(sys.modules, "_arena_reference", reference)
    timer = ModuleType("_aka_benchmark")
    timer.TimedRun = _TimedRun

    def samples(fn, *, repetition, timed_run, prepare_fn=None, **_kwargs):
        assert repetition == 3
        for _ in range(repetition):
            if prepare_fn is not None:
                prepare_fn()
            output = fn()
            timed_run.after_sample(output)
        timed_run.outputs = output
        timed_run._replay = lambda: (prepare_fn(), fn())[1] if prepare_fn else fn()
        timed_run.bound = True
        return [1.0] * repetition, {"benchmark_method": "cuda_graph",
                                   "benchmark_effective_repeats": 1}

    timer.benchmark_cuda_graph_or_events_samples = samples
    monkeypatch.setitem(sys.modules, "_aka_benchmark", timer)
    adapter = _load(ROOT / task / "_arena_eval.py", "reverse_readonly_adapter")

    data_perf = torch.arange(1.0, 517.0)
    original = data_perf.clone()
    res_perf_buffer = torch.empty(512)
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        res_perf_buffer.copy_(torch.flip(data_perf[1:513], [0]))
        if calls == bad_call:
            data_perf[0].add_(1)  # The output is still exactly correct.
        return res_perf_buffer

    row = {"test_case_id": "reverse-readonly"}
    plugin = SimpleNamespace(action="performance", current_row=row, exercised=set())
    checked = adapter.benchmark_type(_Base, plugin, SimpleNamespace())(
        op_callable=operation, config=SimpleNamespace(warm_up=10, repetition=3)
    )
    if bad_call is None:
        checked.run_benchmark()
        assert row["metadata"]["measured_samples_checked"] == 3
        assert torch.equal(data_perf, original)
    else:
        with pytest.raises(AssertionError, match="Input argument data_perf was modified"):
            checked.run_benchmark()
        assert "execution_time_ms" not in row
        if bad_call in (2, 3, 7):
            assert torch.equal(data_perf, original)
    assert _measure_times is _Base.run_benchmark.__globals__["_measure_times"]
