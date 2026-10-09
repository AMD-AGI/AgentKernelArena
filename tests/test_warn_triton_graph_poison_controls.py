"""CPU controls for graph-sample output invalidation and legal GEMM rows."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


TASKS = Path(__file__).resolve().parents[1] / "tasks"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_layernorm_poison_covers_y_as_well_as_statistics():
    task = TASKS / "instruction2triton/rocmbench/layernorm"
    reference = load(task / "_arena_reference.py", "_layernorm_poison_reference")
    x = torch.randn(2, 7)
    w, b = torch.ones(7), torch.zeros(7)
    y, mean, rstd = torch.empty_like(x), torch.empty(2), torch.empty(2)
    write_y = [True]

    def launch(grid, x, y, w, b, mean, rstd, *args):
        xf = x.float()
        center = xf.mean(-1)
        mean.copy_(center)
        rstd.copy_(torch.rsqrt(((xf-center[:, None])**2).mean(-1)+1e-5))
        if write_y[0]:
            y.copy_(torch.nn.functional.layer_norm(x, (7,), w, b, 1e-5))

    module = SimpleNamespace(layernorm_wrapper_fn=launch)
    context = dict(x=x, w=w, b=b, eps=1e-5, normalized_shape_arg=(7,))
    check = reference.prepare(context, module)
    with reference.capture_side_outputs(module) as statistics_for:
        module.layernorm_wrapper_fn((2,), x, y, w, b, mean, rstd)
        check(y, statistics_for(y))
        statistics_for.poison_all()
        assert torch.isnan(y).all() and torch.isnan(mean).all() and torch.isnan(rstd).all()
        write_y[0] = False
        module.layernorm_wrapper_fn((2,), x, y, w, b, mean, rstd)
        with pytest.raises(ValueError, match="nonfinite"):
            check(y, statistics_for(y))
    assert module.layernorm_wrapper_fn is launch


@pytest.mark.parametrize("fault_row", (None, 3, 5, 8))
def test_multreduce_unscored_controls_cover_every_new_legal_m(fault_row):
    task = TASKS / "triton2triton/rocmbench/hard/triton_multreduce_matmul_kernel"
    reference = load(task / "_arena_reference.py", "_multreduce_m_reference")
    seen = []

    def matmul(provider, a, b, bias):
        assert provider == "triton-multreduce"
        rows = a.shape[0]
        seen.append((rows, bias is not None, a.shape[1], b.shape[1]))
        result = a @ b
        if bias is not None:
            result = result + bias[:, None]
        if rows == fault_row:
            result = result.clone()
            result[-1, -1] += 1
        return result

    module = SimpleNamespace(matmul=matmul)
    if fault_row is None:
        reference.check_declared_m_controls(module, "cpu")
        assert seen[:6] == [(m, bool(m % 2), 31, 23) for m in range(3, 9)]
        assert seen[6:] == [
            (3, False, 4096, 4096), (4, True, 257, 129),
            (5, False, 128, 256), (6, True, 129, 257),
            (7, False, 513, 512), (8, True, 512, 513),
        ]
    else:
        with pytest.raises(reference.NumericalMismatch):
            reference.check_declared_m_controls(module, "cpu")
        assert seen[-1][0] == fault_row


def test_multreduce_large_m3_branch_cannot_hide_behind_small_controls():
    task = TASKS / "triton2triton/rocmbench/hard/triton_multreduce_matmul_kernel"
    reference = load(task / "_arena_reference.py", "_multreduce_large_m_reference")

    def matmul(provider, a, b, bias):
        result = a @ b
        if bias is not None:
            result += bias[:, None]
        if a.shape == (3, 4096) and b.shape == (4096, 4096):
            return torch.zeros_like(result)
        return result

    with pytest.raises(reference.NumericalMismatch):
        reference.check_declared_m_controls(SimpleNamespace(matmul=matmul), "cpu")


@pytest.mark.parametrize("skip_middle", (False, True))
def test_multreduce_graph_sample_must_rewrite_poisoned_output(monkeypatch, skip_middle):
    task = TASKS / "triton2triton/rocmbench/hard/triton_multreduce_matmul_kernel"
    reference = load(task / "_arena_reference.py", "_arena_reference")
    adapter = load(task / "_arena_eval.py", "_multreduce_poison_adapter")
    monkeypatch.setitem(sys.modules, "_arena_reference", reference)

    class TimedRun:
        def __init__(self):
            self.bound = False
            self.outputs = None
            self.after_sample = None
            self.replay = None

        def rerun(self):
            self.outputs = self.replay()
            return self.outputs

    timer = ModuleType("_aka_benchmark")
    timer.TimedRun = TimedRun
    skipping = [False]

    def samples(fn, *, warmup, repetition, prepare_fn, timed_run, **kwargs):
        assert warmup == 10 and repetition == 3 and prepare_fn is not None
        for index in range(repetition):
            prepare_fn()
            skipping[0] = skip_middle and index == 1
            result = fn()
            timed_run.after_sample(result)
        skipping[0] = False
        timed_run.outputs = result
        timed_run.bound = True
        timed_run.replay = lambda: (prepare_fn(), fn())[1]
        return [0.5] * repetition, {
            "benchmark_method": "cuda_graph", "benchmark_effective_repeats": 1,
        }

    timer.benchmark_cuda_graph_or_events_samples = samples
    monkeypatch.setitem(sys.modules, "_aka_benchmark", timer)

    def _measure_times(*args, **kwargs):
        raise AssertionError("task did not bind canonical timer")

    class Base:
        def __init__(self, *, op_callable, config):
            self.op_callable, self.config, self.prepare_fn = op_callable, config, None

        def run_benchmark(self, *, baseline_callable=None):
            assert baseline_callable is None
            values, metadata = globals()["_measure_times"](self.op_callable, self.config)
            return {"timing_ms": {"mean": sum(values) / len(values)}, **metadata}

    monkeypatch.setitem(Base.run_benchmark.__globals__, "_measure_times", _measure_times)

    def run_case():
        a = torch.tensor([[0.25, -0.5], [0.75, 0.125]])
        b = torch.tensor([[0.5, -0.25], [0.75, 0.5]])
        bias = torch.tensor([0.125, -0.25])
        c_buffer = torch.empty(2, 2)

        def op():
            if not skipping[0]:
                c_buffer.copy_(a @ b + bias[:, None])
            return c_buffer

        row = {"test_case_id": "scored_matmul"}
        plugin = SimpleNamespace(action="performance", current_row=row, exercised=set())
        checked = adapter.benchmark_type(Base, plugin, SimpleNamespace())(
            op_callable=op, config=SimpleNamespace(warm_up=10, repetition=3))
        checked.run_benchmark()
        return row

    if skip_middle:
        with pytest.raises(ValueError, match="nonfinite"):
            run_case()
    else:
        row = run_case()
        assert row["metadata"]["measured_samples_checked"] == 3
        assert row["execution_time_ms"] == 0.5
