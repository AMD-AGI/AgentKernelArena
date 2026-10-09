"""Side-output and blocked-stride controls for scored ROCmBench paths."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1] / "tasks"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _measure_times(*args, **kwargs):
    raise AssertionError("The adapter did not bind the measured-output collector")


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("missing_statistics", (False, True))
@pytest.mark.parametrize("mutated_operand", (None, "x", "w", "b"))
def test_actual_layernorm_adapter_checks_each_measured_side_output(
    monkeypatch, dtype, missing_statistics, mutated_operand,
):
    task = ROOT / "instruction2triton/rocmbench/layernorm"
    reference = load(task / "_arena_reference.py", "_arena_reference")
    adapter = load(task / "_arena_eval.py", "_layernorm_adapter_control")
    monkeypatch.setitem(sys.modules, "_arena_reference", reference)

    class TimedRun:
        def __init__(self):
            self.outputs = None
            self.after_sample = None
            self.bound = False
            self.replay = None

        def rerun(self):
            self.outputs = self.replay()
            return self.outputs

    timer = ModuleType("_aka_benchmark")
    timer.TimedRun = TimedRun
    observed = []
    input_refs = {}
    originals = {}

    def samples(fn, *, prepare_fn, warmup, repetition, timed_run, **kwargs):
        assert prepare_fn is not None and warmup == 10 and repetition == 3
        for index in range(repetition):
            prepare_fn()
            output = fn()
            observed.append(output)
            if index == 1 and mutated_operand is not None:
                input_refs[mutated_operand].add_(1)
            timed_run.after_sample(output)
        timed_run.outputs = output
        timed_run.bound = True
        timed_run.replay = lambda: (prepare_fn(), fn())[1]
        return [1.0] * repetition, {
            "benchmark_method": "cuda_graph", "benchmark_timed_run_kind": "captured_graph",
            "benchmark_effective_repeats": 1,
        }

    timer.benchmark_cuda_graph_or_events_samples = samples
    monkeypatch.setitem(sys.modules, "_aka_benchmark", timer)

    class Base:
        def __init__(self, *, op_callable, config):
            self.op_callable, self.config, self.prepare_fn = op_callable, config, None

        def run_benchmark(self, *, baseline_callable=None):
            assert baseline_callable is None
            values, metadata = _measure_times(self.op_callable, self.config)
            return {"timing_ms": {"mean": sum(values) / len(values)}, **metadata}

    calls = [0]
    module = SimpleNamespace()

    def wrapper(grid, x, y, w, b, mean, rstd, *rest):
        calls[0] += 1
        xf = x.float()
        expected_mean = xf.mean(-1)
        expected_rstd = torch.rsqrt(((xf - expected_mean[:, None]) ** 2).mean(-1) + 1e-5)
        y.copy_(torch.nn.functional.layer_norm(x, (x.shape[1],), w, b, 1e-5))
        if not missing_statistics or calls[0] == 1:
            mean.copy_(expected_mean)
            rstd.copy_(expected_rstd)

    module.layernorm_wrapper_fn = wrapper
    original_wrapper = wrapper
    row = {"test_case_id": "scored_layernorm"}
    plugin = SimpleNamespace(action="performance", current_row=row, exercised=set())

    def run_case():
        x = torch.randn((3, 129), dtype=dtype)
        w = torch.rand(129, dtype=dtype)
        b = torch.rand(129, dtype=dtype)
        input_refs.update(x=x, w=w, b=b)
        originals.update({name: value.clone() for name, value in input_refs.items()})
        normalized_shape_arg = (129,)
        eps = 1e-5
        y = torch.empty_like(x)
        mean = torch.empty(3, dtype=torch.float32)
        rstd = torch.empty_like(mean)

        def op():
            module.layernorm_wrapper_fn((3,), x, y, w, b, mean, rstd)
            return y

        checked = adapter.benchmark_type(Base, plugin, module)(
            op_callable=op, config=SimpleNamespace(warm_up=10, repetition=3))
        checked.run_benchmark()
        return x, w, b

    if missing_statistics or mutated_operand is not None:
        with pytest.raises((AssertionError, ValueError)):
            run_case()
        assert "execution_time_ms" not in row
    else:
        x, w, b = run_case()
        assert row["metadata"]["measured_samples_checked"] == 3
        assert row["metadata"]["bound_replay_output_checked"]
        assert torch.isfinite(x).all() and torch.isfinite(w).all() and torch.isfinite(b).all()
        assert len(observed) == 3
    assert module.layernorm_wrapper_fn is original_wrapper
    for name, value in input_refs.items():
        torch.testing.assert_close(value, originals[name], rtol=0, atol=0)
    assert Base.run_benchmark.__globals__["_measure_times"] is _measure_times


@pytest.mark.parametrize("fault", (None, "ignore_input_stride", "ignore_output_stride", "skip_rsigma"))
def test_blocked_rmsnorm_full_width_checks_independent_row_strides(fault):
    task = ROOT / "triton2triton/rocmbench/medium/rmsnorm_fwd"
    reference = load(task / "_arena_reference.py", "_blocked_rms_reference")
    cols = 65536
    x = torch.randn((2, cols), dtype=torch.float16) * .1
    g = torch.rand(cols, dtype=torch.float16)
    context = {
        "x": x, "g": g, "y_buffer": torch.empty_like(x),
        "USE_BLOCKED_fwd": True, "blk_size_fwd": 32768,
        "NUM_PRGMS_fwd": 2, "ZERO_CENTERED_GAMMA": False, "eps": 1e-5,
    }
    seen = []

    def operator(x, g, y, rsigma, *args):
        rows, width, centered, block_size, blocked, programs, eps = args[-7:]
        seen.append((rows, width, block_size, blocked, x.stride(0), y.stride(0)))
        assert blocked is True and width == cols and rows == 3
        xf = x.float()
        if fault == "ignore_input_stride":
            xf = torch.as_strided(x, x.shape, (cols, 1)).float()
        expected_rs = torch.rsqrt((xf * xf).mean(-1) + eps)
        expected_y = (xf * expected_rs[:, None] * g.float()).to(y.dtype)
        if fault == "ignore_output_stride":
            torch.as_strided(y, y.shape, (cols, 1)).copy_(expected_y)
        else:
            y.copy_(expected_y)
        if fault != "skip_rsigma":
            rsigma.copy_(expected_rs)

    module = SimpleNamespace(rmsnorm=operator)
    if fault is None:
        reference.check_blocked_stride_control(context, module)
        assert len(seen) == 1
        assert seen[0][4] == cols + 16 and seen[0][5] == cols + 32
    else:
        with pytest.raises((AssertionError, ValueError)):
            reference.check_blocked_stride_control(context, module)
    context["USE_BLOCKED_fwd"] = False
    seen.clear()
    reference.check_blocked_stride_control(context, module)
    assert seen == []
