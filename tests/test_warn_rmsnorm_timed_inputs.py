"""CPU controls for the RMSNorm task's direct and scored-output guards."""

import ast
import importlib.util
import inspect
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / "tasks/triton2triton/rocmbench/medium/rmsnorm_fwd"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CpuTorch:
    def __getattr__(self, name):
        value = getattr(torch, name)
        if name in ("randn", "ones", "zeros", "zeros_like", "empty", "empty_like"):
            def cpu(*args, **kwargs):
                kwargs["device"] = "cpu"
                return value(*args, **kwargs)
            return cpu
        return value


@pytest.mark.parametrize("operand", ("x", "g"))
def test_direct_reference_precedes_editable_kernel_and_rejects_input_mutation(operand):
    source = ast.parse((TASK / "rmsnorm_fwd.py").read_text())
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                    and node.name == "test_rmsnorm")
    function.decorator_list = []
    calls = []

    def reference(x, g, zero, output_dtype=torch.float16, epsilon=1e-6):
        calls.append("reference")
        xf = x.float()
        rsigma = torch.rsqrt(xf.square().mean(-1) + epsilon)
        y = (xf * rsigma[:, None] * (g.float() + int(zero))).to(output_dtype)
        return y, rsigma

    def kernel(x, g, y, rsigma, *_args):
        calls.append("kernel")
        with torch.no_grad():
            (x if operand == "x" else g).add_(1)
            expected_y, expected_rs = reference(x, g, True, y.dtype)
            y.copy_(expected_y)
            rsigma.copy_(expected_rs)
        return y

    namespace = {
        "torch": CpuTorch(), "pytest": pytest,
        "arg_to_torch_dtype": {"fp16": torch.float16, "bf16": torch.bfloat16},
        "set_seed": lambda: torch.manual_seed(42),
        "get_num_sms": lambda: 2,
        "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
        "rmsnorm": kernel, "torch_rmsnorm_fwd": reference, "result_gold": {},
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(TASK / "rmsnorm_fwd.py"), "exec"), namespace)
    with pytest.raises(AssertionError, match="read-only .* input"):
        namespace["test_rmsnorm"](2, 10, True, "fp16", "fp16", SimpleNamespace(node=SimpleNamespace(name="direct")))
    assert calls[:2] == ["reference", "kernel"]


class FakeTimedRun:
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


def _measure_times(*_args, **_kwargs):
    raise AssertionError("The task must bind the canonical sampled timer")


class FakeBase:
    def __init__(self, op_callable, config):
        self.op_callable = op_callable
        self.config = config
        self.prepare_fn = None
        self.use_cuda_graph = True

    def run_benchmark(self, **kwargs):
        assert kwargs["baseline_callable"] is None
        times, metadata = _measure_times(self.op_callable, self.config,
                                         use_cuda_graph=self.use_cuda_graph)
        return {"timing_ms": {"mean": sum(times) / len(times)}, **metadata}


@pytest.mark.parametrize("method", ("cuda_graph", "cuda_event_fallback"))
@pytest.mark.parametrize("fault", (None, "skip_middle", "wrong_middle", "mutate_input_last"))
def test_reported_sample_checks_one_real_call_and_both_buffers(monkeypatch, method, fault):
    reference = load("rms_reference_under_test", TASK / "_arena_reference.py")
    adapter = load("rms_adapter_under_test", TASK / "_arena_eval.py")
    sample = [0]
    observed = []

    def fake_timer(fn, *, warmup, repetition, max_graph_repeats, prepare_fn,
                   timed_run, use_cuda_graph, **_kwargs):
        assert (warmup, repetition, max_graph_repeats) == (10, 100, 1)
        assert prepare_fn is not None
        assert use_cuda_graph is (method == "cuda_graph")
        # The real helper also prepares outside capture, calibration and prime.
        # This fixture exercises every reported sample and the bound replay.
        for index in range(repetition):
            prepare_fn()
            assert torch.isnan(y_buffer).all() and torch.isnan(rsigma_buffer).all()
            sample[0] = index + 1
            if fault == "skip_middle" and index == 50:
                output = y_buffer
            else:
                output = fn()
                if fault == "wrong_middle" and index == 50:
                    rsigma_buffer.add_(1)
                if fault == "mutate_input_last" and index == repetition - 1:
                    x.add_(1)
            timed_run.outputs = output
            timed_run.after_sample(output)
            observed.append(index)

        def replay():
            prepare_fn()
            return fn()

        timed_run._rerun = replay
        timed_run.outputs = y_buffer
        return [0.1] * repetition, {
            "benchmark_method": method,
            "benchmark_timed_run_kind": "captured_graph" if method == "cuda_graph" else "eager_callable",
            "benchmark_samples": repetition,
            "benchmark_effective_repeats": 1,
            "benchmark_max_repeats": max_graph_repeats,
            "benchmark_warmup": warmup,
        }

    monkeypatch.setitem(sys.modules, "_arena_reference", reference)
    monkeypatch.setitem(sys.modules, "_aka_benchmark", SimpleNamespace(
        TimedRun=FakeTimedRun, benchmark_cuda_graph_or_events_samples=fake_timer))
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    g = torch.tensor([0.4, 0.5, -0.2, 1.0])
    y_buffer = torch.empty_like(x)
    rsigma_buffer = torch.empty(1)
    ZERO_CENTERED_GAMMA = False
    out_dtype_str = "fp32"
    eps = 1e-5
    calls = [0]

    def run():
        calls[0] += 1
        xf = x.float()
        rsigma = torch.rsqrt(xf.square().mean(-1) + eps)
        y_buffer.copy_(xf * rsigma[:, None] * g)
        rsigma_buffer.copy_(rsigma)
        return y_buffer

    plugin = SimpleNamespace(current_row={"test_case_id": "scored"}, action="performance", exercised=set())
    Config = SimpleNamespace(warm_up=10, repetition=100)

    # CheckedBenchmark takes the caller's original performance locals as its
    # context. Keep those exact tensor names in this scope.
    def make_benchmark():
        return adapter.benchmark_type(FakeBase, plugin, None)(run, Config)

    # The caller frame must contain the actual original inputs, not just an
    # unrelated wrapper closure, because this is what the task adapter reads.
    def execute():
        x_local = x
        g_local = g
        y_local = y_buffer
        rs_local = rsigma_buffer
        context = {"x": x_local, "g": g_local, "y_buffer": y_local,
                   "rsigma_buffer": rs_local, "ZERO_CENTERED_GAMMA": ZERO_CENTERED_GAMMA,
                   "out_dtype_str": out_dtype_str, "eps": eps}
        benchmark = make_benchmark()
        benchmark.context = context
        benchmark.use_cuda_graph = method == "cuda_graph"
        return benchmark.run_benchmark()

    if fault is None:
        record = execute()
        assert len(observed) == 100
        assert calls[0] == 102  # initial check + 100 scored + bound changed replay
        assert record["benchmark_effective_repeats"] == 1
        assert plugin.current_row["metadata"]["measured_samples_checked"] == 100
        assert plugin.exercised == {"scored"}
    elif fault == "mutate_input_last":
        with pytest.raises(AssertionError, match="read-only x input"):
            execute()
        assert len(observed) == 100
    else:
        expected = ValueError if fault == "skip_middle" else reference.NumericalMismatch
        with pytest.raises(expected):
            execute()
        assert len(observed) == 50  # the first poisoned/wrong middle result failed
