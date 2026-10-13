"""CPU contract tests for task-owned measured-output adapters."""
from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "triton2triton/rocmbench/easy/test_kernel_dot",
    "triton2triton/rocmbench/easy/test_reverse_range",
    "triton2triton/rocmbench/medium/rmsnorm_fwd",
    "triton2triton/rocmbench/hard/test_block_pointer_matmul",
    "triton2triton/rocmbench/hard/triton_multreduce_matmul_kernel",
)


class FakeTimedRun:
    def __init__(self):
        self.outputs = None
        self.after_sample = None
        self.bound = False
        self.replay_output = None

    def rerun(self):
        self.outputs = self.replay_output() if callable(self.replay_output) else self.replay_output
        return self.outputs


def _measure_times(*args, **kwargs):
    raise AssertionError("The adapter must bind the canonical measured-output collector")


class FakeBase:
    def __init__(self, *, op_callable, config, **kwargs):
        self.op_callable = op_callable
        self.config = config
        self.prepare_fn = None

    def run_benchmark(self, *, baseline_callable=None):
        assert baseline_callable is None
        times, metadata = _measure_times(
            self.op_callable, self.config, prepare_fn=self.prepare_fn)
        return {"timing_ms": {"mean": sum(times) / len(times)}, **metadata}


def test_block_pointer_graph_prepares_one_full_output_per_sample(monkeypatch):
    """A cached-C shortcut cannot satisfy a prepared, one-call graph sample."""
    task = ROOT / "tasks/triton2triton/rocmbench/hard/test_block_pointer_matmul"
    spec = importlib.util.spec_from_file_location("_block_pointer_eval", task / "_arena_eval.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    spec = importlib.util.spec_from_file_location("_arena_reference", task / "_arena_reference.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    monkeypatch.setitem(sys.modules, "_arena_reference", reference)

    class PreparedBase(FakeBase):
        def run_benchmark(self, *, baseline_callable=None):
            assert baseline_callable is None
            times, meta = _measure_times(self.op_callable, self.config, prepare_fn=self.prepare_fn)
            return {"timing_ms": {"mean": sum(times) / len(times)}, **meta}

    timer = ModuleType("_aka_benchmark")
    timer.TimedRun = FakeTimedRun
    saw_prepared = []
    def samples(fn, *, prepare_fn, repetition, timed_run, **_kwargs):
        assert prepare_fn is not None
        for _ in range(repetition):
            prepare_fn()
            saw_prepared.append(torch.isnan(c).all().item())
            output = fn()
            timed_run.after_sample(output)
        timed_run.outputs = output
        timed_run.replay_output = lambda: (prepare_fn(), fn())[1]
        timed_run.bound = True
        return [1.0] * repetition, {"benchmark_method": "cuda_graph"}
    timer.benchmark_cuda_graph_or_events_samples = samples
    monkeypatch.setitem(sys.modules, "_aka_benchmark", timer)

    a = torch.tensor([[1., 2.], [3., 4.]])
    b = torch.tensor([[2., 0.], [1., 2.]])
    c = torch.empty_like(a)
    op = lambda: c.copy_(a @ b)
    row = {"test_case_id": "prepared"}
    plugin = SimpleNamespace(action="performance", current_row=row, exercised=set())
    checked = adapter.benchmark_type(PreparedBase, plugin, SimpleNamespace())(
        op_callable=op, config=SimpleNamespace(warm_up=10, repetition=3))
    checked.run_benchmark()
    assert saw_prepared == [True] * 3
    assert row["metadata"]["measured_samples_checked"] == 3
    assert checked.prepare_fn is None




@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("method", ("cuda_graph", "cuda_event_fallback"))
@pytest.mark.parametrize("bad_sample,bad_replay", ((False, False), (True, False), (False, True)))
def test_adapter_checks_measured_samples_and_bound_replay(monkeypatch, task, method, bad_sample, bad_replay):
    expected = object()
    bad = object()
    calls = []
    is_rms = task.endswith("/rmsnorm_fwd")

    reference = ModuleType("_arena_reference")
    def prepare(context, module):
        def check(value, *statistics):
            calls.append(value)
            if value is not expected:
                raise AssertionError("measured output differs from reference")
        return check
    reference.prepare = prepare
    reference.prepare_full = prepare
    reference.check_scale_stride_control = lambda module, device: None
    reference.check_width_tail_controls = lambda module, device: None
    reference.check_stats_stride_control = lambda context, module: None
    reference.check_row_stride_control = lambda context, module: None
    reference.check_blocked_stride_control = lambda context, module: None
    reference.check_stride_controls = lambda context, module: None
    reference.check_partial_tile_controls = lambda context, module: None
    reference.check_interior_transitions = lambda context, invoke: None
    reference.check_declared_m_controls = lambda module, device: None
    reference.poison_outputs = lambda context, result, *statistics: calls.append("poisoned")
    reference.snapshot_inputs = lambda context: {}
    reference.assert_inputs_unchanged = lambda context, saved: None
    reference.check_inputs = lambda context, saved: None
    reference.restore_inputs = lambda context, saved: None
    @contextmanager
    def observe_kernel_side_outputs(module):
        yield SimpleNamespace(latest=(expected, expected))
    reference.observe_kernel_side_outputs = observe_kernel_side_outputs
    @contextmanager
    def capture_side_outputs(module):
        def lookup(output):return ()
        lookup.poison_all=lambda:None
        yield lookup
    reference.capture_side_outputs=capture_side_outputs
    @contextmanager
    def perturbed_inputs(context):
        calls.append("perturbed")
        yield
    reference.perturbed_inputs = perturbed_inputs
    if is_rms:
        path = ROOT / "tasks" / task / "_arena_reference.py"
        spec = importlib.util.spec_from_file_location("_rms_real_reference", path)
        real_reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(real_reference)
        reference.prepare = real_reference.prepare
        reference.snapshot_inputs = real_reference.snapshot_inputs
        reference.assert_inputs_unchanged = real_reference.assert_inputs_unchanged
        reference.perturbed_inputs = real_reference.perturbed_inputs
        real_poison = real_reference.poison_outputs
        def poison(context, result, *statistics):
            calls.append("poisoned")
            return real_poison(context, result)
        reference.poison_outputs = poison
    monkeypatch.setitem(sys.modules, "_arena_reference", reference)

    if is_rms:
        x = torch.tensor([[1., -2., 3., -4.]])
        g = torch.tensor([0.5, -0.25, 0.75, 1.0])
        y_buffer = torch.empty_like(x)
        rsigma_buffer = torch.empty(1)
        context = {"x": x, "g": g, "y_buffer": y_buffer,
                   "rsigma_buffer": rsigma_buffer, "ZERO_CENTERED_GAMMA": False,
                   "out_dtype_str": "fp32", "eps": 1e-5}
        def rms_op():
            xf = x.float()
            rsigma = torch.rsqrt(xf.square().mean(-1) + context["eps"])
            y_buffer.copy_(xf * rsigma[:, None] * g)
            rsigma_buffer.copy_(rsigma)
            return y_buffer
        op_callable = rms_op
    else:
        op_callable = lambda: expected

    timer = ModuleType("_aka_benchmark")
    timer.TimedRun = FakeTimedRun
    def samples(fn, *, warmup, repetition, timed_run, prepare_fn=None, **kwargs):
        assert warmup == 10 and repetition == 3
        if is_rms:
            assert kwargs["max_graph_repeats"] == 1
        for index in range(repetition):
            if prepare_fn is not None:prepare_fn()
            output = bad if bad_sample and index == 1 else fn()
            timed_run.after_sample(output)
        timed_run.outputs = output
        timed_run.replay_output = bad if bad_replay else (fn if is_rms else expected)
        timed_run.bound = True
        return [1.0] * repetition, {
            "benchmark_method": method,
            "benchmark_effective_repeats": 1,
            "benchmark_samples": repetition,
            "benchmark_timed_run_kind": "captured_graph" if method == "cuda_graph" else "eager_callable",
            "benchmark_fallback_reason": "declared event timing" if method == "cuda_event_fallback" else None,
        }
    timer.benchmark_cuda_graph_or_events_samples = samples
    monkeypatch.setitem(sys.modules, "_aka_benchmark", timer)

    path = ROOT / "tasks" / task / "_arena_eval.py"
    spec = importlib.util.spec_from_file_location("_warn_rocmbench_" + task.replace("/", "_"), path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    row = {"test_case_id": "sample"}
    plugin = SimpleNamespace(action="performance", current_row=row, exercised=set())
    checked = adapter.benchmark_type(FakeBase, plugin, SimpleNamespace())(
        op_callable=op_callable, config=SimpleNamespace(warm_up=10, repetition=3))
    if is_rms:
        checked.context = context
    if bad_sample or bad_replay:
        error = TypeError if is_rms else AssertionError
        message = "declared tensor output" if is_rms else "measured output differs"
        with pytest.raises(error, match=message):
            checked.run_benchmark()
        assert "execution_time_ms" not in row
    else:
        checked.run_benchmark()
        assert row["metadata"]["measured_samples_checked"] == 3
        assert row["metadata"]["bound_replay_output_checked"]
        assert row["metadata"]["perturbed_input_replay_checked"]
        assert row["metadata"]["device_timing"]["benchmark_method"] == method
        assert row["metadata"]["device_timing"]["benchmark_timed_run_kind"] == (
            "captured_graph" if method == "cuda_graph" else "eager_callable")
        if method == "cuda_event_fallback":
            assert row["metadata"]["device_timing"]["benchmark_fallback_reason"] == "declared event timing"
    if is_rms:
        assert "poisoned" in calls
    elif task in (
        'triton2triton/rocmbench/easy/test_reverse_range',
        'triton2triton/rocmbench/hard/test_block_pointer_matmul',
        'triton2triton/rocmbench/hard/triton_multreduce_matmul_kernel',
    ):
        assert "poisoned" in calls  # sample preparation invalidates the scored buffer
    else:
        assert ("poisoned" in calls) == (not bad_sample)
    assert _measure_times is FakeBase.run_benchmark.__globals__["_measure_times"]


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("cached_input", (False, True))
def test_adapter_replays_changed_operand_and_restores_it(monkeypatch, task, cached_input):
    score_input = [1]
    reference = ModuleType("_arena_reference")
    is_rms = task.endswith("/rmsnorm_fwd")
    def prepare(context, module):
        expected = context["score_input"][0]
        def check(output, *statistics):
            if output != expected:
                raise AssertionError("bound replay used a cached operand or output")
        return check
    @contextmanager
    def perturbed_inputs(context):
        original = context["score_input"][0]
        context["score_input"][0] = 2
        try:
            yield
        finally:
            context["score_input"][0] = original
    reference.prepare = prepare
    reference.prepare_full = prepare
    reference.check_scale_stride_control = lambda module, device: None
    reference.check_width_tail_controls = lambda module, device: None
    reference.check_stats_stride_control = lambda context, module: None
    reference.check_row_stride_control = lambda context, module: None
    reference.check_blocked_stride_control = lambda context, module: None
    reference.check_stride_controls = lambda context, module: None
    reference.check_declared_m_controls = lambda module, device: None
    reference.check_partial_tile_controls = lambda context, module: None
    reference.check_interior_transitions = lambda context, invoke: None
    reference.perturbed_inputs = perturbed_inputs
    reference.poison_outputs = lambda context, output, *statistics: None
    reference.snapshot_inputs = lambda context: {}
    reference.assert_inputs_unchanged = lambda context, saved: None
    reference.check_inputs = lambda context, saved: None
    reference.restore_inputs = lambda context, saved: None
    @contextmanager
    def observe_kernel_side_outputs(module):
        yield SimpleNamespace(latest=(score_input[0], score_input[0]))
    reference.observe_kernel_side_outputs = observe_kernel_side_outputs
    @contextmanager
    def capture_side_outputs(module):
        def lookup(output):return ()
        lookup.poison_all=lambda:None
        yield lookup
    reference.capture_side_outputs=capture_side_outputs
    if is_rms:
        path = ROOT / "tasks" / task / "_arena_reference.py"
        spec = importlib.util.spec_from_file_location("_rms_readonly_reference", path)
        real_reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(real_reference)
        reference.snapshot_inputs = real_reference.snapshot_inputs
        reference.assert_inputs_unchanged = real_reference.assert_inputs_unchanged
    monkeypatch.setitem(sys.modules, "_arena_reference", reference)

    timer = ModuleType("_aka_benchmark")
    timer.TimedRun = FakeTimedRun
    def samples(fn, *, repetition, timed_run, prepare_fn=None, **kwargs):
        for _ in range(repetition):
            if prepare_fn is not None:prepare_fn()
            output = fn()
            timed_run.after_sample(output)
        timed_run.outputs = output
        timed_run.replay_output = lambda: (prepare_fn(),fn())[1] if prepare_fn is not None else fn()
        timed_run.bound = True
        return [1.0] * repetition, {"benchmark_method": "cuda_graph",
                                   "benchmark_effective_repeats": 1,
                                   "benchmark_samples": repetition}
    timer.benchmark_cuda_graph_or_events_samples = samples
    monkeypatch.setitem(sys.modules, "_aka_benchmark", timer)

    path = ROOT / "tasks" / task / "_arena_eval.py"
    spec = importlib.util.spec_from_file_location("_warn_changed_" + task.replace("/", "_"), path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    row = {"test_case_id": "changed"}
    plugin = SimpleNamespace(action="performance", current_row=row, exercised=set())
    op = lambda: 1 if cached_input else score_input[0]
    checked = adapter.benchmark_type(FakeBase, plugin, SimpleNamespace())(
        op_callable=op, config=SimpleNamespace(warm_up=10, repetition=3))
    if is_rms:
        checked.context = {"score_input": score_input, "x": torch.tensor([[1., -2.]]),
                           "g": torch.tensor([0.5, 1.25])}
    if cached_input:
        with pytest.raises(AssertionError, match="cached operand or output"):
            checked.run_benchmark()
        assert "execution_time_ms" not in row
    else:
        checked.run_benchmark()
        assert row["metadata"]["perturbed_input_replay_checked"]
    assert score_input == [1]
    assert _measure_times is FakeBase.run_benchmark.__globals__["_measure_times"]


@pytest.mark.parametrize("operand", ("x", "g"))
def test_rms_readonly_snapshot_rejects_each_changed_operand_byte(operand):
    task = ROOT / "tasks/triton2triton/rocmbench/medium/rmsnorm_fwd"
    spec = importlib.util.spec_from_file_location("_rms_readonly_bytes", task / "_arena_reference.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    context = {"x": torch.tensor([[1.0, -2.0]]), "g": torch.tensor([0.5, 1.25])}
    saved = reference.snapshot_inputs(context)
    reference.assert_inputs_unchanged(context, saved)
    context[operand].add_(1)
    with pytest.raises(AssertionError, match=f"read-only {operand} input"):
        reference.assert_inputs_unchanged(context, saved)


@pytest.mark.parametrize("task", TASKS)
def test_task_perturbation_changes_sources_and_restores_on_failure(task):
    path = ROOT / "tasks" / task / "_arena_reference.py"
    spec = importlib.util.spec_from_file_location("_warn_reference_" + task.replace("/", "_"), path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    if task.endswith("/gemm"):
        context = {"a": matrix.clone(), "b": matrix.clone(),
                   "a_fp32_ref": matrix.clone(), "b_fp32_ref": matrix.clone()}
    elif task.endswith("/layernorm"):
        context = {"x": matrix.clone(), "w": torch.tensor([0.5, 1.0]),
                   "b": torch.tensor([0.25, 0.75])}
    elif task.endswith("/test_flashattention_fwd"):
        context = {key: matrix.clone() for key in ("q", "k", "v")}
    elif task.endswith("/test_kernel_dot"):
        context = {"Z_initial": matrix.clone()}
    elif task.endswith("/test_reverse_range"):
        context = {"data_perf": torch.arange(1.0, 517.0)}
    elif task.endswith("/rmsnorm_fwd"):
        context = {"x": matrix.clone(), "g": torch.tensor([0.5, 1.0])}
    elif task.endswith("/test_block_pointer_matmul"):
        context = {"a": matrix.clone(), "b": matrix.clone()}
    else:
        context = {"a": matrix.clone(), "b": matrix.clone(),
                   "bias": torch.tensor([0.25, 0.75])}
    original = {key: value.clone() for key, value in context.items()}
    with pytest.raises(RuntimeError, match="forced exit"):
        with reference.perturbed_inputs(context):
            assert all(not torch.equal(context[key], value) for key, value in original.items())
            raise RuntimeError("forced exit")
    for key, value in original.items():
        torch.testing.assert_close(context[key], value)


def _replay_case(task):
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    if task.endswith("/gemm"):
        return ({"a": matrix.clone(), "b": matrix.clone(),
                 "a_fp32_ref": matrix.clone(), "b_fp32_ref": matrix.clone(),
                 "a_scale": None, "b_scale": None, "current_scale_a8_b8": None,
                 "c": torch.empty_like(matrix)}, ("a", "b"))
    if task.endswith("/layernorm"):
        return ({"x": matrix.clone(), "w": torch.tensor([0.5, 1.0]),
                 "b": torch.tensor([0.25, 0.75]), "normalized_shape_arg": (2,),
                 "eps": 1e-5}, ("x", "w", "b"))
    if task.endswith("/test_flashattention_fwd"):
        torch.manual_seed(41)
        shape = (1, 1, 128, 64)
        return ({"q": torch.empty(shape).normal_(0.1, 0.2),
                 "k": torch.empty(shape).normal_(0.4, 0.2),
                 "v": torch.empty(shape).normal_(0.3, 0.2),
                 "sm_scale": 0.2}, ("q", "k", "v"))
    if task.endswith("/test_kernel_dot"):
        return ({"Z_initial": matrix.clone(), "Z_tensor": torch.empty_like(matrix)},
                ("Z_initial",))
    if task.endswith("/test_reverse_range"):
        return ({"data_perf": torch.arange(1.0, 517.0),
                 "res_perf_buffer": torch.empty(512)}, ("data_perf",))
    if task.endswith("/rmsnorm_fwd"):
        return ({"x": matrix.clone(), "g": torch.tensor([0.5, 1.0]),
                 "eps": 1e-5, "ZERO_CENTERED_GAMMA": False, "out_dtype_str": "fp32",
                 "y_buffer": torch.empty_like(matrix), "rsigma_buffer": torch.empty(2)},
                ("x", "g"))
    if task.endswith("/test_block_pointer_matmul"):
        return ({"a": matrix.clone(), "b": matrix.clone(),
                 "c": torch.empty_like(matrix)}, ("a", "b"))
    return ({"a": matrix.clone(), "b": matrix.clone(),
             "bias": torch.tensor([0.25, 0.75]), "c_buffer": torch.empty_like(matrix)},
            ("a", "b", "bias"))


def _render_candidate(task, context):
    c = context
    if task.endswith("/gemm") or task.endswith("/test_block_pointer_matmul"):
        return c["a"] @ c["b"]
    if task.endswith("/layernorm"):
        return torch.nn.functional.layer_norm(c["x"], c["normalized_shape_arg"],
                                              c["w"], c["b"], c["eps"])
    if task.endswith("/test_flashattention_fwd"):
        q, k, v = c["q"], c["k"], c["v"]
        n = q.shape[-2]
        scores = (q @ k.transpose(-1, -2)) * c["sm_scale"]
        scores.masked_fill_(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
        return torch.softmax(scores.float(), dim=-1).to(q.dtype) @ v
    if task.endswith("/test_kernel_dot"):
        return c["Z_initial"] @ c["Z_initial"]
    if task.endswith("/test_reverse_range"):
        return torch.flip(c["data_perf"][1:513], [0])
    if task.endswith("/rmsnorm_fwd"):
        x, gamma = c["x"].float(), c["g"].float()
        rsigma = torch.rsqrt((x * x).mean(-1) + c["eps"])
        c["y_buffer"].copy_(x * rsigma[:, None] * gamma)
        c["rsigma_buffer"].copy_(rsigma)
        return c["y_buffer"]
    output = c["a"] @ c["b"]
    if c["bias"] is not None:
        output = output + c["bias"][:, None]
    return output


def _set_scored_output(task, context, output):
    if task.endswith("/gemm") or task.endswith("/test_block_pointer_matmul"):
        context["c"].copy_(output)
    elif task.endswith("/test_kernel_dot"):
        context["Z_tensor"].copy_(output)
    elif task.endswith("/test_reverse_range"):
        context["res_perf_buffer"].copy_(output)
    elif task.endswith("/rmsnorm_fwd"):
        assert output is context["y_buffer"]
    elif task.endswith("/triton_multreduce_matmul_kernel"):
        context["c_buffer"].copy_(output)


@pytest.mark.parametrize("task", TASKS)
def test_changed_reference_rejects_each_independently_cached_operand(task):
    path = ROOT / "tasks" / task / "_arena_reference.py"
    spec = importlib.util.spec_from_file_location("_warn_partial_" + task.replace("/", "_"), path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    context, operands = _replay_case(task)
    pristine = {key: context[key].clone() for key in operands}
    for stale in operands:
        with reference.perturbed_inputs(context):
            changed_check = reference.prepare(context, None)
            hybrid = dict(context)
            hybrid[stale] = pristine[stale]
            stale_output = _render_candidate(task, hybrid)
            _set_scored_output(task, context, stale_output)
            with pytest.raises((AssertionError, ValueError)):
                if task.endswith('/layernorm'):
                    xf=hybrid['x'].float()
                    mean=xf.mean(-1)
                    rstd=torch.rsqrt(((xf-mean[:,None])**2).mean(-1)+hybrid['eps'])
                    changed_check(stale_output,(mean,rstd))
                else:
                    changed_check(stale_output)
