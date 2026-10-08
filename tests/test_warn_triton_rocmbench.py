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
    "instruction2triton/rocmbench/gemm",
    "instruction2triton/rocmbench/layernorm",
    "instruction2triton/rocmbench/test_flashattention_fwd",
    "instruction2triton/rocmbench/test_kernel_dot",
    "instruction2triton/rocmbench/test_reverse_range",
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
        times, metadata = _measure_times(self.op_callable, self.config)
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


@pytest.mark.parametrize("missing_side", (None, "L", "m"))
def test_flash_observer_binds_real_kernel_side_buffers(missing_side):
    task = ROOT / "tasks/instruction2triton/rocmbench/test_flashattention_fwd"
    spec = importlib.util.spec_from_file_location("_flash_side_reference", task / "_arena_reference.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.manual_seed(17)
    q = torch.randn(1, 1, 4, 16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = .35

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(q_arg, k_arg, v_arg, scale_arg, L, m, output):
                scores = (q_arg.float() @ k_arg.float().transpose(-1, -2)) * scale_arg
                scores.masked_fill_(torch.ones(4, 4, dtype=torch.bool).triu_(1), float("-inf"))
                row_max = scores.max(dim=-1).values
                if missing_side != "m":
                    m.copy_(row_max.reshape(1, 4))
                if missing_side != "L":
                    L.copy_(torch.exp(scores - row_max[..., None]).sum(-1).reshape(1, 4))
                output.copy_(torch.softmax(scores, -1).to(q_arg.dtype) @ v_arg)
            return launch

    kernel = FakeKernel()
    module = SimpleNamespace(flash_fwd_kernel=kernel)
    context = {"q": q, "k": k, "v": v, "sm_scale": scale}
    check = reference.prepare_full(context, module)
    output = torch.empty_like(q)
    L = torch.full((1, 4), float("nan"))
    m = torch.full_like(L, float("nan"))
    with reference.observe_kernel_side_outputs(module) as observed:
        module.flash_fwd_kernel[(1,)](q, k, v, scale, L, m, output)
        assert observed.latest[0] is L and observed.latest[1] is m
        if missing_side is None:
            check(output, observed.latest)
        else:
            with pytest.raises((AssertionError, ValueError)):
                check(output, observed.latest)
    assert module.flash_fwd_kernel is kernel


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("method", ("cuda_graph", "cuda_event_fallback"))
@pytest.mark.parametrize("bad_sample,bad_replay", ((False, False), (True, False), (False, True)))
def test_adapter_checks_measured_samples_and_bound_replay(monkeypatch, task, method, bad_sample, bad_replay):
    expected = object()
    bad = object()
    calls = []

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
    reference.poison_outputs = lambda context, result, *statistics: calls.append("poisoned")
    reference.snapshot_inputs = lambda context: {}
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
    monkeypatch.setitem(sys.modules, "_arena_reference", reference)

    timer = ModuleType("_aka_benchmark")
    timer.TimedRun = FakeTimedRun
    def samples(fn, *, warmup, repetition, timed_run, prepare_fn=None, **kwargs):
        assert warmup == 10 and repetition == 3
        for index in range(repetition):
            if prepare_fn is not None:prepare_fn()
            output = bad if bad_sample and index == 1 else fn()
            timed_run.after_sample(output)
        timed_run.outputs = output
        timed_run.replay_output = bad if bad_replay else expected
        timed_run.bound = True
        return [1.0] * repetition, {
            "benchmark_method": method,
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
        op_callable=lambda: expected, config=SimpleNamespace(warm_up=10, repetition=3))
    if bad_sample or bad_replay:
        with pytest.raises(AssertionError, match="measured output differs"):
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
    assert ("poisoned" in calls) == (not bad_sample)
    assert _measure_times is FakeBase.run_benchmark.__globals__["_measure_times"]


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("cached_input", (False, True))
def test_adapter_replays_changed_operand_and_restores_it(monkeypatch, task, cached_input):
    score_input = [1]
    reference = ModuleType("_arena_reference")
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
    reference.perturbed_inputs = perturbed_inputs
    reference.poison_outputs = lambda context, output, *statistics: None
    reference.snapshot_inputs = lambda context: {}
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
        return [1.0] * repetition, {"benchmark_method": "cuda_graph"}
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
    if cached_input:
        with pytest.raises(AssertionError, match="cached operand or output"):
            checked.run_benchmark()
        assert "execution_time_ms" not in row
    else:
        checked.run_benchmark()
        assert row["metadata"]["perturbed_input_replay_checked"]
    assert score_input == [1]
    assert _measure_times is FakeBase.run_benchmark.__globals__["_measure_times"]


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
        return x * rsigma[:, None] * gamma, rsigma
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
        context["y_buffer"].copy_(output[0])
        context["rsigma_buffer"].copy_(output[1])
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


def test_gemm_float32_reference_alias_is_perturbed_once_and_restored():
    path = ROOT / "tasks/instruction2triton/rocmbench/gemm/_arena_reference.py"
    spec = importlib.util.spec_from_file_location("_warn_gemm_alias", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
    context = {"a": a, "a_fp32_ref": a, "b": b, "b_fp32_ref": b}
    original_a, original_b = a.clone(), b.clone()
    with reference.perturbed_inputs(context):
        torch.testing.assert_close(a, torch.flip(original_a, (0,)))
        torch.testing.assert_close(b, torch.flip(original_b, (1,)))
    torch.testing.assert_close(a, original_a)
    torch.testing.assert_close(b, original_b)
