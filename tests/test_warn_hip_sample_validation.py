"""Reported HIP benchmark samples must be checked, including an early bad sample."""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1] / "tasks"
MODEL_TASKS = (
    "torch2hip/kernelbench/level2/l2n55_Matmul_MaxPool_Sum_Scale",
    "torch2hip/kernelbench/level2/l2n73_Conv2d_BatchNorm_Scaling",
    "torch2hip/gpumode/10024_Feedforward",
    "torch2hip/gpumode/11184_Sigmoid",
    "torch2hip/gpumode/11754_layer_normalization",
    "hip2hip/gpumode/CrossEntropyLossLabelSmoothing",
    "hip2hip/gpumode/TanH",
    "hip2hip/gpumode/layer_normalization",
    "hip2hip/gpumode/MultiHeadAttention",
    "hip2hip/gpumode/NormalAttention_dot",
    "hip2hip/gpumode/Sigmoid",
    "hip2hip/gpumode/GateGRUSelectionLayer",
    "hip2hip/gpumode/InnerProd",
)
NATIVE_TASKS = (
    "hip2hip/others/roipoint_pool3d",
    "hip2hip/others/points_in_boxes",
    "hip2hip/others/roiaware_pool3d",
)


class FakeTimedRun:
    def __init__(self):
        self.after_sample = None
        self.outputs = None
        self._invoke = None

    def rerun(self):
        self.outputs = self._invoke()
        return self.outputs


def load_validator(task, folder):
    path = ROOT / task / folder / "replay_validation.py"
    spec = importlib.util.spec_from_file_location("replay_validation_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_file(path):
    spec = importlib.util.spec_from_file_location("hip_task_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def fake_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "_aka_benchmark", types.SimpleNamespace(TimedRun=FakeTimedRun))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setitem(
        sys.modules, "case_controls",
        types.SimpleNamespace(reference=lambda x, _eps, _dist: x.tanh().clone(),
                              assert_declared_control=lambda _model, _inputs: None),
    )


def bench(invoke, *, timed_run, repetition, bad_first=False, method="event", **_kwargs):
    assert _kwargs["max_graph_repeats"] == 1
    if method == "graph":
        captured = invoke().clone()

        def replay():
            captured.copy_(invoke())
            return captured

        timed_run._invoke = replay
    else:
        timed_run._invoke = invoke
    for index in range(repetition):
        value = timed_run._invoke()
        if bad_first and index == 0:
            value.add_(1)
        timed_run.outputs = value
        if timed_run.after_sample is not None:
            timed_run.after_sample(value)
    return 0.1, {
        "benchmark_method": "cuda_graph" if method == "graph" else "cuda_event_fallback",
        "benchmark_timed_run_kind": "captured_graph" if method == "graph" else "eager_callable",
        "benchmark_samples": repetition,
    }


class Model(torch.nn.Module):
    smooth_eps = 0.0
    smooth_dist = 0.0

    def forward(self, x, fn=None):
        return x.tanh().clone()


@pytest.mark.parametrize("task", ("hip2hip/gpumode/Sigmoid", "torch2hip/gpumode/11184_Sigmoid"))
def test_sigmoid_unscored_math_controls_reject_shared_wrong_result(task, monkeypatch):
    root = ROOT / task
    runner = load_file(root / "eval_tools/evaluate.py")
    checks = load_file(root / "eval_tools/correctness_check.py")
    monkeypatch.setitem(sys.modules, "correctness_check", checks)
    module = load_file(root / "pytorch_code_module/py_11184_Sigmoid.py").Sigmoid()
    functional = load_file(root / "pytorch_code_functional/py_11184_Sigmoid_func.py").Sigmoid()
    controls = runner.check_sigmoid_controls(module, functional, None, 1e-4, 1e-5, device="cpu")
    assert controls[:3] == ["scalar", "empty", "extreme"]
    if task.startswith("torch2hip/"):
        assert {"half_scalar", "inverse_extreme", "steep_scalar", "shallow_scalar"} <= set(controls)
    assert (module.a, module.max, functional.a, functional.max) == (1, 10, 1, 10)

    module.forward = lambda value: torch.zeros_like(value)
    functional.forward = lambda value, fn=None: torch.zeros_like(value)
    with pytest.raises(ValueError, match="mathematical reference"):
        runner.check_sigmoid_controls(module, functional, None, 1e-4, 1e-5, device="cpu")


@pytest.mark.parametrize("task", MODEL_TASKS)
@pytest.mark.parametrize("method", ("event", "graph"))
def test_model_validates_every_reported_sample(task, method):
    validator = load_validator(task, "eval_tools")
    perf = types.SimpleNamespace(
        cal_kernel_perf=lambda rtol=0.0, atol=0.0: None,
        benchmark_cuda_graph_or_events=lambda invoke, **kw: bench(invoke, method=method, **kw),
        _compare_results=lambda a, b, **kw: torch.allclose(a, b, **kw),
    )
    validator.install(perf, lambda expected, actual: torch.testing.assert_close(actual.shape, expected.shape))
    elapsed, metadata = perf.cal_hip_latency(Model(), [torch.tensor([0.5])], n_iter=3)
    assert elapsed == 0.1
    assert metadata["validated_sample_count"] == 3
    assert metadata["replay_validation_valid"] is True

    perf.benchmark_cuda_graph_or_events = lambda invoke, **kw: bench(invoke, method=method, bad_first=True, **kw)
    with pytest.raises((ValueError, AssertionError), match="Timed operator output disagrees"):
        perf.cal_hip_latency(Model(), [torch.tensor([0.5])], n_iter=3)


@pytest.mark.parametrize("task", NATIVE_TASKS)
@pytest.mark.parametrize("method", ("event", "graph"))
def test_native_validates_every_reported_sample(task, method):
    validator = load_validator(task, "scripts")
    source = torch.tensor([2], dtype=torch.int32)
    invoke = lambda: source.clone()
    check = lambda output: torch.testing.assert_close(output, source, rtol=0, atol=0)
    def change_input():
        source.add_(1)
        return lambda output: torch.testing.assert_close(output, torch.tensor([3], dtype=source.dtype), rtol=0, atol=0)

    benchmark = lambda invoke, **kw: bench(invoke, method=method, **kw)
    elapsed, metadata = validator.measure(benchmark, invoke, (source,), check, (change_input,), repetition=3)
    assert elapsed == 0.1
    assert metadata["validated_sample_count"] == 3
    assert metadata["replay_validation_valid"] is True
    assert metadata["changed_input_replay_valid"] is True
    assert source.item() == 2

    with pytest.raises(AssertionError):
        validator.measure(lambda invoke, **kw: bench(invoke, method=method, bad_first=True, **kw),
                          invoke, (source,), check, (change_input,), repetition=3)


@pytest.mark.parametrize("task", (
    "torch2hip/gpumode/10024_Feedforward",
    "hip2hip/gpumode/CrossEntropyLossLabelSmoothing",
    "hip2hip/gpumode/MultiHeadAttention",
    "hip2hip/gpumode/GateGRUSelectionLayer",
    "hip2hip/gpumode/InnerProd",
))
def test_model_replay_rejects_frozen_second_operand(task, monkeypatch):
    validator = load_validator(task, "eval_tools")
    class TwoInput(torch.nn.Module):
        smooth_eps = 0.0
        smooth_dist = 0.0

        def forward(self, a, b, fn=None):
            return (a + b).tanh().clone()

    monkeypatch.setitem(sys.modules, "case_controls", types.SimpleNamespace(
        reference=lambda a, b, _eps, _dist: (a + b).tanh().clone(),
        assert_declared_control=lambda _model, _inputs: None))
    a = torch.tensor([[0.5, 0.2]])
    b = torch.tensor([[0.2, 0.8]])

    def frozen_second_benchmark(invoke, *, timed_run, repetition, **_kwargs):
        frozen_b = b.clone()
        timed_run._invoke = lambda: (a + frozen_b).tanh().clone()
        for _ in range(repetition):
            timed_run.outputs = timed_run._invoke()
            timed_run.after_sample(timed_run.outputs)
        return 0.1, {"benchmark_method": "cuda_graph",
                     "benchmark_timed_run_kind": "captured_graph",
                     "benchmark_samples": repetition}

    perf = types.SimpleNamespace(
        cal_kernel_perf=lambda rtol=0.0, atol=0.0: None,
        benchmark_cuda_graph_or_events=frozen_second_benchmark,
        _compare_results=lambda left, right, **kw: torch.allclose(left, right, **kw),
    )
    validator.install(perf, lambda expected, actual: torch.testing.assert_close(actual.shape, expected.shape))
    with pytest.raises((ValueError, AssertionError), match="Timed operator output disagrees"):
        perf.cal_hip_latency(TwoInput(), [a, b], n_iter=3)
    torch.testing.assert_close(a, torch.tensor([[0.5, 0.2]]))
    torch.testing.assert_close(b, torch.tensor([[0.2, 0.8]]))


def test_loss_changed_logits_distinguish_all_declared_mean_cases():
    root = ROOT / "hip2hip/gpumode/CrossEntropyLossLabelSmoothing"
    validator = load_validator("hip2hip/gpumode/CrossEntropyLossLabelSmoothing", "eval_tools")
    module = load_file(root / "pytorch_code_module/py_12501_CrossEntropyLossLabelSmoothing.py")
    controls = load_file(root / "eval_tools/case_controls.py")
    torch.manual_seed(0)
    model = module.CrossEntropyLossLabelSmoothing()
    for case_index, inputs in enumerate(module.get_inputs()):
        controls.configure_models([model], inputs)
        pristine = [value.clone() for value in inputs]
        expected = model(*inputs)
        for operand in (0, 1):
            changed, changed_inputs = validator.changed_input_reference(
                inputs, operand, expected, model, torch.allclose, 1e-4, 1e-5)
            assert not torch.allclose(expected, changed, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(changed, model(*changed_inputs), rtol=0, atol=0)
            torch.testing.assert_close(inputs[1 - operand], pristine[1 - operand], rtol=0, atol=0)
            torch.testing.assert_close(changed_inputs[1 - operand], pristine[1 - operand], rtol=0, atol=0)
            assert not torch.equal(changed_inputs[operand], pristine[operand])
            target = changed_inputs[1]
            assert torch.isfinite(target).all() and (target >= 0).all()
            torch.testing.assert_close(target.sum(-1), torch.ones_like(target[..., 0]),
                                       rtol=1e-5, atol=1e-6)
            for before, after in zip(pristine, inputs):
                after.copy_(before)
        # The performance runner resets the seed after each case; the next
        # get_inputs() yield must see that same state.
        torch.manual_seed(1337 + case_index)


@pytest.mark.parametrize("frozen_index", (0, 1))
def test_loss_large_mean_replay_rejects_frozen_operand(monkeypatch, frozen_index):
    root = ROOT / "hip2hip/gpumode/CrossEntropyLossLabelSmoothing"
    validator = load_validator("hip2hip/gpumode/CrossEntropyLossLabelSmoothing", "eval_tools")
    module = load_file(root / "pytorch_code_module/py_12501_CrossEntropyLossLabelSmoothing.py")
    controls = load_file(root / "eval_tools/case_controls.py")
    monkeypatch.setitem(sys.modules, "case_controls", controls)
    torch.manual_seed(0)
    model = module.CrossEntropyLossLabelSmoothing()
    inputs = None
    for case_index, case_inputs in enumerate(module.get_inputs()):
        inputs = case_inputs
        torch.manual_seed(1337 + case_index)
    controls.configure_models([model], inputs)
    original = [value.clone() for value in inputs]

    def frozen_operand_benchmark(_invoke, *, timed_run, repetition, **_kwargs):
        frozen = inputs[frozen_index].clone()

        def replay():
            arguments = list(inputs)
            arguments[frozen_index] = frozen
            return model(*arguments)

        timed_run._invoke = replay
        for _ in range(repetition):
            timed_run.outputs = timed_run._invoke()
            timed_run.after_sample(timed_run.outputs)
        return 0.1, {"benchmark_method": "cuda_graph",
                     "benchmark_timed_run_kind": "captured_graph",
                     "benchmark_samples": repetition}

    perf = types.SimpleNamespace(
        cal_kernel_perf=lambda rtol=1e-4, atol=1e-5: None,
        benchmark_cuda_graph_or_events=frozen_operand_benchmark,
        _compare_results=lambda left, right, **kw: torch.allclose(left, right, **kw),
    )
    validator.install(perf, lambda expected, actual: torch.testing.assert_close(actual.shape, expected.shape))
    with pytest.raises(ValueError, match="Timed operator output disagrees"):
        perf.cal_hip_latency(model, inputs, n_iter=3)
    for before, after in zip(original, inputs):
        torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_matmul_pool_sum_changed_input_survives_large_reduction():
    root = ROOT / "torch2hip/kernelbench/level2/l2n55_Matmul_MaxPool_Sum_Scale"
    validator = load_validator("torch2hip/kernelbench/level2/l2n55_Matmul_MaxPool_Sum_Scale", "eval_tools")
    module = load_file(root / "pytorch_code_module/py_l2n55_Matmul_MaxPool_Sum_Scale.py")
    # Keep the scored reduction path and largest batch, using a CPU-sized
    # matrix so the probe does not allocate the declared 32768-square weights.
    torch.manual_seed(0)
    model = module.Matmul_MaxPool_Sum_Scale(1024, 4096, 2, 0.5)
    inputs = [torch.rand(512, 1024)]
    with torch.no_grad():
        expected = model(*inputs)
        changed, changed_inputs = validator.changed_input_reference(
            inputs, 0, expected, model, torch.allclose, 1e-4, 1e-5)
    assert not torch.allclose(expected, changed, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(changed, model(*changed_inputs), rtol=0, atol=0)


@pytest.mark.parametrize("fault", ("x_face", "y_face", "z_endpoint", "rotation"))
def test_roipoint_exact_face_reference_rejects_wrong_membership(monkeypatch, fault):
    root = ROOT / "hip2hip/others/roipoint_pool3d"
    monkeypatch.setitem(sys.modules, "_aka_benchmark", types.SimpleNamespace(
        TimedRun=FakeTimedRun,
        benchmark_cuda_graph_or_events=lambda *args, **kwargs: None,
        hip_source_graph_capture_policy=lambda *args: (True, None)))
    previous_directory = os.getcwd()
    try:
        runner = load_file(root / "scripts/task_runner.py")
    finally:
        os.chdir(previous_directory)
    controls = load_file(root / "scripts/reference_checks.py")
    controls.check_face_reference(runner)
    original = runner.check_point_in_box

    def wrong_membership(point, box):
        if fault == "x_face" and box[0] == 0 and point[0] == 1:
            return True
        if fault == "y_face" and box[0] == 0 and point[1] == 1:
            return True
        if fault == "z_endpoint" and box[0] == 0 and point[2] == 0:
            return False
        if fault == "rotation":
            unrotated = box.clone()
            unrotated[6] = 0
            return original(point, unrotated)
        return original(point, box)

    monkeypatch.setattr(runner, "check_point_in_box", wrong_membership)
    with pytest.raises(AssertionError, match="CPU box reference has wrong"):
        controls.check_face_reference(runner)


@pytest.mark.parametrize("task", NATIVE_TASKS)
def test_native_replay_rejects_frozen_second_operand(task):
    validator = load_validator(task, "scripts")
    first = torch.tensor([2], dtype=torch.int32)
    second = torch.tensor([3], dtype=torch.int32)
    invoke = lambda: (first + second).clone()
    check = lambda output: torch.testing.assert_close(output, torch.tensor([5]), rtol=0, atol=0)

    def change_first():
        first.add_(1)
        return lambda output: torch.testing.assert_close(output, torch.tensor([6]), rtol=0, atol=0)

    def change_second():
        second.add_(1)
        return lambda output: torch.testing.assert_close(output, torch.tensor([6]), rtol=0, atol=0)

    def frozen_second_benchmark(_invoke, *, timed_run, repetition, **_kwargs):
        frozen_second = second.clone()
        timed_run._invoke = lambda: (first + frozen_second).clone()
        for _ in range(repetition):
            timed_run.outputs = timed_run._invoke()
            timed_run.after_sample(timed_run.outputs)
        return 0.1, {"benchmark_samples": repetition}

    with pytest.raises(AssertionError):
        validator.measure(frozen_second_benchmark, invoke, (first, second), check,
                          (change_first, change_second), repetition=3)
    assert first.item() == 2 and second.item() == 3
