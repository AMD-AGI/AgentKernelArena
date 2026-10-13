"""Reported HIP benchmark samples must be checked, including an early bad sample."""

import importlib.util
import os
import shutil
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
    "hip2hip/CrossEntropyLossLabelSmoothing",
    "hip2hip/TanH",
    "hip2hip/layer_normalization",
    "hip2hip/MultiHeadAttention",
    "hip2hip/NormalAttention_dot",
    "hip2hip/Sigmoid",
    "hip2hip/GateGRUSelectionLayer",
    "hip2hip/InnerProd",
)
NATIVE_TASKS = (
    "hip2hip/roipoint_pool3d",
    "hip2hip/points_in_boxes",
    "hip2hip/roiaware_pool3d",
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


@pytest.mark.parametrize("task", ("hip2hip/Sigmoid", "torch2hip/gpumode/11184_Sigmoid"))
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
def test_model_validates_every_reported_sample(task, method, monkeypatch):
    validator = load_validator(task, "eval_tools")
    perf = types.SimpleNamespace(
        cal_kernel_perf=lambda rtol=0.0, atol=0.0: None,
        benchmark_cuda_graph_or_events=lambda invoke, **kw: bench(invoke, method=method, **kw),
        _compare_results=lambda a, b, **kw: torch.allclose(a, b, **kw),
    )
    validator.install(perf, lambda expected, actual: torch.testing.assert_close(actual.shape, expected.shape))
    if task == "hip2hip/GateGRUSelectionLayer":
        gate = load_file(ROOT / task / "pytorch_code_functional/py_5334_GateGRUSelectionLayer_func.py")
        def fixture():
            model = gate.GateGRUSelectionLayer(2, 4, 0.0).eval()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(0.1)
            return model, [torch.tensor([[[[0.5, -0.3]]]]),
                           torch.tensor([[[[0.2, 0.7]]]])]
    elif task == "torch2hip/gpumode/11754_layer_normalization":
        root = ROOT / task
        controls = load_file(root / "eval_tools/case_controls.py")
        module = load_file(root / "pytorch_code_module/py_11754_layer_normalization.py")
        monkeypatch.setitem(sys.modules, "case_controls", controls)

        def fixture():
            model = module.layer_normalization(4).eval()
            inputs = [torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)]
            controls.configure_models([model], inputs)
            assert not torch.equal(model.gamma, torch.ones_like(model.gamma))
            assert not torch.equal(model.beta, torch.zeros_like(model.beta))
            return model, inputs
    else:
        fixture = lambda: (Model(), [torch.tensor([0.5])])
    model, inputs = fixture()
    pristine_inputs = [value.clone() for value in inputs]
    pristine_state = {name: value.clone() for name, value in model.state_dict().items()}
    elapsed, metadata = perf.cal_hip_latency(model, inputs, n_iter=3)
    assert elapsed == 0.1
    assert metadata["validated_sample_count"] == 3
    assert metadata["replay_validation_valid"] is True
    if task == "torch2hip/gpumode/11754_layer_normalization":
        assert metadata["changed_affine_replay_count"] == 2
        assert metadata["model_state_tensor_count"] == 2
        for original, current in zip(pristine_inputs, inputs):
            torch.testing.assert_close(current, original, rtol=0, atol=0)
        for name, original in pristine_state.items():
            torch.testing.assert_close(model.state_dict()[name], original, rtol=0, atol=0)

    perf.benchmark_cuda_graph_or_events = lambda invoke, **kw: bench(invoke, method=method, bad_first=True, **kw)
    with pytest.raises((ValueError, AssertionError), match="Timed operator output disagrees"):
        model, inputs = fixture()
        perf.cal_hip_latency(model, inputs, n_iter=3)


@pytest.mark.parametrize("method", ("event", "graph"))
@pytest.mark.parametrize("parameter", ("gamma", "beta"))
def test_layernorm_replay_rejects_frozen_live_affine(method, parameter, monkeypatch):
    task = "torch2hip/gpumode/11754_layer_normalization"
    root = ROOT / task
    validator = load_validator(task, "eval_tools")
    controls = load_file(root / "eval_tools/case_controls.py")
    module = load_file(root / "pytorch_code_module/py_11754_layer_normalization.py")
    monkeypatch.setitem(sys.modules, "case_controls", controls)
    model = module.layer_normalization(4).eval()
    inputs = [torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)]
    controls.configure_models([model], inputs)
    pristine_input = inputs[0].clone()
    pristine_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def frozen_affine_benchmark(_invoke, *, timed_run, repetition, **kwargs):
        assert kwargs["max_graph_repeats"] == 1
        fixed = getattr(model, parameter).detach().clone()

        def stale_parameter_call():
            x = inputs[0]
            gamma = fixed if parameter == "gamma" else model.gamma
            beta = fixed if parameter == "beta" else model.beta
            return gamma * (x - x.mean(-1, keepdim=True)) / (
                x.std(-1, keepdim=True) + model.epsilon) + beta

        timed_run._invoke = stale_parameter_call
        for _ in range(repetition):
            timed_run.outputs = timed_run._invoke()
            timed_run.after_sample(timed_run.outputs)
        return 0.1, {
            "benchmark_method": "cuda_graph" if method == "graph" else "cuda_event_fallback",
            "benchmark_timed_run_kind": "captured_graph" if method == "graph" else "eager_callable",
            "benchmark_samples": repetition,
        }

    perf = types.SimpleNamespace(
        cal_kernel_perf=lambda rtol=0.0, atol=0.0: None,
        benchmark_cuda_graph_or_events=frozen_affine_benchmark,
        _compare_results=lambda a, b, **kw: torch.allclose(a, b, **kw),
    )
    validator.install(perf, lambda expected, actual: torch.testing.assert_close(actual.shape, expected.shape))
    with pytest.raises(ValueError, match=f"Timed operator ignored live {parameter}"):
        perf.cal_hip_latency(model, inputs, n_iter=3)
    torch.testing.assert_close(inputs[0], pristine_input, rtol=0, atol=0)
    for name, original in pristine_state.items():
        torch.testing.assert_close(model.state_dict()[name], original, rtol=0, atol=0)


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
    "hip2hip/CrossEntropyLossLabelSmoothing",
    "hip2hip/MultiHeadAttention",
    "hip2hip/GateGRUSelectionLayer",
    "hip2hip/InnerProd",
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
    root = ROOT / "hip2hip/CrossEntropyLossLabelSmoothing"
    validator = load_validator("hip2hip/CrossEntropyLossLabelSmoothing", "eval_tools")
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
    root = ROOT / "hip2hip/CrossEntropyLossLabelSmoothing"
    validator = load_validator("hip2hip/CrossEntropyLossLabelSmoothing", "eval_tools")
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
    scored_inputs = inputs[0].clone()
    with torch.no_grad():
        expected = model(*inputs)
        changed, changed_inputs = validator.changed_input_reference(
            inputs, 0, expected, model, torch.allclose, 1e-4, 1e-5)
    assert not torch.allclose(expected, changed, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(changed, model(*changed_inputs), rtol=0, atol=0)
    # The scored random inputs and prior replay below -4 cannot reach this
    # valid interval. The original reduction still distinguishes dropping it.
    with torch.no_grad():
        middle_negative = -3 - scored_inputs / 2
        correct = model(middle_negative)
        dropped = model(torch.where((middle_negative > -4) & (middle_negative < -2),
                                    0, middle_negative))
    assert not torch.allclose(correct, dropped, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("fault", ("x_face", "y_face", "z_endpoint", "rotation"))
def test_roipoint_exact_face_reference_rejects_wrong_membership(monkeypatch, tmp_path, fault):
    from src.perf_helper_materialization import materialize_perf_helpers_in_workspace

    original = ROOT / "hip2hip/roipoint_pool3d"
    root = tmp_path / "roipoint_pool3d"
    shutil.copytree(original, root, ignore=shutil.ignore_patterns("__pycache__", "build", "*.pyc"))
    assert materialize_perf_helpers_in_workspace(root, root=ROOT.parent)
    helper = root / "scripts/_aka_benchmark.py"
    assert helper.read_bytes() == (ROOT.parent / "src/tools/perf/aka_benchmark.py").read_bytes()
    previous_directory = os.getcwd()
    previous_path = sys.path[:]
    missing = object()
    protected_names = ("_aka_benchmark", "reference_checks", "replay_validation", "task_runner")
    previous_modules = {name: sys.modules.get(name, missing) for name in protected_names}
    try:
        runner = load_file(root / "scripts/task_runner.py")
    finally:
        os.chdir(previous_directory)
        sys.path[:] = previous_path
        for name, module in previous_modules.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    assert Path(runner._PROTECTED_BENCHMARK.__file__).resolve() == helper.resolve()
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
