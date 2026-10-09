"""CPU controls for three HIP task contracts; GPU qualification is separate."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


TASKS = Path(__file__).resolve().parents[1] / "tasks/hip2hip/gpumode"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def modules(name, number):
    task = TASKS / name
    original = load(task / f"pytorch_code_module/py_{number}_{name}.py", name + "_original")
    functional = load(task / f"pytorch_code_functional/py_{number}_{name}_func.py",
                      name + "_functional")
    adapter = load(task / "eval_tools/evaluate.py", name + "_adapter")
    return original, functional, adapter


def compare(left, right, **bounds):
    return torch.allclose(left, right, **bounds)


@pytest.mark.parametrize("ignored", (
    "reset.weight", "reset.bias", "update.weight", "update.bias",
    "proposal.weight", "proposal.bias",
))
def test_gate_gru_rejects_each_fixed_parameter_and_restores_state(ignored):
    original, functional, adapter = modules("GateGRUSelectionLayer", "5334")
    torch.manual_seed(0)
    module = original.GateGRUSelectionLayer(4, 16, 0.5).eval()
    selected = functional.GateGRUSelectionLayer(4, 16, 0.5).eval()
    selected.load_state_dict(module.state_dict())
    inputs = [torch.randn(2, 2, 2, 4), torch.randn(2, 2, 2, 4)]
    expected = module(*inputs)
    before = {key: value.clone() for key, value in selected.state_dict().items()}
    adapter.check_gate_parameter_variants(module, selected, None, inputs,
                                          expected, compare, 1e-4, 1e-5)

    def frozen_parameter(x1, x2, *parameters):
        values = list(parameters)
        values[adapter.GATE_PARAMETERS.index(ignored)] = before[ignored]
        return functional.module_fn(x1, x2, *values)

    with pytest.raises(ValueError, match="changed GateGRU"):
        adapter.check_gate_parameter_variants(module, selected, frozen_parameter,
                                              inputs, expected, compare, 1e-4, 1e-5)
    assert all(torch.equal(value, before[key])
               for key, value in module.state_dict().items())
    assert all(torch.equal(value, before[key])
               for key, value in selected.state_dict().items())


@pytest.mark.parametrize("name,number", (("Sigmoid", "11184"), ("TanH", "11178")))
def test_layout_and_dtype_controls_reject_strided_or_wrong_dtype(name, number):
    original, functional, adapter = modules(name, number)
    module = getattr(original, name)().eval()
    selected = getattr(functional, name)().eval()
    assert adapter.check_layout_dtype_controls(module, selected, None, compare,
                                               1e-4, 1e-5, device="cpu") == [
        "strided_float32", "strided_float64", "float64", "float16", "bfloat16"]

    def rejects_strided(value, a, maximum):
        if not value.is_contiguous():
            raise ValueError("strided input rejected")
        return functional.module_fn(value, a, maximum)

    with pytest.raises(ValueError, match="strided input rejected"):
        adapter.check_layout_dtype_controls(module, selected, rejects_strided,
                                            compare, 1e-4, 1e-5, device="cpu")

    def wrong_dtype(value, a, maximum):
        return functional.module_fn(value.float(), a, maximum)

    with pytest.raises(ValueError, match="shape/dtype/device"):
        adapter.check_layout_dtype_controls(module, selected, wrong_dtype,
                                            compare, 1e-4, 1e-5, device="cpu")


class FakeTimedRun:
    def __init__(self):
        self.after_sample = None
        self.outputs = None
        self.bound = False
        self.invoke = None

    def rerun(self):
        self.outputs = self.invoke()
        return self.outputs


def fake_graph(fn, *, warmup, repetition, timed_run, max_graph_repeats,
               use_cuda_graph, **kwargs):
    assert kwargs.get("prepare_fn") is None
    assert (warmup, repetition, max_graph_repeats, use_cuda_graph) == (10, 100, 1, True)
    for _ in range(warmup):
        fn()
    for _ in range(repetition):
        timed_run.outputs = fn()
        timed_run.after_sample(timed_run.outputs)
    timed_run.invoke = fn
    timed_run.bound = True
    return 0.01, {"benchmark_method": "cuda_graph",
                  "benchmark_timed_run_kind": "captured_graph",
                  "benchmark_effective_repeats": 1, "benchmark_samples": repetition}


@pytest.mark.parametrize("ignored", ("reset.weight", "proposal.bias"))
def test_gate_gru_captured_replay_rejects_frozen_live_parameter(ignored, monkeypatch):
    _, functional, adapter = modules("GateGRUSelectionLayer", "5334")
    monkeypatch.syspath_prepend(str(TASKS / "GateGRUSelectionLayer/eval_tools"))
    replay = load(TASKS / "GateGRUSelectionLayer/eval_tools/replay_validation.py",
                  "gate_gru_replay")
    monkeypatch.setitem(sys.modules, "_aka_benchmark",
                        SimpleNamespace(TimedRun=FakeTimedRun))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def cal_kernel_perf(rtol=1e-4, atol=1e-5):
        pass

    perf = SimpleNamespace(cal_kernel_perf=cal_kernel_perf,
                           benchmark_cuda_graph_or_events=fake_graph,
                           _compare_results=compare)
    replay.install(perf, adapter.output_contract)
    torch.manual_seed(0)
    model = functional.GateGRUSelectionLayer(4, 16, 0.5).eval()
    inputs = [torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4)]
    pristine = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        _, meta = perf.cal_hip_latency(model, inputs, hip_fn=functional.module_fn)
    assert meta["changed_gate_parameter_replay_count"] == 6

    def frozen_parameter(x1, x2, *parameters):
        values = list(parameters)
        values[adapter.GATE_PARAMETERS.index(ignored)] = pristine[ignored]
        return functional.module_fn(x1, x2, *values)

    with torch.no_grad(), pytest.raises(ValueError, match="Timed operator output disagrees"):
        perf.cal_hip_latency(model, inputs, hip_fn=frozen_parameter)
    assert all(torch.equal(value, pristine[name])
               for name, value in model.state_dict().items())


def test_sigmoid_unscored_timed_graph_checks_strided_input_and_changed_replay(monkeypatch):
    original, functional, adapter = modules("Sigmoid", "11184")
    monkeypatch.syspath_prepend(str(TASKS / "Sigmoid/eval_tools"))
    perf = SimpleNamespace(benchmark_cuda_graph_or_events=fake_graph,
                           _compare_results=compare)
    monkeypatch.setitem(sys.modules, "_aka_benchmark",
                        SimpleNamespace(TimedRun=FakeTimedRun))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(adapter, "prepare_models", lambda args, device: (
        original.Sigmoid().eval(), functional.Sigmoid().eval()))
    monkeypatch.setattr(adapter, "compile_hip", lambda path, slot: functional.module_fn)
    args = SimpleNamespace(baseline_hip="hip/ref.hip", candidate="hip/selected.hip")
    assert adapter.check_unscored_layout_timing(args, "candidate", perf, device="cpu") == [
        "strided_float32", "strided_float64"]

    remembered = {}

    def stale_replay(value, a, maximum):
        key = value.dtype
        if key not in remembered:
            remembered[key] = functional.module_fn(value.clone(), a, maximum)
        return remembered[key].clone()

    monkeypatch.setattr(adapter, "compile_hip", lambda path, slot: stale_replay)
    with pytest.raises(ValueError, match="Timed operator output disagrees"):
        adapter.check_unscored_layout_timing(args, "candidate", perf, device="cpu")

    def stale_input(value, a, maximum):
        return functional.module_fn(torch.zeros_like(value), a, maximum)

    monkeypatch.setattr(adapter, "compile_hip", lambda path, slot: stale_input)
    with pytest.raises((ValueError, AssertionError), match="Sigmoid|disagrees"):
        adapter.check_unscored_layout_timing(args, "candidate", perf, device="cpu")


def test_scored_case_manifests_remain_original():
    import json
    for name, count in (("GateGRUSelectionLayer", 5), ("Sigmoid", 11), ("TanH", 11)):
        cases = json.loads((TASKS / name / "workload.json").read_text())["cases"]
        assert len(cases) == count
        assert all(case["params"]["model_init_seed"] == 0 for case in cases)
        assert all(case["params"]["correctness_seed"] == 1337 + index
                   for index, case in enumerate(cases))
