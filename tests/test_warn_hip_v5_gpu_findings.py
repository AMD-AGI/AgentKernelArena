"""Focused CPU controls for HIP-v5 validator findings."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch


TASKS = Path(__file__).resolve().parents[1] / "tasks"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["roiaware_pool3d", "points_in_boxes", "roipoint_pool3d"])
def test_native_graph_tasks_reject_candidate_controlled_event_downgrade(monkeypatch, name):
    task = TASKS / "hip2hip/others" / name
    adapter = load(task / "scripts/evaluate.py", name + "_timing_adapter")
    workload = json.loads((task / "workload.json").read_text())
    case = workload["cases"][0]
    measured = [{"test_case_id": case["test_case_id"], "params": case["params"],
                 "execution_time_ms": 0.01, "benchmark_method": "cuda_graph"}]
    assert adapter.checked_performance([case], measured, workload["graph_policy"])[0]["status"] == "PASS"
    # A candidate translation unit can set this in a static initializer when
    # the native extension is loaded, after the adapter configured graph timing.
    monkeypatch.setenv("AKA_BENCHMARK_FORCE_EVENT", "1")
    measured[0]["benchmark_method"] = "cuda_event_fallback"
    with pytest.raises(RuntimeError, match="changed the declared device timing method"):
        adapter.checked_performance([case], measured, workload["graph_policy"])


@pytest.mark.parametrize("relative", [
    "gpumode/10024_Feedforward",
    "gpumode/11184_Sigmoid",
    "gpumode/11754_layer_normalization",
    "kernelbench/level2/l2n55_Matmul_MaxPool_Sum_Scale",
    "kernelbench/level2/l2n73_Conv2d_BatchNorm_Scaling",
])
def test_torch2hip_adapter_requires_complete_timed_sample_validation(relative):
    task = TASKS / "torch2hip" / relative
    adapter = load(task / "eval_tools/evaluate.py", relative.replace("/", "_") + "_timing_adapter")
    valid = {"benchmark_method": "cuda_graph", "replay_validation_valid": True,
             "timed_output_checked": True, "benchmark_samples": 100,
             "validated_sample_count": 100}
    assert adapter.checked_timed_benchmark(valid, "candidate", None) is valid
    assert adapter.checked_timed_benchmark(valid, "baseline", None) is valid
    reference = valid.copy()
    measured = {**valid, "reference_benchmark": reference}
    assert adapter.checked_timed_benchmark(measured, "baseline", "hip_ref/ref.hip") is reference
    for field, invalid in (
        ("replay_validation_valid", False),
        ("replay_validation_valid", None),
        ("timed_output_checked", False),
        ("benchmark_samples", None),
        ("benchmark_samples", 0),
        ("validated_sample_count", 99),
        ("validated_sample_count", True),
    ):
        incomplete = {**valid, field: invalid}
        with pytest.raises(RuntimeError, match="did not validate all reported samples"):
            adapter.checked_timed_benchmark(incomplete, "candidate", None)
        with pytest.raises(RuntimeError, match="did not validate all reported samples"):
            adapter.checked_timed_benchmark({"reference_benchmark": incomplete}, "baseline",
                                            "hip_ref/ref.hip")
    with pytest.raises(RuntimeError, match="omitted timing metadata"):
        adapter.checked_timed_benchmark(valid, "baseline", "hip_ref/ref.hip")


@pytest.mark.parametrize("relative", [
    "gpumode/10024_Feedforward",
    "gpumode/11184_Sigmoid",
    "gpumode/11754_layer_normalization",
    "kernelbench/level2/l2n55_Matmul_MaxPool_Sum_Scale",
    "kernelbench/level2/l2n73_Conv2d_BatchNorm_Scaling",
])
@pytest.mark.parametrize("fault", ["missing_sample", "event_method"])
def test_torch2hip_performance_action_rejects_invalid_timing(relative, fault, tmp_path, monkeypatch):
    task = TASKS / "torch2hip" / relative
    adapter = load(task / "eval_tools/evaluate.py", relative.replace("/", "_") + "_action_adapter")
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    (tmp_path / "build").mkdir()
    inputs = [{"shape": [2], "dtype": "torch.float32", "stride": [1]}]
    rows = [{"test_case_id": "case_0", "params": {"inputs": inputs}}]
    perf = types.SimpleNamespace(
        _compare_results=lambda *a, **kw: True,
        load_modu_obj=lambda *a: None,
        load_func_obj=lambda *a: None,
        load_function_from_path=lambda *a: lambda: iter([[torch.ones(2)]]),
    )

    def benchmark(*paths, **kwargs):
        list(perf.load_function_from_path("module", "get_inputs")())
        perf._write_perf_report({"status": "ok", "test_cases": [{
            "case_idx": 0, "correct": True, "ori_time": 0.2, "opt_time": 0.1,
            "benchmark_method": "cuda_event_fallback" if fault == "event_method" else "cuda_graph",
            "benchmark_samples": 100,
            "validated_sample_count": 99 if fault == "missing_sample" else 100,
            "replay_validation_valid": True,
            "timed_output_checked": True,
        }]})

    perf.cal_kernel_perf = benchmark
    monkeypatch.setitem(sys.modules, "cal_kernel_perf", perf)
    monkeypatch.setitem(sys.modules, "replay_validation",
                        types.SimpleNamespace(install=lambda *a: None))
    args = types.SimpleNamespace(candidate="kernel.hip", module="module.py",
                                 functional="functional.py", model_class="Example",
                                 baseline_hip=None)
    reason = ("did not validate all reported samples" if fault == "missing_sample"
              else "changed the declared graph timing method")
    with pytest.raises(RuntimeError, match=reason):
        adapter.performance(args, "candidate", rows)


@pytest.mark.parametrize("relative", [
    "gpumode/10024_Feedforward",
    "gpumode/11184_Sigmoid",
    "gpumode/11754_layer_normalization",
    "kernelbench/level2/l2n55_Matmul_MaxPool_Sum_Scale",
    "kernelbench/level2/l2n73_Conv2d_BatchNorm_Scaling",
])
def test_torch2hip_rejects_internal_baseline_method_downgrade(relative):
    adapter = load(TASKS / "torch2hip" / relative / "eval_tools/evaluate.py",
                   relative.replace("/", "_") + "_graph_gate")
    adapter.require_graph_method({"benchmark_method": "cuda_graph",
                                  "reference_benchmark_method": "cuda_graph"}, "cuda_graph")
    with pytest.raises(RuntimeError, match="changed the declared graph timing method"):
        adapter.require_graph_method({"benchmark_method": "cuda_graph",
                                      "reference_benchmark_method": "cuda_event_fallback"}, "cuda_graph")


def test_conv_batchnorm_controls_reject_omitted_bn_and_event_method():
    import torch.nn.functional as nnf

    task = TASKS / "torch2hip/kernelbench/level2/l2n73_Conv2d_BatchNorm_Scaling"
    module_source = load(task / "pytorch_code_module/py_l2n73_Conv2d_BatchNorm_Scaling.py", "conv_bn_module")
    functional_source = load(task / "pytorch_code_functional/py_l2n73_Conv2d_BatchNorm_Scaling_func.py", "conv_bn_functional")
    adapter = load(task / "eval_tools/evaluate.py", "conv_bn_adapter")
    torch.manual_seed(0)
    module = module_source.Conv2d_BatchNorm_Scaling(8, 64, 3, 2).eval()
    functional = functional_source.Conv2d_BatchNorm_Scaling(8, 64, 3, 2).eval()
    functional.load_state_dict(module.state_dict())
    inputs = [torch.randn(2, 8, 7, 7)]
    expected = module(*inputs)
    compare = lambda left, right, **bounds: torch.allclose(left, right, **bounds)
    original_module = {name: value.clone() for name, value in module.state_dict().items()}
    original_functional = {name: value.clone() for name, value in functional.state_dict().items()}
    adapter.check_nonidentity_batchnorm(module, functional, None, inputs, expected,
                                        compare, 1e-4, 1e-5)

    def omits_bn(x, conv_weight, conv_bias, _bn_weight, _bn_bias, _bn_mean,
                 _bn_var, _bn_eps, scaling_factor):
        return nnf.conv2d(x, conv_weight, conv_bias) * scaling_factor

    with pytest.raises(ValueError, match="nonidentity BatchNorm"):
        adapter.check_nonidentity_batchnorm(module, functional, omits_bn, inputs, expected,
                                            compare, 1e-4, 1e-5)
    assert all(torch.equal(value, original_module[name]) for name, value in module.state_dict().items())
    assert all(torch.equal(value, original_functional[name]) for name, value in functional.state_dict().items())
    adapter.require_graph_method({"benchmark_method": "cuda_graph"}, "cuda_graph")
    with pytest.raises(RuntimeError, match="declared graph timing method"):
        adapter.require_graph_method({"benchmark_method": "cuda_event_fallback"}, "cuda_event_fallback")
    with pytest.raises(RuntimeError, match="declared graph timing method"):
        adapter.require_graph_method({"benchmark_method": "cuda_graph",
                                      "reference_benchmark_method": "cuda_event_fallback"}, "cuda_graph")


@pytest.mark.parametrize("ignored", ["weight", "bias"])
def test_conv_parameter_controls_reject_fixed_seed_zero_parameter(ignored):
    import torch.nn.functional as nnf

    task = TASKS / "torch2hip/kernelbench/level2/l2n73_Conv2d_BatchNorm_Scaling"
    module_source = load(task / "pytorch_code_module/py_l2n73_Conv2d_BatchNorm_Scaling.py", "conv_parameter_module")
    functional_source = load(task / "pytorch_code_functional/py_l2n73_Conv2d_BatchNorm_Scaling_func.py",
                             "conv_parameter_functional")
    adapter = load(task / "eval_tools/evaluate.py", "conv_parameter_adapter")
    torch.manual_seed(0)
    module = module_source.Conv2d_BatchNorm_Scaling(8, 64, 3, 2).eval()
    functional = functional_source.Conv2d_BatchNorm_Scaling(8, 64, 3, 2).eval()
    functional.load_state_dict(module.state_dict())
    inputs = [torch.rand(2, 8, 7, 7)]
    expected = module(*inputs)
    compare = lambda left, right, **bounds: torch.allclose(left, right, **bounds)
    initial_module = {name: value.clone() for name, value in module.state_dict().items()}
    initial_functional = {name: value.clone() for name, value in functional.state_dict().items()}
    adapter.check_conv_parameter_variants(module, functional, None, inputs, expected,
                                          compare, 1e-4, 1e-5)
    frozen_weight = functional.conv.weight.detach().clone()
    frozen_bias = functional.conv.bias.detach().clone()

    def fixed_parameter(x, weight, bias, bn_weight, bn_bias, bn_mean, bn_var, bn_eps, scale):
        return nnf.batch_norm(nnf.conv2d(x, frozen_weight if ignored == "weight" else weight,
                                        frozen_bias if ignored == "bias" else bias),
                              bn_mean, bn_var, bn_weight, bn_bias, training=False, eps=bn_eps) * scale

    with pytest.raises(ValueError, match=f"changed convolution {ignored}"):
        adapter.check_conv_parameter_variants(module, functional, fixed_parameter, inputs,
                                              expected, compare, 1e-4, 1e-5)
    assert all(torch.equal(value, initial_module[name]) for name, value in module.state_dict().items())
    assert all(torch.equal(value, initial_functional[name]) for name, value in functional.state_dict().items())


@pytest.mark.parametrize("ignored", ["fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"])
def test_feedforward_parameter_controls_reject_fixed_original_tensor(ignored):
    task = TASKS / "torch2hip/gpumode/10024_Feedforward"
    module_source = load(task / "pytorch_code_module/py_10024_Feedforward.py", "feedforward_module")
    functional_source = load(task / "pytorch_code_functional/py_10024_Feedforward_func.py", "feedforward_functional")
    adapter = load(task / "eval_tools/evaluate.py", "feedforward_adapter")
    torch.manual_seed(0)
    module = module_source.Feedforward(4).eval()
    functional = functional_source.Feedforward(4).eval()
    with torch.no_grad():
        for left, right in (("fc1.weight", "fc1_weight"), ("fc1.bias", "fc1_bias"),
                            ("fc2.weight", "fc2_weight"), ("fc2.bias", "fc2_bias")):
            functional.state_dict()[right].copy_(module.state_dict()[left])
    inputs_by_case = list(module_source.get_inputs())
    compare = lambda expected, actual, **bounds: torch.allclose(expected, actual, **bounds)
    original_module = {name: value.clone() for name, value in module.state_dict().items()}
    original_functional = {name: value.clone() for name, value in functional.state_dict().items()}
    assert [case[0].shape[0] for case in inputs_by_case] == [1, 2, 4, 8, 16]
    for inputs in inputs_by_case:
        adapter.check_parameter_variants(module, functional, functional_source.module_fn,
                                         inputs, compare, 1e-4, 1e-5)
    assert all(torch.equal(value, original_module[name]) for name, value in module.state_dict().items())
    assert all(torch.equal(value, original_functional[name]) for name, value in functional.state_dict().items())
    names = ("fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias")
    frozen = original_functional[ignored]

    def ignores_one_parameter(x, y, *parameters):
        selected = list(parameters)
        selected[names.index(ignored)] = frozen
        return functional_source.module_fn(x, y, *selected)

    for inputs in inputs_by_case:
        with pytest.raises(ValueError, match=f"changed parameter: {ignored}"):
            adapter.check_parameter_variants(module, functional, ignores_one_parameter,
                                             inputs, compare, 1e-4, 1e-5)
    assert all(torch.equal(value, original_module[name]) for name, value in module.state_dict().items())
    assert all(torch.equal(value, original_functional[name]) for name, value in functional.state_dict().items())
