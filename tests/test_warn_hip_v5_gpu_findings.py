"""Focused CPU controls for HIP-v5 validator findings."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch


TASKS = Path(__file__).resolve().parents[1] / "tasks"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["roiaware_pool3d", "points_in_boxes"])
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
