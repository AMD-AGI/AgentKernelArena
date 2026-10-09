"""CPU regressions for HIP-v2 semantic findings; GPU validation is separate."""

import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch


TASKS = Path(__file__).resolve().parents[1] / "tasks"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("ignored", ["scale", "bias"])
def test_innerprod_rejects_shape_keyed_affine_constants(ignored):
    task = TASKS / "hip2hip/gpumode/InnerProd"
    source = load(task / "pytorch_code_module/py_11709_InnerProd.py", "innerprod_module")
    func = load(task / "pytorch_code_functional/py_11709_InnerProd_func.py", "innerprod_func")
    adapter = load(task / "eval_tools/evaluate.py", "innerprod_adapter")
    module, functional = source.InnerProd(4).eval(), func.InnerProd(4).eval()
    with torch.no_grad():
        for model in (module, functional):
            model.scale.copy_(torch.tensor([0.625, 0.75, 0.875, 1.0]))
            model.bias.fill_(0.125)
    inputs = [torch.tensor([[[1., 2., 3., 4.]]]),
              torch.arange(1., 17.).reshape(1, 4, 2, 2)]
    expected = module(*inputs)
    original_module = {name: value.clone() for name, value in module.state_dict().items()}
    original_functional = {name: value.clone() for name, value in functional.state_dict().items()}
    compare = lambda left, right, **bounds: torch.allclose(left, right, **bounds)
    adapter.check_affine_parameter_variants(module, functional, None, inputs, expected,
                                            compare, 1e-4, 1e-4)
    fixed = {name: value.clone() for name, value in functional.state_dict().items()}

    def ignores_parameter(feat_img, feat_sound, scale, bias, mode):
        return func.module_fn(feat_img, feat_sound,
                              fixed["scale"] if ignored == "scale" else scale,
                              fixed["bias"] if ignored == "bias" else bias, mode)

    with pytest.raises(ValueError, match="changed InnerProd"):
        adapter.check_affine_parameter_variants(module, functional, ignores_parameter,
                                                inputs, expected, compare, 1e-4, 1e-4)
    assert all(torch.equal(value, original_module[name])
               for name, value in module.state_dict().items())
    assert all(torch.equal(value, original_functional[name])
               for name, value in functional.state_dict().items())


@pytest.mark.parametrize("ignored", ["a", "max"])
def test_tanh_rejects_fixed_scalar_parameters(ignored):
    task = TASKS / "hip2hip/gpumode/TanH"
    source = load(task / "pytorch_code_module/py_11178_TanH.py", "tanh_module")
    func = load(task / "pytorch_code_functional/py_11178_TanH_func.py", "tanh_func")
    adapter = load(task / "eval_tools/evaluate.py", "tanh_adapter")
    module, functional = source.TanH().eval(), func.TanH().eval()
    inputs = [torch.tensor([-2., -0.5, 0.25, 1.5])]
    expected = module(*inputs)
    compare = lambda left, right, **bounds: torch.allclose(left, right, **bounds)
    adapter.check_scalar_variants(module, functional, None, inputs, expected,
                                  compare, 1e-4, 1e-5)

    def fixed_scalar(v, a, max_val):
        return func.module_fn(v, 1 if ignored == "a" else a,
                              10 if ignored == "max" else max_val)

    with pytest.raises(ValueError, match="changed TanH"):
        adapter.check_scalar_variants(module, functional, fixed_scalar, inputs, expected,
                                      compare, 1e-4, 1e-5)
    assert (module.a, module.max, functional.a, functional.max) == (1, 10, 1, 10)


@pytest.mark.parametrize("task", ["InnerProd", "TanH"])
def test_hip2hip_rejects_event_downgrade_in_both_slots(task):
    adapter = load(TASKS / f"hip2hip/gpumode/{task}/eval_tools/evaluate.py", task + "_method")
    adapter.require_graph_method({"benchmark_method": "cuda_graph",
                                  "reference_benchmark_method": "cuda_graph"}, "cuda_graph")
    for case, method in (({"benchmark_method": "cuda_event_fallback"}, "cuda_event_fallback"),
                         ({"benchmark_method": "cuda_graph",
                           "reference_benchmark_method": "cuda_event_fallback"}, "cuda_graph")):
        with pytest.raises(RuntimeError, match="declared graph timing method"):
            adapter.require_graph_method(case, method)


@pytest.mark.parametrize("task", ["InnerProd", "TanH"])
@pytest.mark.parametrize("fault", ["candidate_event", "internal_reference_event"])
def test_hip2hip_performance_action_rejects_event_downgrade(task, fault, tmp_path, monkeypatch):
    adapter = load(TASKS / f"hip2hip/gpumode/{task}/eval_tools/evaluate.py",
                   task + "_action_method")
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    (tmp_path / "build").mkdir()
    tensor = torch.ones(2)
    inputs = [{"shape": [2], "dtype": "torch.float32", "stride": [1]}]
    rows = [{"test_case_id": "case_0", "params": {"inputs": inputs}}]
    perf = types.SimpleNamespace(
        _compare_results=lambda *a, **kw: True,
        load_modu_obj=lambda *a: None,
        load_func_obj=lambda *a: None,
        load_function_from_path=lambda *a: lambda: iter([[tensor]]),
    )

    def benchmark(*paths, **kwargs):
        list(perf.load_function_from_path("module", "get_inputs")())
        reference = {"benchmark_method": "cuda_event_fallback"
                     if fault == "internal_reference_event" else "cuda_graph",
                     "replay_validation_valid": True,
                     "benchmark_samples": 100, "validated_sample_count": 100}
        perf._write_perf_report({"status": "ok", "test_cases": [{
            "case_idx": 0, "correct": True, "ref_time": 0.2, "opt_time": 0.1,
            "benchmark_method": "cuda_event_fallback" if fault == "candidate_event" else "cuda_graph",
            "reference_benchmark_method": reference["benchmark_method"],
            "reference_benchmark": reference,
            "replay_validation_valid": True,
            "benchmark_samples": 100, "validated_sample_count": 100,
        }]})

    perf.cal_kernel_perf = benchmark
    monkeypatch.setitem(sys.modules, "cal_kernel_perf", perf)
    monkeypatch.setitem(sys.modules, "replay_validation",
                        types.SimpleNamespace(install=lambda *a: None))
    args = types.SimpleNamespace(candidate="kernel.hip", baseline_hip="baseline.hip",
                                 module="module.py", functional="functional.py",
                                 model_class="Example")
    with pytest.raises(RuntimeError, match="changed the declared graph timing method"):
        adapter.performance(args, "candidate", rows)


def test_decode_partial_page_and_partition_are_correctness_only(monkeypatch):
    task = TASKS / "image_kernel/mi300x_sglang_hip_pa_decode"
    harness = load(task / "scripts/task_runner.py", "decode_partial_runner")
    adapter = load(task / "scripts/task_adapter.py", "decode_partial_adapter")
    adapter.validate_workloads(harness)
    manifest = json.loads((task / "workloads.json").read_text())
    assert manifest["original_cases"]["correctness"] == json.loads(json.dumps(harness.CASES))
    assert manifest["original_cases"]["performance"] == json.loads(json.dumps(harness.PERF_CASES))
    assert [row["test_case_id"] for row in manifest["cases"][:4]] == [
        "correctness-0", "correctness-1", "pa_decode_ctx1024_s256", "pa_decode_ctx8192_s128"]
    added = manifest["cases"][4]
    assert added["checks"] == ["correctness"]
    assert added["params"]["context_lengths"] == [1025, 1024, 1009, 1008]
    assert harness._context_lengths(1025, 4, [1025, 1024, 1009, 1008]) == [1025, 1024, 1009, 1008]
    for invalid in ([1025, 1024], [1025, 1024, 0, 1008], [1024] * 4):
        with pytest.raises(ValueError):
            harness._context_lengths(1025, 4, invalid)

    real_randperm = torch.randperm
    monkeypatch.setattr(torch, "set_default_device", lambda device: None)
    monkeypatch.setattr(torch, "randperm", lambda n, **kw: real_randperm(n, **{**kw, "device": "cpu"}))

    def cache_factory(blocks, block_size, layers, heads, head_size, cache_dtype, dtype, seed, device):
        return ([torch.zeros((blocks, heads, head_size // 16, block_size, 16), dtype=dtype)],
                [torch.zeros((blocks, heads, head_size, block_size), dtype=dtype)])

    monkeypatch.setitem(sys.modules, "csrc.cpp_itfs.pa", SimpleNamespace(
        pa_ragged_test=SimpleNamespace(kv_cache_factory=cache_factory)))
    case = harness._make_case(**harness.EXTRA_CASES[0])
    lengths = [1025, 1024, 1009, 1008]
    assert case["seq_lens"].tolist() == lengths
    counts = [(length + 15) // 16 for length in lengths]
    indptr = [0]
    for count in counts:
        indptr.append(indptr[-1] + count)
    assert case["kv_indptr"].tolist() == indptr
    assert case["kv_last_page_lens"].tolist() == [(length - 1) % 16 + 1 for length in lengths]
    assert case["max_num_partitions"] == 5
    for index, count in enumerate(counts):
        torch.testing.assert_close(case["kv_page_indices"][indptr[index]:indptr[index + 1]],
                                   case["block_tables"][index, :count], atol=0, rtol=0)


def test_decode_correctness_executes_added_case(monkeypatch):
    harness = load(TASKS / "image_kernel/mi300x_sglang_hip_pa_decode/scripts/task_runner.py",
                   "decode_partial_execution")
    seen = []
    monkeypatch.setattr(harness, "_make_case", lambda **cfg: seen.append(cfg) or cfg)
    monkeypatch.setattr(harness, "_run_aiter", lambda case: torch.tensor([float(case["ctx_lens"])]))
    monkeypatch.setattr(harness, "_run_torch", lambda case: torch.tensor([float(case["ctx_lens"])]))
    harness.run_correctness()
    assert seen == [*harness.CASES, *[cfg for _, cfg in harness.PERF_CASES], *harness.EXTRA_CASES]
    monkeypatch.setattr(harness, "_run_aiter", lambda case: torch.tensor([0.])
                        if "context_lengths" in case else torch.tensor([float(case["ctx_lens"])]))
    with pytest.raises(AssertionError):
        harness.run_correctness()


def test_decode_manifest_rejects_missing_or_changed_partial_control(tmp_path, monkeypatch):
    task = TASKS / "image_kernel/mi300x_sglang_hip_pa_decode"
    harness = load(task / "scripts/task_runner.py", "decode_partial_manifest_runner")
    adapter = load(task / "scripts/task_adapter.py", "decode_partial_manifest_adapter")
    original = json.loads((task / "workloads.json").read_text())
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    path = tmp_path / "workloads.json"
    for fault in ("missing", "changed"):
        altered = json.loads(json.dumps(original))
        if fault == "missing":
            del altered["additional_cases"]
        else:
            altered["cases"][-1]["params"]["context_lengths"][0] = 1024
        path.write_text(json.dumps(altered))
        with pytest.raises(ValueError):
            adapter.validate_workloads(harness)
