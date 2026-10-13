"""Regression controls for the migrated flip task's numerical and replay gates."""
from __future__ import annotations

import importlib.util
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/triton2triton/rocmbench/test_triton_flip"
PERF = ROOT / "src/tools/perf"


def _contract():
    spec = importlib.util.spec_from_file_location("flip_reference_control", TASK / "_arena_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = torch.arange(21, dtype=torch.float32).reshape(3, 7).clone()
    z = torch.empty_like(x)
    return module.FlipContract(x, z)


def _checked_call(contract, standin):
    contract.prepare_sample()
    contract.before_sample(1)
    result = standin(contract.x, contract.z)
    contract.after_sample(result)


def test_changed_input_and_poisoned_output_reject_stale_or_missing_writes():
    contract = _contract()
    _checked_call(contract, lambda x, z: z.copy_(torch.flip(x, [1])))
    cached = contract.z.clone()
    with pytest.raises(AssertionError):
        _checked_call(contract, lambda x, z: z.copy_(cached))
    with pytest.raises(AssertionError):
        _checked_call(contract, lambda x, z: z)


def test_control_rejects_input_mutation_and_wrong_return_buffer():
    contract = _contract()

    def mutate_input(x, z):
        z.copy_(torch.flip(x, [1]))
        x.zero_()
        return z

    with pytest.raises(AssertionError):
        _checked_call(contract, mutate_input)

    def stale_destination(x, z):
        return torch.flip(x, [1])

    with pytest.raises(AssertionError):
        _checked_call(contract, stale_destination)


@pytest.mark.parametrize("dtype,original,wrong", [
    (torch.int32, 100, 101),
    (torch.float16, 100, 100.5),
    (torch.float32, 100, 100.5),
    (torch.bfloat16, 100, 101),
    (torch.int32, 16_777_217, 16_777_216),
])
def test_exact_replay_oracle_rejects_small_and_float_cast_hidden_errors(dtype, original, wrong):
    spec = importlib.util.spec_from_file_location("flip_reference_exact", TASK / "_arena_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = torch.tensor([[original, 50]], dtype=dtype)
    actual = expected.clone()
    actual[0, 0] = wrong
    if original == 100:
        assert torch.allclose(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)
    else:
        assert torch.equal(actual.float(), expected.float())
    with pytest.raises(module.NumericalMismatch, match="Exact output mismatch"):
        module.compare(actual, expected, exact=True)
    contract = module.FlipContract(torch.flip(expected, [1]).clone(), torch.empty_like(expected))
    contract.prepare_sample()
    contract.before_sample(1)
    contract.z.copy_(expected)
    contract.z[0, 0] = wrong
    with pytest.raises(module.NumericalMismatch, match="Exact output mismatch"):
        contract.after_sample(contract.z)


def test_exact_oracle_keeps_shape_dtype_and_finiteness_guards():
    spec = importlib.util.spec_from_file_location("flip_reference_guards", TASK / "_arena_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = torch.tensor([[100.0, 50.0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="shape/device"):
        module.compare(expected.reshape(2, 1), expected, exact=True)
    with pytest.raises(ValueError, match="dtype"):
        module.compare(expected.to(torch.float16), expected, exact=True)
    nonfinite = expected.clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        module.compare(nonfinite, expected, exact=True)


@pytest.mark.parametrize("dtype_str,original,wrong", [
    ("int32", 100, 101),
    ("float16", 100, 100.5),
    ("float32", 100, 100.5),
    ("bfloat16", 100, 101),
    ("int32", 16_777_217, 16_777_216),
])
@pytest.mark.parametrize("alias_fault", [False, True])
def test_original_correctness_body_rejects_exact_value_errors(dtype_str, original, wrong, alias_fault):
    # Execute the protected test body with a CPU launch stand-in. This catches
    # accidental reintroduction of float conversion or approximate tolerance.
    source = (TASK / "test_triton_flip.py").read_text()
    node = deepcopy(next(n for n in ast.parse(source).body
                         if isinstance(n, ast.FunctionDef) and n.name == "test_flip"))
    node.decorator_list = []
    dtype = {"int32": torch.int32, "float16": torch.float16,
             "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_str]
    launch_count = [0]

    class WrongKernel:
        def __getitem__(self, grid):
            assert grid == (1,)

            def launch(x, z, n, m):
                launch_count[0] += 1
                value = torch.flip(x, [1]).clone()
                assert x[0, -1].item() == torch.tensor(original, dtype=dtype).item()
                if (launch_count[0] == 2) == alias_fault:
                    value[0, 0] = wrong
                z.copy_(value)

            return launch

    np_dtype = np.int32 if dtype_str == "int32" else np.float32
    namespace = {
        "torch": torch, "np": np, "result_gold": {}, "flip_kernel": WrongKernel(),
        "set_seed": lambda: None,
        "torch_dtype_from_str": lambda _: dtype,
        "gen_numpy_array_for_torch_conversion":
            lambda shape, dtype_str, **kwargs: np.full(shape, original, dtype=np_dtype),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 "test_triton_flip.py", "exec"), namespace)
    request = SimpleNamespace(node=SimpleNamespace(name="test_flip[cpu-control]"))
    with pytest.raises(AssertionError, match="Tensor-likes are not"):
        namespace["test_flip"](2, 9, dtype_str, request, device="cpu")
    assert launch_count[0] == (2 if alias_fault else 1)


def _run_direct_correctness_with_cpu_kernel(dtype_str, launch):
    source = (TASK / "test_triton_flip.py").read_text()
    node = deepcopy(next(n for n in ast.parse(source).body
                         if isinstance(n, ast.FunctionDef) and n.name == "test_flip"))
    node.decorator_list = []
    dtype = {"int32": torch.int32, "float16": torch.float16,
             "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_str]

    class Kernel:
        def __getitem__(self, grid):
            assert grid == (1,)
            return launch

    def generate(shape, dtype_str, *, low, high):
        assert (low, high) == (-100, 100)
        return np.full(shape, 100, dtype=np.int32 if dtype_str == "int32" else np.float32)

    namespace = {
        "torch": torch, "np": np, "result_gold": {}, "flip_kernel": Kernel(),
        "set_seed": lambda: None,
        "torch_dtype_from_str": lambda _: dtype,
        "gen_numpy_array_for_torch_conversion": generate,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 "test_triton_flip.py", "exec"), namespace)
    request = SimpleNamespace(node=SimpleNamespace(name="test_flip[cpu-control]"))
    namespace["test_flip"](2, 9, dtype_str, request, device="cpu")


@pytest.mark.parametrize("dtype_str", ["int32", "float16", "float32", "bfloat16"])
def test_direct_correctness_exercises_signed_and_exact_values(dtype_str):
    observed = []

    def launch(x, z, n, m):
        observed.append(x.clone())
        z.copy_(torch.flip(x, [1]))

    _run_direct_correctness_with_cpu_kernel(dtype_str, launch)
    assert len(observed) == 2
    flat = observed[0].flatten()
    assert flat[0].item() < 0
    assert flat[1].item() == 0
    if dtype_str == "int32":
        assert flat[0].item() == -16_777_217
        assert flat[2].item() == 16_777_217
    else:
        assert torch.signbit(flat[1])
        assert flat[2].item() == 0.5
        assert flat[3].item() == 100.5
        assert torch.isposinf(flat[4])
        assert torch.isneginf(flat[5])
        assert torch.isnan(flat[6])


@pytest.mark.parametrize("dtype_str", ["int32", "float16", "float32", "bfloat16"])
@pytest.mark.parametrize("fault,reason", [
    ("clamp_negative", "Tensor-likes are not"),
    ("mutate_input", "Separate flip modified input"),
])
def test_direct_correctness_rejects_negative_clamp_and_input_mutation(dtype_str, fault, reason):
    launches = [0]

    def launch(x, z, n, m):
        launches[0] += 1
        value = torch.flip(x, [1])
        if fault == "clamp_negative":
            z.copy_(torch.clamp_min(value, 0))
        else:
            z.copy_(value)
            x[0, 0] = 0

    with pytest.raises(AssertionError, match=reason):
        _run_direct_correctness_with_cpu_kernel(dtype_str, launch)
    assert launches == [1]


def test_original_scored_manifest_and_extra_odd_shape_controls():
    cases = json.loads((TASK / "workloads.json").read_text())["cases"]
    scored = [case for case in cases if case["params"]["function"] == "test_performance"]
    correctness = [case for case in cases if case["params"]["function"] == "test_flip"]
    expected_scored = {
        (n, m, dtype, warps)
        for n, m in ((128, 64), (64, 128))
        for dtype in ("int32", "float16", "float32", "bfloat16")
        for warps in (1, 2, 4)
    }
    assert len(scored) == 24
    assert {
        (args["N_const"], args["M_const"], args["dtype_str"], args["num_warps_launch"])
        for case in scored
        for args in (case["params"]["arguments"],)
    } == expected_scored
    assert len(correctness) == 40
    assert {(case["params"]["arguments"]["N_rows"],
             case["params"]["arguments"]["M_cols"])
            for case in correctness} == {
                (1, 512), (8, 64), (256, 16), (512, 8),
                (3, 7), (5, 33), (17, 9), (63, 65), (1, 1), (7, 1),
            }


def test_kernel_public_interface_guard_rejects_extra_arguments(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("flip_eval_guard", TASK / "_arena_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "candidate.py"
    source.write_text("@triton.jit\ndef flip_kernel(X, Z, N, M, BLOCK):\n    return X\n")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="four-argument"):
        module.inspect_candidate({"source": source.name, "kernel_symbols": ["flip_kernel"]})


@pytest.mark.parametrize("fault", ["none", "drop_after", "two_calls", "unbound", "graph_fallback"])
def test_benchmark_adapter_rejects_incomplete_sample_evidence(monkeypatch, fault):
    spec = importlib.util.spec_from_file_location("flip_eval_replay", TASK / "_arena_eval.py")
    eval_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_module)
    ref_spec = importlib.util.spec_from_file_location("_arena_reference", TASK / "_arena_reference.py")
    reference = importlib.util.module_from_spec(ref_spec)
    ref_spec.loader.exec_module(reference)
    monkeypatch.setitem(sys.modules, "_arena_reference", reference)

    class FakeTimedRun:
        def __init__(self):
            self.bound = False
            self.before_sample = None
            self.after_sample = None
            self._rerun = None

        def rerun(self):
            return self._rerun()

    def fake_benchmark(op, *, prepare_fn, timed_run, repetition, **kwargs):
        for _ in range(repetition):
            prepare_fn()
            timed_run.before_sample(2 if fault == "two_calls" else 1)
            result = op()
            if fault != "drop_after":
                timed_run.after_sample(result)
        if fault != "unbound":
            timed_run.bound = True
            timed_run._rerun = lambda: (prepare_fn(), op())[1]
        return [1.0] * repetition, {
            "benchmark_method": "cuda_event_fallback" if fault == "graph_fallback" else "cuda_graph",
            "benchmark_effective_repeats": 1,
        }

    benchmark_module = ModuleType("_aka_benchmark")
    benchmark_module.TimedRun = FakeTimedRun
    benchmark_module.benchmark_cuda_graph_or_events_samples = fake_benchmark
    monkeypatch.setitem(sys.modules, "_aka_benchmark", benchmark_module)
    adapter_module = ModuleType("performance_utils_pytest")
    adapter_module._BENCHMARK_RESULTS = []
    adapter_module._compute_timing_stats = lambda times, config: {"mean": sum(times) / len(times)}
    monkeypatch.setitem(sys.modules, "performance_utils_pytest", adapter_module)

    class Base:
        def __init__(self, *, op_callable, **kwargs):
            self.op_callable = op_callable
            self.op_name = "flip"
            self.config = SimpleNamespace(warm_up=10, repetition=3)
            self.prepare_fn = None
            self.use_cuda_graph = True
            self.fallback_reason = None

    plugin = SimpleNamespace(action="performance", current_row={"test_case_id": "case"}, exercised=set())
    checked = eval_module.benchmark_type(Base, plugin, None)

    def make_instance():
        x_perf_tensor = torch.arange(21, dtype=torch.float32).reshape(3, 7).clone()
        z_perf_buffer = torch.empty_like(x_perf_tensor)
        op = lambda: z_perf_buffer.copy_(torch.flip(x_perf_tensor, [1]))
        return checked(op_callable=op)

    benchmark = make_instance()
    if fault == "none":
        benchmark.run_benchmark(current_params_dict={})
        assert plugin.current_row["metadata"]["all_timed_samples_checked"] == 3
        assert plugin.current_row["metadata"]["changed_input_replay_checked"]
        assert plugin.current_row["metadata"]["device_timing"] == {
            "benchmark_method": "cuda_graph", "benchmark_effective_repeats": 1,
        }
    else:
        with pytest.raises((AssertionError, RuntimeError)):
            benchmark.run_benchmark(current_params_dict={})
        assert "case" not in plugin.exercised
        assert not adapter_module._BENCHMARK_RESULTS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="canonical replay control requires a GPU")
@pytest.mark.parametrize("fault", ["stale", "no_write", "small_delta"])
def test_canonical_gpu_replay_rejects_wrong_timed_output(tmp_path, monkeypatch, fault):
    shutil.copy2(PERF / "aka_benchmark.py", tmp_path / "_aka_benchmark.py")
    shutil.copy2(PERF / "performance_utils_pytest.py", tmp_path / "performance_utils_pytest.py")
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = importlib.util.spec_from_file_location("flip_eval_native_replay", TASK / "_arena_eval.py")
    eval_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_module)
    ref_spec = importlib.util.spec_from_file_location("_arena_reference", TASK / "_arena_reference.py")
    reference = importlib.util.module_from_spec(ref_spec)
    ref_spec.loader.exec_module(reference)
    monkeypatch.setitem(sys.modules, "_arena_reference", reference)
    for name in ("_aka_benchmark", "performance_utils_pytest"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    class Base:
        def __init__(self, *, op_callable):
            self.op_callable = op_callable
            self.op_name = "flip"
            self.config = SimpleNamespace(warm_up=2, repetition=3)
            self.prepare_fn = None
            self.use_cuda_graph = True
            self.fallback_reason = None

    plugin = SimpleNamespace(action="performance", current_row={"test_case_id": "gpu"}, exercised=set())
    checked = eval_module.benchmark_type(Base, plugin, None)

    def make_instance():
        x_perf_tensor = torch.arange(21, device="cuda", dtype=torch.float32).reshape(3, 7).clone()
        z_perf_buffer = torch.empty_like(x_perf_tensor)
        stale = torch.flip(x_perf_tensor, [1]).clone()
        calls = [0]

        def op():
            calls[0] += 1
            if calls[0] <= 3:
                z_perf_buffer.copy_(torch.flip(x_perf_tensor, [1]))
            elif fault == "stale":
                z_perf_buffer.copy_(stale)
            elif fault == "small_delta":
                z_perf_buffer.copy_(torch.flip(x_perf_tensor, [1]) + 0.5)
            else:
                # Capture real GPU work while deliberately leaving poison intact.
                z_perf_buffer.add_(0)
            return z_perf_buffer

        return checked(op_callable=op)

    with pytest.raises(RuntimeError, match="CUDA-graph capture failed; timed_run cannot validate") as failure:
        make_instance().run_benchmark(current_params_dict={})
    # The canonical helper wraps sample-check failures. Require the numerical
    # failure from the measured replay, not an unrelated capture/runtime error.
    cause = failure.value.__cause__
    assert isinstance(cause, reference.NumericalMismatch)
    frames = []
    trace = cause.__traceback__
    while trace is not None:
        frames.append(trace.tb_frame.f_code.co_name)
        trace = trace.tb_next
    assert "after_sample" in frames
    assert "gpu" not in plugin.exercised


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native Triton control requires a GPU")
def test_native_correctness_action_checks_actual_flip_kernel(tmp_path):
    # A workspace receives the canonical helpers before its task-owned action.
    workspace = tmp_path / "flip"
    shutil.copytree(TASK, workspace)
    shutil.copy2(PERF / "aka_benchmark.py", workspace / "_aka_benchmark.py")
    shutil.copy2(PERF / "performance_utils_pytest.py", workspace / "performance_utils_pytest.py")
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    proc = subprocess.run(
        [sys.executable, "_arena_eval.py", "baseline", "correctness"],
        cwd=workspace, env=env, text=True, capture_output=True, timeout=600,
    )
    marker = "ARENA_EVAL_RESULT="
    line = next((line for line in proc.stdout.splitlines() if line.startswith(marker)), None)
    assert line is not None, proc.stdout[-3000:] + proc.stderr[-3000:]
    report = json.loads(line[len(marker):])
    failed = next((row for row in report.get("cases", []) if row.get("status") != "PASS"), {})
    assert proc.returncode == 0, failed.get("reason", report.get("reason", report))
    assert report["status"] == "PASS"
    assert len(report["cases"]) == 64
    assert all(case["status"] == "PASS" for case in report["cases"])
