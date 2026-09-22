"""CPU checks of task control flow, held-out injection, and replay validation.

Kernels and device timing are substituted; these are not GPU validation or
performance measurements. The actual runners, injector, and replay handle run.
"""

import ast
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from src.held_out.injection import apply_injection
from src.perf_helper_materialization import materialize_perf_helpers_in_workspace

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cpu_runner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def load(task_name, shapes):
        task_dir = ROOT / "tasks/triton2triton/vllm" / task_name
        workspace = tmp_path / task_name
        scripts = workspace / "scripts"
        shutil.copytree(task_dir, workspace, ignore=shutil.ignore_patterns('build', '__pycache__', '.pytest_cache'))
        path = scripts / "task_runner.py"
        monkeypatch.syspath_prepend(str(workspace))
        monkeypatch.syspath_prepend(str(scripts))
        for file in workspace.glob('*.py'):
            monkeypatch.delitem(sys.modules, file.stem, raising=False)

        assert apply_injection(workspace, {
            "file": "scripts/task_runner.py",
            "find_marker": "TEST_SHAPES",
            "replacement_code": f"TEST_SHAPES = {shapes!r}",
        })
        materialize_perf_helpers_in_workspace(workspace)
        tree = ast.parse(path.read_text())
        # Redirect only device literals in the temporary runner. Production
        # sources and the runner's case iteration/validation remain unchanged.
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "cuda":
                node.value = "cpu"
        module = ModuleType(f"test_{task_name}")
        module.__file__ = str(path)
        exec(compile(tree, str(path), "exec"), module.__dict__)
        return module

    return load


@pytest.mark.parametrize("fail_case", [None, 6, 8])
def test_moe_checks_every_injected_shape(cpu_runner, monkeypatch, fail_case):
    shapes = [(m, 128, 2, 32, 2, 32) for m in range(4, 12)]
    task = cpu_runner("triton_fused_moe_gptq_awq", shapes)
    visited = []
    variants = []

    def kernel(x, weights, scales, zeros, ids, routed, *,
               mul_routed_weight, group_size, use_int4):
        visited.append(x.shape[0])
        variants.append((use_int4, zeros is not None,
                         mul_routed_weight, routed is not None))
        if len(visited) == fail_case:
            raise RuntimeError("deliberate held-out failure")
        return task.reference(dict(A=x, qweight=weights, scales=scales, zeros=zeros,
                                   ids=ids, weights=routed),
                              dict(mul_routed_weight=mul_routed_weight,
                                   group_size=group_size, use_int4=use_int4))

    monkeypatch.setattr(task, "load_module", lambda: SimpleNamespace(
        fused_moe_gptq_awq=kernel,
    ))
    if fail_case is None:
        monkeypatch.setattr(sys, "argv", ["task_runner.py", "correctness"])
        with pytest.raises(SystemExit) as exit_info:
            task.main()
        assert exit_info.value.code == 0
        report = json.loads((Path(task.TASK_DIR) / "build/correctness_report.json").read_text())
        assert report["status"] == "ok"
        assert report["num_shapes"] == len(visited) == len(shapes)
        assert visited == [shape[0] for shape in shapes]
        assert set(variants) == {(True, True, True, True)}  # original scored INT4 path
    else:
        ok, error = task.run_correctness()
        assert not ok
        assert "deliberate held-out failure" in str(error)
        assert visited == [shape[0] for shape in shapes[:fail_case]]


@pytest.mark.parametrize("task_name, shapes, extra_shapes", [
    ("triton_awq_dequantize", [(48, 8, 16), (80, 16, 16)],
     [(33, 9), (96, 33), (512, 65), (1024, 128)]),
    ("triton_scale_swizzle", [(384, 12), (640, 8)], []),
])
def test_injected_shapes_reach_correctness_and_performance(
    cpu_runner, monkeypatch, task_name, shapes, extra_shapes,
):
    task = cpu_runner(task_name, shapes)
    visited = []
    bad_shape = None

    if task_name == "triton_awq_dequantize":
        # This test verifies case routing, not AWQ arithmetic. Avoid the large
        # scalar reference for correctness-only boundary shapes on CPU.
        def reference(weights, scales, zeros, group_size):
            return torch.zeros(weights.shape[0], weights.shape[1] * 8,
                               dtype=scales.dtype)

        monkeypatch.setattr(task, "reference_awq_dequantize", reference)

        def kernel(data, scales, zeros):
            visited.append(tuple(data.shape))
            result = reference(data, scales, zeros, data.shape[0] // scales.shape[0])
            return result + 1 if tuple(data.shape) == bad_shape else result

        module = SimpleNamespace(awq_dequantize_triton=kernel)
    else:
        def kernel(data):
            visited.append(tuple(data.shape))
            result = task.reference_scale_swizzle(data)
            return result ^ 1 if tuple(data.shape) == bad_shape else result

        module = SimpleNamespace(triton_mx_block_rearrange=kernel)

    monkeypatch.setattr(task, "load_module", lambda: module)

    def benchmark(fn, **kwargs):
        fn()
        return 1.0, {}  # Synthetic sentinel; no device timing is performed.

    monkeypatch.setattr(task, "_benchmark_cuda_graph_or_events", benchmark)
    injected = [shape[:2] for shape in shapes]
    assert task.run_correctness() == (True, None)
    assert visited == injected  # extra controls have separate v2 manifest indices
    visited.clear()
    cases = task.run_performance()
    assert visited == injected
    assert len(cases) == len(shapes)
    assert all(case["execution_time_ms"] == 1.0 for case in cases)

    visited.clear()
    bad_shape = injected[1]
    ok, error = task.run_correctness()
    assert not ok
    assert "Shape 2" in error
    assert visited == injected


@pytest.mark.parametrize("behavior, valid", [
    ("in_place", True), ("view", True), ("none", True),
    ("no_op", False), ("detached_zeros", False),
    ("partial", False), ("nan", False), ("first_call_only", False),
])
def test_zero_replay_checks_destination_and_restores_state(
    cpu_runner, monkeypatch, behavior, valid,
):
    task = cpu_runner("triton_write_zeros_to_output", [(2, 4)])
    initial_states = []

    def kernel(output):
        initial_states.append(output.clone())
        if behavior == "no_op":
            return output
        if behavior == "detached_zeros":
            return torch.zeros_like(output)
        if behavior == "partial":
            output[0].zero_()
            return output
        if behavior == "nan":
            return output.fill_(float("nan"))
        if behavior != "first_call_only" or len(initial_states) == 1:
            output.zero_()
        if behavior == "view":
            return output.view_as(output)
        if behavior == "none":
            return None
        return output

    def benchmark(fn, *, timed_run, prepare_fn=None, **kwargs):
        if prepare_fn is not None: prepare_fn()
        captured_output = fn()

        def replay():
            if prepare_fn is not None: prepare_fn()
            fn()
            return captured_output

        replay()
        timed_run._bind(replay, captured_output)
        return 1.0, {}  # Synthetic sentinel; no device timing is performed.

    monkeypatch.setattr(task, "load_module", lambda: SimpleNamespace(write_zeros=kernel))
    monkeypatch.setattr(task, "_benchmark_cuda_graph_or_events", benchmark)
    root = Path(task.TASK_DIR)
    def local(name):
        spec = __import__('importlib.util', fromlist=['spec_from_file_location']).spec_from_file_location(name, root / (name+'.py'))
        module = __import__('importlib.util', fromlist=['module_from_spec']).module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    local('_arena_replay').install(task, local('_arena_contract'))
    cases = task.run_performance()
    assert len(cases) == 1
    assert cases[0]["execution_time_ms"] == (1.0 if valid else -1.0)
    assert len(initial_states) >= 2
    assert torch.count_nonzero(initial_states[0]) > 0
    if valid:
        assert torch.equal(initial_states[0], initial_states[1])
        assert torch.all(initial_states[-1] == 1)  # real changed-input replay
