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
        scripts.mkdir(parents=True)
        path = scripts / "task_runner.py"
        shutil.copyfile(task_dir / "scripts/task_runner.py", path)
        shutil.copyfile(task_dir / "config.yaml", workspace / "config.yaml")
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
        return task.reference_fused_moe(
            x, weights, scales, zeros, ids, routed,
            mul_routed_weight, group_size, use_int4,
        ).half()

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
        assert set(variants) == set(task.CORRECTNESS_VARIANTS)
    else:
        ok, error = task.run_correctness()
        assert not ok
        assert f"Shape {fail_case}: exception: deliberate held-out failure" in error
        assert visited == [shape[0] for shape in shapes[:fail_case]]


@pytest.mark.parametrize("task_name, shapes, extra_shapes", [
    ("triton_awq_dequantize", [(48, 8, 16), (80, 16, 16)],
     [(33, 9), (96, 33), (512, 65), (1024, 128)]),
    ("triton_scale_swizzle", [(257, 9), (513, 7)], [(129, 4), (128, 5)]),
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
    assert visited == injected + extra_shapes
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

    def benchmark(fn, *, prepare_fn, timed_run, **kwargs):
        prepare_fn()
        captured_output = fn()

        def replay():
            prepare_fn()
            fn()
            return captured_output

        replay()
        timed_run._bind(replay, captured_output)
        return 1.0, {}  # Synthetic sentinel; no device timing is performed.

    monkeypatch.setattr(task, "load_module", lambda: SimpleNamespace(write_zeros=kernel))
    monkeypatch.setattr(task, "_benchmark_cuda_graph_or_events", benchmark)
    cases = task.run_performance()
    assert len(cases) == 1
    assert cases[0]["execution_time_ms"] == (1.0 if valid else -1.0)
    assert len(initial_states) == 3
    assert torch.count_nonzero(initial_states[0]) > 0
    assert all(torch.equal(state, initial_states[0]) for state in initial_states)
