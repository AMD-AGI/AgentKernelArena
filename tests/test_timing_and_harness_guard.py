import ast
import json
from pathlib import Path

import pytest


def test_fused_moe_harness_uses_dynamic_routing_and_matched_eager_timing():
    runner = (
        Path(__file__).resolve().parents[1]
        / "tasks/triton2triton/vllm/triton_fused_moe/scripts/task_runner.py"
    )
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    correctness = functions["run_correctness"]
    performance = functions["run_performance"]
    for function in (correctness, performance):
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        assert any(
            isinstance(call.func, ast.Name)
            and call.func.id == "_make_routing_variants"
            for call in calls
        )
        assert any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "topk_ids"
            and call.func.attr == "copy_"
            for call in calls
        )

    correctness_routing_calls = [
        node
        for node in ast.walk(correctness)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_make_routing_variants"
    ]
    assert len(correctness_routing_calls) == 1
    correctness_routing_keywords = {
        keyword.arg: keyword.value
        for keyword in correctness_routing_calls[0].keywords
    }
    assert ast.literal_eval(correctness_routing_keywords["count"]) == 3
    correctness_loops = [
        node
        for node in ast.walk(correctness)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "enumerate"
        and isinstance(node.iter.args[0], ast.Name)
        and node.iter.args[0].id == "routing_variants"
    ]
    assert len(correctness_loops) == 1
    correctness_loop_calls = [
        node for node in ast.walk(correctness_loops[0]) if isinstance(node, ast.Call)
    ]
    correctness_candidate_call = next(
        call
        for call in correctness_loop_calls
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "mod"
        and call.func.attr == "fused_moe"
    )
    correctness_reference_call = next(
        call
        for call in correctness_loop_calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "reference_fused_moe"
    )
    assert correctness_reference_call.lineno < correctness_candidate_call.lineno
    assert any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "torch"
        and call.func.attr == "equal"
        for call in correctness_loop_calls
    )

    benchmark_calls = [
        node
        for node in ast.walk(performance)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_benchmark_cuda_graph_or_events"
    ]
    assert len(benchmark_calls) == 1
    benchmark_keywords = {
        keyword.arg: keyword.value for keyword in benchmark_calls[0].keywords
    }
    assert ast.literal_eval(benchmark_keywords["use_cuda_graph"]) is False
    assert ast.literal_eval(benchmark_keywords["fallback_reason"]) == (
        "dynamic_host_routing_requires_eager_execution"
    )

    routing_calls = [
        node
        for node in ast.walk(performance)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_make_routing_variants"
    ]
    assert len(routing_calls) == 1
    routing_keywords = {
        keyword.arg: keyword.value for keyword in routing_calls[0].keywords
    }
    assert ast.unparse(routing_keywords["count"]) == (
        "WARMUP_ITERATIONS + BENCHMARK_ITERATIONS"
    )
    bench_functions = [
        node
        for node in ast.walk(performance)
        if isinstance(node, ast.FunctionDef) and node.name == "_bench_fn"
    ]
    assert len(bench_functions) == 1
    bench_calls = [
        node for node in ast.walk(bench_functions[0]) if isinstance(node, ast.Call)
    ]
    copy_call = next(
        call
        for call in bench_calls
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "topk_ids"
        and call.func.attr == "copy_"
    )
    fused_call = next(
        call
        for call in bench_calls
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "mod"
        and call.func.attr == "fused_moe"
    )
    assert copy_call.lineno < fused_call.lineno


def test_device_timing_preferred_over_host_time(tmp_path):
    from src.testcases import parse_test_cases_from_json

    report = tmp_path / "performance_report.json"
    report.write_text(json.dumps([
        {
            "test_case_id": "shape0",
            "host_time_ms": 10.0,
            "device_time_ms": 1.25,
        }
    ]))

    cases = parse_test_cases_from_json(report)

    assert len(cases) == 1
    assert cases[0].execution_time_ms == 1.25
    assert cases[0].metadata["_timing_source"] == "device_time_ms"


def test_host_only_timing_is_rejected(tmp_path):
    from src.testcases import parse_test_cases_from_json

    report = tmp_path / "performance_report.json"
    report.write_text(json.dumps([
        {
            "test_case_id": "shape0",
            "host_time_ms": 10.0,
            "wall_time_ms": 11.0,
        }
    ]))

    assert parse_test_cases_from_json(report) == []


def test_host_only_single_object_timing_is_rejected(tmp_path):
    from src.testcases import parse_test_cases_from_json

    report = tmp_path / "performance_report.json"
    report.write_text(json.dumps({
        "host_time_ms": 10.0,
        "wall_time_ms": 11.0,
    }))

    assert parse_test_cases_from_json(report) == []


def test_harness_guard_rejects_harness_edits(tmp_path):
    from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    runner = scripts / "task_runner.py"
    runner.write_text("print('measure honestly')\n")

    snapshot = snapshot_workspace_harness(tmp_path)
    runner.write_text("print('fake a faster result')\n")

    with pytest.raises(RuntimeError, match="Protected test/harness files changed"):
        verify_workspace_harness(snapshot)


def test_harness_guard_rejects_harness_deletion(tmp_path):
    from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    runner = scripts / "task_runner.py"
    runner.write_text("print('measure honestly')\n")

    snapshot = snapshot_workspace_harness(tmp_path)
    runner.unlink()

    with pytest.raises(RuntimeError, match="deleted="):
        verify_workspace_harness(snapshot)


def test_harness_guard_allows_source_edits(tmp_path):
    from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness

    source = tmp_path / "source"
    source.mkdir()
    kernel = source / "kernel.py"
    kernel.write_text("def kernel(): pass\n")
    (tmp_path / "config.yaml").write_text("task_type: triton2triton\n")

    snapshot = snapshot_workspace_harness(tmp_path)
    kernel.write_text("def kernel(): return 1\n")

    verify_workspace_harness(snapshot)


def test_harness_guard_discards_agent_created_scratch_file(tmp_path):
    """A scratch file the agent invents cannot have influenced the baseline score.

    Real case: an agent spent 28 minutes optimizing, then left `dev/extra_test.py`
    behind. The name matched the global `*_test.py` rule and the whole run was thrown
    away. Deleting the file restores the measured state exactly, so the run survives.
    """
    from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness

    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.py").write_text("def kernel(): pass\n")

    snapshot = snapshot_workspace_harness(tmp_path)

    dev = tmp_path / "dev"
    dev.mkdir()
    scratch = dev / "extra_test.py"
    scratch.write_text("print('scratch sweep')\n")

    verify_workspace_harness(snapshot)

    assert not scratch.exists(), "the agent-created protected-pattern file must be removed"


def test_harness_guard_discards_added_file_but_still_rejects_real_tampering(tmp_path):
    from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    runner = scripts / "task_runner.py"
    runner.write_text("print('measure honestly')\n")

    snapshot = snapshot_workspace_harness(tmp_path)

    added = tmp_path / "sweep_test.py"
    added.write_text("print('scratch')\n")
    runner.write_text("print('fake a faster result')\n")

    with pytest.raises(RuntimeError) as excinfo:
        verify_workspace_harness(snapshot)

    assert "modified=" in str(excinfo.value)
    assert not added.exists()


def test_harness_guard_discards_file_added_inside_protected_dir(tmp_path):
    from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_existing.py").write_text("def test_a(): pass\n")

    snapshot = snapshot_workspace_harness(tmp_path)

    sneaky = tests_dir / "conftest_override.py"
    sneaky.write_text("def pytest_collection_modifyitems(items): items.clear()\n")

    verify_workspace_harness(snapshot)

    assert not sneaky.exists()
    assert (tests_dir / "test_existing.py").exists()


def test_harness_guard_logs_every_discard(tmp_path):
    from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness

    snapshot = snapshot_workspace_harness(tmp_path)
    (tmp_path / "scratch_test.py").write_text("print('x')\n")

    warnings = []

    class _Recorder:
        def warning(self, message):
            warnings.append(message)

    verify_workspace_harness(snapshot, logger=_Recorder())

    assert any("scratch_test.py" in w for w in warnings), \
        "a removal that leaves no trace is worse than a hard failure"
