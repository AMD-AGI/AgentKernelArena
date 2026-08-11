import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_grouped_gemm_runner():
    path = (
        Path(__file__).resolve().parents[1]
        / "tasks/image_kernel/mi355x_sglang_triton_mxfp8_grouped_gemm"
        / "scripts/task_runner.py"
    )
    spec = importlib.util.spec_from_file_location("_test_grouped_gemm_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_grouped_gemm_timed_validation_checks_both_outputs(monkeypatch):
    import torch

    runner = _load_grouped_gemm_runner()
    ref1 = torch.ones(2, 3)
    ref2 = torch.ones(2, 4)
    out1 = torch.zeros_like(ref1)
    out2 = torch.zeros_like(ref2)
    timed = runner._TimedRun()

    def _rerun_with_corrupt_gemm1():
        out2.copy_(ref2)
        return out1, out2

    timed._bind(_rerun_with_corrupt_gemm1, (out1, out2))
    monkeypatch.setattr(
        runner,
        "_timed_references",
        lambda _inputs: (ref1, ref2),
    )
    inputs = {"cfg": {"id": "shape0", "params": {"max_relerr": 0.08}}}

    with pytest.raises(AssertionError, match="timed_gemm1"):
        runner._assert_timed_outputs(inputs, timed)


def test_grouped_gemm_timed_run_rejects_event_fallback(monkeypatch):
    import torch

    runner = _load_grouped_gemm_runner()
    from src.tools.perf.aka_benchmark import benchmark_cuda_graph_or_events

    monkeypatch.setattr(
        runner, "_benchmark_cuda_graph_or_events", benchmark_cuda_graph_or_events
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def _reject_stream():
        raise RuntimeError("capture unavailable")

    monkeypatch.setattr(torch.cuda, "Stream", _reject_stream)

    with pytest.raises(RuntimeError, match="timed_run cannot validate"):
        runner._benchmark_cuda_graph(
            lambda: None,
            warmup=0,
            repetition=1,
            timed_run=runner._TimedRun(),
        )


def test_timed_run_fails_closed_without_cuda(monkeypatch):
    import torch

    from src.tools.perf.vllm_cuda_graph_block import (
        _TimedRun,
        _benchmark_cuda_graph_or_events,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="requires an observable CUDA-graph replay"):
        _benchmark_cuda_graph_or_events(
            lambda: None,
            warmup=0,
            repetition=1,
            timed_run=_TimedRun(),
        )


def test_timed_run_fails_closed_when_cuda_graph_is_disabled(monkeypatch):
    import torch

    from src.tools.perf.vllm_cuda_graph_block import (
        _TimedRun,
        _benchmark_cuda_graph_or_events,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    with pytest.raises(RuntimeError, match="CUDA-graph timing is disabled"):
        _benchmark_cuda_graph_or_events(
            lambda: None,
            warmup=0,
            repetition=1,
            use_cuda_graph=False,
            timed_run=_TimedRun(),
        )


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
