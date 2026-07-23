import json

import pytest


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
