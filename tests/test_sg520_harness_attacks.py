"""CPU regressions for report replay and source redirection admission gaps."""

import json
import os
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from src.harness_guard import (
    describe_workspace_harness,
    snapshot_workspace_harness,
    verify_task_source_aliases,
    verify_workspace_harness,
)
from src.performance import (
    clear_performance_report_files,
    measure_performance,
    performance_report_candidates,
)

SOURCE = "source/quant_kernels.cu"
PERF_CONFIG = {"task_type": "hip2hip", "performance_command": ["benchmark"]}
QUANT_TASK = Path(__file__).resolve().parents[1] / (
    "experimental/headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8"
)


@pytest.mark.parametrize("name", [
    "ut/oracle.py", "ut/native/include/quant.h", "cases.json", "provenance/SOURCE.json",
])
def test_sg520_task_rejects_hidden_reference_and_header_edits(tmp_path, name):
    workspace = tmp_path / "workspace"
    shutil.copytree(QUANT_TASK, workspace, symlinks=True)
    snapshot = snapshot_workspace_harness(workspace, task_root=QUANT_TASK)
    path = workspace / name
    path.write_bytes(path.read_bytes() + b"\nATTACK\n")
    with pytest.raises(RuntimeError, match="Protected test/harness files changed"):
        verify_workspace_harness(snapshot)


@pytest.mark.parametrize("inside_workspace", [False, True])
def test_sg520_task_rejects_source_swap_with_identical_bytes(tmp_path, inside_workspace):
    workspace = tmp_path / "workspace"
    shutil.copytree(QUANT_TASK, workspace, symlinks=True)
    snapshot = snapshot_workspace_harness(workspace, task_root=QUANT_TASK)
    assert snapshot.symlink_protected_sources == (SOURCE,)
    source = workspace / SOURCE
    target = (workspace if inside_workspace else tmp_path) / "replacement.cu"
    source.rename(target)
    source.symlink_to(target)
    with pytest.raises(RuntimeError, match="new source symlinks are forbidden"):
        verify_workspace_harness(snapshot)


def _source_task(tmp_path, *, protect=True, shipped_alias=False):
    task = tmp_path / "task"
    (task / "source").mkdir(parents=True)
    (task / SOURCE).write_text("__global__ void quant() {}\n")
    if shipped_alias:
        (task / SOURCE).rename(task / "implementation.cu")
        (task / SOURCE).symlink_to("../implementation.cu")
    (task / "config.yaml").write_text(yaml.safe_dump({
        "task_type": "hip2hip",
        "source_file_path": [SOURCE],
        "target_file_path": SOURCE,
        "target_kernel_functions": ["quant"],
        "harness_protection": {"reject_new_source_symlinks": protect},
    }))
    workspace = tmp_path / "workspace"
    shutil.copytree(task, workspace, symlinks=True)
    return task, workspace


@pytest.mark.parametrize("inside_workspace", [False, True])
@pytest.mark.parametrize("parent_link", [False, True])
@pytest.mark.parametrize("absolute_link", [False, True])
@pytest.mark.parametrize("before_snapshot", [False, True])
def test_opt_in_guard_rejects_new_source_links(
    tmp_path, inside_workspace, parent_link, absolute_link, before_snapshot
):
    task, workspace = _source_task(tmp_path)
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    destination = (workspace if inside_workspace else tmp_path) / "redirected"
    source = workspace / ("source" if parent_link else SOURCE)
    source.rename(destination)
    source.symlink_to(
        destination if absolute_link else os.path.relpath(destination, source.parent),
        target_is_directory=parent_link,
    )

    with pytest.raises(RuntimeError, match="new source symlinks are forbidden"):
        if before_snapshot:
            snapshot_workspace_harness(workspace, task_root=task)
        else:
            verify_workspace_harness(snapshot)
    with pytest.raises(RuntimeError, match="new source symlinks are forbidden"):
        verify_task_source_aliases(workspace, task)


@pytest.mark.parametrize("target", ["missing.cu", "quant_kernels.cu"])
def test_opt_in_guard_rejects_broken_or_cyclic_new_links(tmp_path, target):
    task, workspace = _source_task(tmp_path)
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    source = workspace / SOURCE
    source.unlink()
    source.symlink_to(target)
    with pytest.raises(RuntimeError, match="new source symlinks are forbidden"):
        verify_workspace_harness(snapshot)


def test_opt_in_guard_allows_regular_source_replacement(tmp_path):
    task, workspace = _source_task(tmp_path)
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    replacement = workspace / "replacement.cu"
    replacement.write_text("__global__ void quant() { /* optimized */ }\n")
    replacement.replace(workspace / SOURCE)
    verify_workspace_harness(snapshot)
    assert snapshot.symlink_protected_sources == (SOURCE,)
    assert describe_workspace_harness(workspace)["symlink_protected_sources"] == [SOURCE]


def test_opt_in_guard_preserves_shipped_alias_and_protects_its_target(tmp_path):
    task, workspace = _source_task(tmp_path, shipped_alias=True)
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    target = workspace / "implementation.cu"
    target.write_text("__global__ void quant() { /* optimized */ }\n")
    verify_workspace_harness(snapshot)
    verify_task_source_aliases(workspace, task)
    assert SOURCE in snapshot.source_aliases
    assert snapshot.symlink_protected_sources == ("implementation.cu",)

    # Returning to the same resolved file is insufficient: a new symlink has
    # replaced the canonical source node even though the shipped alias survives.
    target.rename(workspace / "replacement.cu")
    target.symlink_to("replacement.cu")
    with pytest.raises(RuntimeError, match="source alias|new source symlinks"):
        verify_workspace_harness(snapshot)


def test_source_link_policy_is_preserved_in_snapshot(tmp_path):
    task, workspace = _source_task(tmp_path)
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    (task / "config.yaml").write_text("{}\n")
    source = workspace / SOURCE
    source.unlink()
    source.symlink_to(task / SOURCE)
    with pytest.raises(RuntimeError, match="new source symlinks are forbidden"):
        verify_workspace_harness(snapshot)


def test_source_link_policy_remains_opt_in(tmp_path):
    task, workspace = _source_task(tmp_path, protect=False)
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    source = workspace / SOURCE
    source.unlink()
    source.symlink_to(task / SOURCE)
    verify_workspace_harness(snapshot)
    assert snapshot.symlink_protected_sources == ()


def _report(path, timing=0.000001):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{
        "test_case_id": "quant_case", "execution_time_ms": timing,
        "metadata": {"benchmark_method": "cuda_graph"},
    }]))


def test_cleanup_failure_cannot_replay_stale_performance(tmp_path, monkeypatch, caplog):
    report = tmp_path / "build/performance_report.json"
    _report(report)
    original_unlink = Path.unlink

    def refuse_report_removal(path, *args, **kwargs):
        if path == report:
            raise PermissionError("report removal denied")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_report_removal)
    command = Mock(return_value=(True, "", ""))
    monkeypatch.setattr("src.performance.run_command", command)

    assert measure_performance(tmp_path, PERF_CONFIG) == []
    command.assert_not_called()
    assert "Cannot establish fresh performance output" in caplog.text
    assert report.exists()


def test_report_directory_blocks_measurement_without_deleting_it(tmp_path, monkeypatch):
    report = tmp_path / "build/performance_report.json"
    report.mkdir(parents=True)
    command = Mock(return_value=(True, "Performance: 0.000001 ms", ""))
    monkeypatch.setattr("src.performance.run_command", command)

    assert measure_performance(tmp_path, PERF_CONFIG) == []
    command.assert_not_called()
    assert report.is_dir()


@pytest.mark.parametrize("target_exists", [False, True])
def test_cleanup_removes_report_symlink_without_touching_target(tmp_path, target_exists):
    report = tmp_path / "performance_report.json"
    target = tmp_path / "agent-output.json"
    if target_exists:
        _report(target)
    report.symlink_to(target.name)
    clear_performance_report_files(tmp_path)
    assert not report.is_symlink()
    assert target.exists() is target_exists


def test_empty_success_cannot_replay_old_reports(tmp_path, monkeypatch):
    reports = performance_report_candidates(tmp_path)
    for report in reports:
        _report(report)
    monkeypatch.setattr("src.performance.run_command", Mock(return_value=(True, "", "")))
    assert measure_performance(tmp_path, PERF_CONFIG) == []
    assert all(not report.exists() for report in reports)


@pytest.mark.parametrize("success", [False, True])
def test_fresh_report_is_scored_only_after_successful_command(tmp_path, monkeypatch, success):
    report = tmp_path / "build/performance_report.json"
    _report(report)

    def benchmark(*args, **kwargs):
        assert not report.exists()
        _report(report, timing=1.25)
        return success, "", "" if success else "benchmark failed"

    monkeypatch.setattr("src.performance.run_command", benchmark)
    cases = measure_performance(tmp_path, PERF_CONFIG)
    assert [(case.test_case_id, case.execution_time_ms) for case in cases] == (
        [("quant_case", 1.25)] if success else []
    )
