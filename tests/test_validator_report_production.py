"""Report-production checks execute only the data-only canonical CLI."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import yaml

from agents.task_validator.report_schema import (
    MAX_REPORT_BYTES,
    check_draft_report,
    finalize_report,
    validation_report_is_complete,
)
from agents.task_validator.validation_prompt import build_validation_prompt
from test_task_validator import _valid_raw_report


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "agents/task_validator/report_schema.py"


def cli(path, task="hip2hip/example"):
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--check-draft", str(path), "--task-name", task],
        text=True, capture_output=True, check=False, timeout=5,
    )
    return result.returncode, json.loads(result.stdout)


@pytest.mark.parametrize("malformed", ["list", "missing", "unknown", "duplicate_check", "duplicate_status", "invalid_status"])
def test_checker_and_finalizer_reject_malformed_reports_and_preserve_exact_draft(tmp_path, malformed):
    raw = _valid_raw_report()
    if malformed == "list":
        raw["checks"] = [{"name": key, **value} for key, value in raw["checks"].items()]
    elif malformed == "missing":
        del raw["checks"]["correctness"]
    elif malformed == "unknown":
        raw["checks"]["unknown_check"] = {"status": "PASS"}
    elif malformed == "invalid_status":
        raw["checks"]["result_template_compatibility"]["status"] = "SKIP"
        raw["checks"]["result_template_compatibility"]["skip_reason_code"] = "repository_field_not_declared"
    content = yaml.safe_dump(raw, sort_keys=False)
    if malformed == "duplicate_check":
        content = content.replace("  correctness:\n", "  correctness: {status: FAIL}\n  correctness:\n")
    elif malformed == "duplicate_status":
        content = content.replace("    status: PASS\n", "    status: FAIL\n    status: PASS\n", 1)
    path = tmp_path / "validation_report.yaml"
    path.write_text(content)
    before = path.read_bytes()
    code, result = cli(path)
    assert code != 0 and result["schema_status"] == "FAIL" and result["errors"]
    assert path.read_bytes() == before and not (tmp_path / ".validation_complete").exists()
    report = finalize_report(tmp_path, expected_task_name="hip2hip/example")
    assert report["overall_status"] == "FAIL"
    assert report["submitted_report"]["raw_yaml"].encode("utf-8") == before
    assert report["submitted_report"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert validation_report_is_complete(tmp_path)  # A finalized failure, never PASS.


def test_valid_failed_task_passes_format_check_without_changing_findings(tmp_path):
    raw = _valid_raw_report()
    raw["checks"]["correctness"]["status"] = "FAIL"
    raw["checks"]["correctness"]["attempts"][0]["exit_code"] = 1
    for name in ("performance", "benchmark_integrity"):
        raw["checks"][name] = {"status": "SKIP", "skip_reason_code": "dependency_failed", "details": "Correctness failed"}
    raw["overall_status"] = "FAIL"
    path = tmp_path / "validation_report.yaml"
    path.write_text(yaml.safe_dump(raw))
    before = path.read_bytes()
    code, result = cli(path)
    assert code == 0 and result["schema_status"] == "PASS"
    assert result["projected_task_status"] == "FAIL"
    assert path.read_bytes() == before
    assert not (tmp_path / ".validation_complete").exists()


def test_producer_uses_prompt_finish_command_to_correct_only_structure(tmp_path):
    task = tmp_path / "task with spaces"
    task.mkdir()
    config = task / "config.yaml"
    config.write_text("task_type: hip2hip\nsource_file_path: [kernel.hip]\n")
    prompt = build_validation_prompt(str(config), str(task), {"agent": {"python_path": sys.executable}})
    finish = prompt.split("## Required final report-format check", 1)[1]
    command = shlex.split(finish.split("```bash\n", 1)[1].split("\n```", 1)[0])
    assert command[:3] == [sys.executable, str(CHECKER), "--check-draft"]
    assert "Do not write a list of named checks" in finish
    raw = _valid_raw_report(task_name=task.name)
    original = {"task": (task / "kernel.hip"), "command": (task / "command-result.json")}
    original["task"].write_text("// Original GPU source; never executed by this test\n")
    original["command"].write_text('{"exit_code": 0, "measurements": [1.0, 2.0]}\n')
    preserved = {key: path.read_bytes() for key, path in original.items()}
    report_path = task / "validation_report.yaml"
    draft = dict(raw, checks=[{"name": key, **value} for key, value in raw["checks"].items()])
    report_path.write_text(yaml.safe_dump(draft))
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode == 1 and "checks must be a mapping" in rejected.stdout
    # The producer fixes its document structure from the same observed data.
    report_path.write_text(yaml.safe_dump(raw))
    checked = subprocess.run(command, capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout
    assert all(path.read_bytes() == preserved[key] for key, path in original.items())
    assert not (task / ".validation_complete").exists()
    assert finalize_report(task, expected_task_name=task.name)["overall_status"] == "PASS"


def test_completion_rejects_duplicate_mapping_even_with_matching_report_hash(tmp_path):
    path = tmp_path / "validation_report.yaml"
    path.write_text(yaml.safe_dump(_valid_raw_report()))
    finalize_report(tmp_path, expected_task_name="hip2hip/example")
    path.write_text(path.read_text().replace("overall_status: PASS", "overall_status: FAIL\noverall_status: PASS", 1))
    marker_path = tmp_path / ".validation_complete"
    marker = yaml.safe_load(marker_path.read_text())
    marker["report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    marker_path.write_text(yaml.safe_dump(marker))
    assert not validation_report_is_complete(tmp_path)


def test_checker_missing_report_does_not_create_evidence(tmp_path):
    result = check_draft_report(tmp_path / "missing.yaml", expected_task_name="hip2hip/example")
    assert result["schema_status"] == "FAIL" and result["report_sha256"] is None
    assert list(tmp_path.iterdir()) == []


def test_report_symlink_never_embeds_external_sentinel(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external.yaml"
    sentinel = b"PRIVATE_EXTERNAL_SENTINEL_927\n"
    external.write_bytes(sentinel)
    path = workspace / "validation_report.yaml"
    path.symlink_to(external)
    code, result = cli(path)
    assert code == 1 and result["report_sha256"] is None
    report = finalize_report(workspace, expected_task_name="hip2hip/example")
    assert report["overall_status"] == "FAIL" and "submitted_report" not in report
    assert sentinel not in path.read_bytes()
    assert external.read_bytes() == sentinel


@pytest.mark.parametrize("kind", ["fifo", "oversized"])
def test_nonregular_or_oversized_report_rejected_without_preserving_bytes(tmp_path, kind):
    path = tmp_path / "validation_report.yaml"
    if kind == "fifo":
        os.mkfifo(path)
    else:
        with path.open("wb") as stream:
            stream.truncate(MAX_REPORT_BYTES + 1)
    code, result = cli(path)
    assert code == 1 and result["report_sha256"] is None
    report = finalize_report(tmp_path, expected_task_name="hip2hip/example")
    assert report["overall_status"] == "FAIL" and "submitted_report" not in report


@pytest.mark.parametrize("parent_kind", ["symlink", "traversal"])
def test_unsafe_parent_is_refused_without_touching_external_report(tmp_path, parent_kind):
    outside = tmp_path / "outside"
    outside.mkdir()
    report = outside / "validation_report.yaml"
    sentinel = b"PRIVATE_PARENT_TARGET_491\n"
    report.write_bytes(sentinel)
    (outside / ".validation_complete").write_text("original marker")
    workspace = tmp_path / "workspace"
    if parent_kind == "symlink":
        workspace.symlink_to(outside, target_is_directory=True)
    else:
        workspace.mkdir()
        workspace = workspace / ".." / "outside"
    code, result = cli(workspace / "validation_report.yaml")
    assert code == 1 and result["report_sha256"] is None
    with pytest.raises((OSError, ValueError)):
        finalize_report(workspace, expected_task_name="hip2hip/example")
    assert report.read_bytes() == sentinel
    assert (outside / ".validation_complete").read_text() == "original marker"


@pytest.mark.parametrize("replacement", ["file", "parent"])
def test_replacement_during_read_cannot_supply_embedded_target_bytes(tmp_path, monkeypatch, replacement):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path = workspace / "validation_report.yaml"
    path.write_text(yaml.safe_dump(_valid_raw_report()))
    original_inode = path.stat().st_ino
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = b"PRIVATE_REPLACEMENT_TARGET_611\n"
    (outside / "validation_report.yaml").write_bytes(sentinel)
    original_read = os.read
    replaced = False

    def replace_after_read(descriptor, count):
        nonlocal replaced
        data = original_read(descriptor, count)
        if not replaced and os.fstat(descriptor).st_ino == original_inode:
            replaced = True
            if replacement == "file":
                path.unlink()
                path.symlink_to(outside / "validation_report.yaml")
            else:
                workspace.rename(tmp_path / "original-workspace")
                workspace.symlink_to(outside, target_is_directory=True)
        return data

    monkeypatch.setattr(os, "read", replace_after_read)
    if replacement == "parent":
        with pytest.raises(OSError):
            finalize_report(workspace, expected_task_name="hip2hip/example")
    else:
        report = finalize_report(workspace, expected_task_name="hip2hip/example")
        assert report["overall_status"] == "FAIL" and "submitted_report" not in report
        assert sentinel not in path.read_bytes()
    assert replaced and (outside / "validation_report.yaml").read_bytes() == sentinel
