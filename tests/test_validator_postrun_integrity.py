"""Exercise the real main/finalizer boundary without running a GPU or an agent."""
import json
import hashlib
import logging
from pathlib import Path
import shutil

import pytest
import yaml

import main
from agents.task_validator.report_schema import finalize_report, validation_report_is_complete
from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.module_registration import AgentType
from src.validator_integrity import begin_validator_integrity, finish_validator_integrity
from test_task_validator import _valid_raw_report


def task_files(root, *, freeze=True):
    (root / "scripts").mkdir(parents=True)
    (root / "ut").mkdir()
    (root / "source").mkdir()
    (root / "source/kernel.py").write_text("def kernel(x):\n    return x + 1\n")
    (root / "scripts/task_runner.py").write_text("print('checked')\n")
    (root / "ut/helper.py").write_text("VALUE = 3\n")
    (root / "ut/copy.py").write_text("VALUE = 3\n")
    config = {
        "task_type": "triton2triton", "source_file_path": ["source/kernel.py"],
        "target_kernel_functions": ["kernel"],
        "compile_command": ["python3 scripts/task_runner.py compile"],
        "correctness_command": ["python3 scripts/task_runner.py correctness"],
        "performance_command": ["python3 scripts/task_runner.py performance"],
        "harness_protection": {
            "freeze_task_inputs": freeze,
            "reject_new_source_symlinks": True,
            "editable_python_function_bodies": {"source/kernel.py": ["kernel"]},
            "runtime_outputs": ["ut/result.json", "ut/cache"],
        },
    }
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    return config


def write_report(workspace):
    (workspace / "validation_report.yaml").write_text(yaml.safe_dump(_valid_raw_report()))
    finalize_report(workspace, expected_task_name="triton2triton/example")


def run_validator(monkeypatch, tmp_path, mutation=None, *, alias=False, after_finalize=None):
    task = tmp_path / "task"
    task_files(task)
    if alias:
        (task / "ut/alias.py").symlink_to("helper.py")
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / "triton2triton_example_test"

    def setup(*args, **kwargs):
        shutil.copytree(task, workspace, symlinks=True)
        return workspace

    monkeypatch.setattr(main, "setup_workspace", setup)
    observed = {}

    def launcher(**kwargs):
        if mutation is not None:
            mutation(workspace)
        write_report(workspace)
        # A valid early launcher report is not yet a completed main audit.
        observed["pending_rejected"] = not validation_report_is_complete(workspace)
        if after_finalize is not None:
            after_finalize(workspace)

    completed, actual = main.run_task(
        eval_config={}, agent=AgentType.TASK_VALIDATOR, agent_launcher=launcher,
        task_name="triton2triton/example", task_config_dir=str(task / "config.yaml"),
        run_directory=run, timestamp="test", logger=logging.getLogger(__name__),
        task_index=1, total_tasks=1,
    )
    assert actual == workspace and observed["pending_rejected"]
    receipt_path = workspace / "validation_integrity.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
    return completed, workspace, receipt


@pytest.mark.parametrize("mutation", [
    "modify_support", "delete_support", "add_shadow", "add_helper",
    "retarget_support_alias", "replace_support_with_link", "add_source_alias",
    "add_dangling_alias", "source_body_edit", "delete_source", "modify_config",
])
def test_main_rejects_post_validator_mutations_without_discard(monkeypatch, tmp_path, mutation):
    def change(root):
        if mutation == "modify_support":
            (root / "ut/helper.py").write_text("VALUE = 999\n")
        elif mutation == "delete_support":
            (root / "ut/helper.py").unlink()
        elif mutation == "add_shadow":
            (root / "torch.py").write_text("# Newly shadowed import\n")
        elif mutation == "add_helper":
            (root / "ut/new_helper.py").write_text("VALUE = 9\n")
        elif mutation == "retarget_support_alias":
            (root / "ut/alias.py").unlink()
            (root / "ut/alias.py").symlink_to("copy.py")
        elif mutation == "replace_support_with_link":
            (root / "ut/helper.py").unlink()
            (root / "ut/helper.py").symlink_to("copy.py")
        elif mutation == "add_source_alias":
            (root / "ut/new_alias.py").symlink_to("../source/kernel.py")
        elif mutation == "add_dangling_alias":
            (root / "ut/new_alias.py").symlink_to("missing.py")
        elif mutation == "source_body_edit":
            (root / "source/kernel.py").write_text("def kernel(x):\n    return x + 2\n")
        elif mutation == "delete_source":
            (root / "source/kernel.py").unlink()
        else:
            path = root / "config.yaml"
            config = yaml.safe_load(path.read_text())
            config["harness_protection"]["freeze_task_inputs"] = False
            path.write_text(yaml.safe_dump(config))

    completed, workspace, receipt = run_validator(
        monkeypatch, tmp_path, change, alias=mutation == "retarget_support_alias"
    )
    assert not completed
    report = yaml.safe_load((workspace / "validation_report.yaml").read_text())
    assert report["overall_status"] == "FAIL"
    assert validation_report_is_complete(workspace)  # A finalized failed audit, never a PASS.
    assert receipt["status"] == "FAIL" and receipt["harness"]["files_discarded"] == []
    if mutation == "add_shadow":
        assert (workspace / "torch.py").is_file()
        assert "torch.py" in receipt["harness"]["added"]
    if mutation in {"add_helper", "add_source_alias", "add_dangling_alias"}:
        assert any(p.name.startswith("new_") for p in (workspace / "ut").iterdir())
    if mutation == "source_body_edit":
        assert receipt["harness"]["status"] == "PASS"  # The ordinary body mask is still truthful.
        assert receipt["raw_declared_sources_unchanged"] is False
    if receipt["harness"]["changed_paths"]:
        assert receipt["harness"]["after_digests"] is None
        assert receipt["after_source_fingerprint"] is None
    if mutation == "modify_config":
        assert "config.yaml" in receipt["harness"]["modified"]
        assert receipt["harness"]["after_digests"] is None
        assert receipt["harness"]["after_config_digests"]["config.yaml"] != receipt["harness"]["before_digests"]["config.yaml"]


def test_runtime_outputs_pass_and_receipt_is_portable_and_hash_bound(monkeypatch, tmp_path):
    def outputs(root):
        for relative in ["build/private/worker.json", "build/result.json", "ut/result.json", "ut/cache/generated.py", ".validator_audit/notes.json", "ut/__pycache__/helper.pyc"]:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")

    completed, workspace, receipt = run_validator(monkeypatch, tmp_path, outputs)
    assert completed and receipt["status"] == "PASS"
    assert receipt["raw_declared_sources_unchanged"] is True
    assert receipt["harness"]["before_digests"] == receipt["harness"]["after_digests"]
    assert receipt["harness"]["digest_modes"]["source/kernel.py"] == "masked_python_function_bodies_sha256"
    assert receipt["before"]["source_manifest"]["entries"][0]["sha256"]
    assert receipt["before"]["source_fingerprint"] == receipt["after_source_fingerprint"]
    assert validation_report_is_complete(workspace)
    archive = tmp_path / "archive" / "task"
    shutil.copytree(workspace, archive, symlinks=True)
    assert validation_report_is_complete(archive)
    exported = archive / "validation_integrity.json"
    exported.write_text(exported.read_text() + " ")
    assert not validation_report_is_complete(archive)


def test_pending_main_audit_cannot_be_resumed_or_archived_as_pass(tmp_path):
    root = tmp_path / "run" / "task"
    config = task_files(root)
    snapshot = snapshot_workspace_harness(root, task_root=root, strict=True)
    begin_validator_integrity(root, config, snapshot)
    write_report(root)
    assert not validation_report_is_complete(root)
    archive = tmp_path / "archive" / "task"
    shutil.copytree(root, archive)
    assert not validation_report_is_complete(archive)


@pytest.mark.parametrize("state", ["pending", "corrupt_receipt", "missing_receipt"])
def test_archived_required_integrity_cannot_be_downgraded_by_stripping_marker(tmp_path, state):
    root = tmp_path / "run" / "task"
    config = task_files(root)
    context = begin_validator_integrity(root, config, snapshot_workspace_harness(root, strict=True))
    write_report(root)
    if state != "pending":
        assert finish_validator_integrity(context, task_name="triton2triton/example")
    archive = tmp_path / "archive" / "task"
    shutil.copytree(root, archive, symlinks=True)
    assert not (archive.parent / ".validator-integrity").exists()
    if state != "pending":
        assert validation_report_is_complete(archive)
        receipt = archive / "validation_integrity.json"
        if state == "corrupt_receipt":
            receipt.write_text(receipt.read_text() + " ")
        else:
            receipt.unlink()
    assert not validation_report_is_complete(archive)
    report_path = archive / "validation_report.yaml"
    report_bytes = report_path.read_bytes()
    assert yaml.safe_load(report_bytes)["overall_status"] == "PASS"
    marker_path = archive / ".validation_complete"
    marker = yaml.safe_load(marker_path.read_text())
    original_hash = marker["report_sha256"]
    del marker["validator_integrity"]
    marker_path.write_text(yaml.safe_dump(marker))
    assert report_path.read_bytes() == report_bytes
    assert marker["report_sha256"] == original_hash == hashlib.sha256(report_bytes).hexdigest()
    assert not validation_report_is_complete(archive)


def test_direct_finalizer_without_integrity_lifecycle_remains_compatible(tmp_path):
    root = tmp_path / "run" / "task"
    task_files(root)
    write_report(root)
    report = yaml.safe_load((root / "validation_report.yaml").read_text())
    marker = yaml.safe_load((root / ".validation_complete").read_text())
    assert "validator_integrity" not in report and "validator_integrity" not in marker
    assert validation_report_is_complete(root)
    archive = tmp_path / "archive" / "task"
    shutil.copytree(root, archive)
    assert validation_report_is_complete(archive)


def test_archive_marker_and_receipt_cannot_substitute_report_attempt(monkeypatch, tmp_path):
    completed, workspace, _ = run_validator(monkeypatch, tmp_path)
    assert completed
    archive = tmp_path / "archive" / "task"
    shutil.copytree(workspace, archive)
    report_path = archive / "validation_report.yaml"
    report_bytes = report_path.read_bytes()
    report = yaml.safe_load(report_bytes)
    assert report["validator_integrity"]["required"] is True
    receipt_path = archive / "validation_integrity.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["attempt"] = "substituted-attempt"
    receipt_path.write_text(json.dumps(receipt))
    marker_path = archive / ".validation_complete"
    marker = yaml.safe_load(marker_path.read_text())
    marker["validator_integrity"]["attempt"] = receipt["attempt"]
    marker["validator_integrity"]["receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    marker_path.write_text(yaml.safe_dump(marker))
    assert report_path.read_bytes() == report_bytes
    assert not validation_report_is_complete(archive)


def test_renormalizing_archive_preserves_required_lifecycle(monkeypatch, tmp_path):
    completed, workspace, _ = run_validator(monkeypatch, tmp_path)
    assert completed
    archive = tmp_path / "archive" / "task"
    shutil.copytree(workspace, archive)
    before = yaml.safe_load((archive / "validation_report.yaml").read_text())
    after = finalize_report(archive, expected_task_name="triton2triton/example")
    assert after["validator_integrity"] == before["validator_integrity"]
    assert not validation_report_is_complete(archive)


def test_report_written_before_late_mutation_is_rejected(monkeypatch, tmp_path):
    def late_mutation(root):
        (root / "ut/late_helper.py").write_text("VALUE = 99\n")

    completed, workspace, receipt = run_validator(monkeypatch, tmp_path, after_finalize=late_mutation)
    assert not completed and receipt["status"] == "FAIL"
    assert "ut/late_helper.py" in receipt["harness"]["added"]
    assert yaml.safe_load((workspace / "validation_report.yaml").read_text())["overall_status"] == "FAIL"


def test_launcher_exception_after_report_cannot_leave_pass(monkeypatch, tmp_path):
    def fail_after_report(root):
        raise RuntimeError("late worker failed")

    completed, workspace, receipt = run_validator(monkeypatch, tmp_path, after_finalize=fail_after_report)
    assert not completed and receipt["harness"]["status"] == "PASS"
    assert "late worker failed" in receipt["launcher_error"]
    assert validation_report_is_complete(workspace)
    assert yaml.safe_load((workspace / "validation_report.yaml").read_text())["overall_status"] == "FAIL"


def test_missing_before_record_preserves_observed_failure_receipt(monkeypatch, tmp_path):
    def remove_before(root):
        control = root.parent / ".validator-integrity" / root.name
        active = json.loads((control / "active.json").read_text())
        (control / active["attempt"] / "before.json").unlink()

    completed, workspace, receipt = run_validator(monkeypatch, tmp_path, after_finalize=remove_before)
    assert not completed and receipt["status"] == "FAIL"
    assert receipt["before"]["harness_digests"] == receipt["harness"]["before_digests"]
    assert any("pre-agent integrity record is unavailable" in error for error in receipt["errors"])
    assert validation_report_is_complete(workspace)


def test_export_failure_does_not_rerun_task_helpers_or_complete(monkeypatch, tmp_path):
    import src.validator_integrity as integrity
    import src.harness_guard as guard

    atomic_json = integrity._atomic_json

    def fail_export(path, value):
        if path.name == integrity.EXPORT_NAME:
            raise OSError("export unavailable")
        atomic_json(path, value)

    def forbid_late_helpers(root):
        def unexpected(*args, **kwargs):
            pytest.fail("task-controlled helper path reran after integrity failure")
        monkeypatch.setattr(guard, "describe_workspace_harness", unexpected)

    monkeypatch.setattr(integrity, "_atomic_json", fail_export)
    completed, workspace, receipt = run_validator(monkeypatch, tmp_path, after_finalize=forbid_late_helpers)
    assert not completed and receipt is None
    assert not validation_report_is_complete(workspace)
    assert yaml.safe_load((workspace / "validation_report.yaml").read_text())["overall_status"] == "FAIL"
    control = workspace.parent / ".validator-integrity" / workspace.name
    active = json.loads((control / "active.json").read_text())
    assert active["status"] == "PENDING"
    assert (control / active["attempt"] / "result.json").is_file()


def test_optimization_still_allows_body_edits_and_discards_new_harness_files(tmp_path):
    root = tmp_path / "task"
    task_files(root)
    snapshot = snapshot_workspace_harness(root, task_root=root)
    (root / "source/kernel.py").write_text("def kernel(x):\n    return x + 2\n")
    scratch = root / "scratch_test.py"
    scratch.write_text("# Scratch before centralized evaluation\n")
    verify_workspace_harness(snapshot)
    assert not scratch.exists()


def test_raw_source_freeze_does_not_expand_non_opted_in_task_behavior(tmp_path):
    root = tmp_path / "run" / "task"
    config = task_files(root, freeze=False)
    snapshot = snapshot_workspace_harness(root, task_root=root, strict=True)
    context = begin_validator_integrity(root, config, snapshot)
    (root / "source/kernel.py").write_text("def kernel(x):\n    return x + 2\n")
    write_report(root)
    assert finish_validator_integrity(context, task_name="triton2triton/example")
    receipt = json.loads((root / "validation_integrity.json").read_text())
    assert receipt["before"]["raw_declared_source_freeze"] is False
    assert receipt["raw_declared_sources_unchanged"] is None


def test_declared_source_freeze_uses_existing_repository_path_resolution(tmp_path):
    root = tmp_path / "run" / "task"
    root.mkdir(parents=True)
    (root / "vendor").mkdir()
    (root / "vendor/kernel.cu").write_text("__global__ void kernel() {}\n")
    config = {"repo_subdir": "vendor", "source_file_path": ["kernel.cu"],
              "harness_protection": {"freeze_task_inputs": True}}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    context = begin_validator_integrity(root, config, snapshot_workspace_harness(root, task_root=root, strict=True))
    assert context.sources.manifest["entries"][0]["workspace_relative_path"] == "vendor/kernel.cu"
    (root / "vendor/kernel.cu").write_text("__global__ void kernel() { changed(); }\n")
    write_report(root)
    assert not finish_validator_integrity(context, task_name="triton2triton/example")


def test_trusted_evaluation_freezes_declared_sources_without_freeze_policy(tmp_path):
    from test_trusted_harness_coverage import make_task

    root = make_task(tmp_path / "run" / "task")
    config = yaml.safe_load((root / "config.yaml").read_text())
    context = begin_validator_integrity(root, config, snapshot_workspace_harness(root, strict=True))
    assert context.sources is not None
    (root / "source/kernel.py").write_text("def kernel():\n    return 999\n")
    write_report(root)
    assert not finish_validator_integrity(context, task_name="triton2triton/example")


def test_shipped_directory_alias_is_observed_and_retargeting_rejected(tmp_path):
    root = tmp_path / "run" / "task"
    config = task_files(root)
    (root / "ut/source_alias").symlink_to("../source", target_is_directory=True)
    snapshot = snapshot_workspace_harness(root, task_root=root, strict=True)
    assert verify_workspace_harness(snapshot, reject_added=True)["status"] == "PASS"
    context = begin_validator_integrity(root, config, snapshot)
    (root / "ut/source_alias").unlink()
    (root / "ut/source_alias").symlink_to(".", target_is_directory=True)
    write_report(root)
    assert not finish_validator_integrity(context, task_name="triton2triton/example")
    receipt = json.loads((root / "validation_integrity.json").read_text())
    assert "ut/source_alias" in receipt["harness"]["changed_paths"]
    assert receipt["harness"]["after_digests"] is None


def test_real_rmsnorm_aliases_and_runtime_outputs_pass_strict_snapshot(tmp_path):
    from src.perf_helper_materialization import materialize_perf_helpers_in_workspace
    from test_qwen_rmsnorm_harness_guard import TASK

    root = tmp_path / "run" / "task"
    shutil.copytree(TASK, root, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "build"))
    materialize_perf_helpers_in_workspace(root)
    config = yaml.safe_load((root / "config.yaml").read_text())
    context = begin_validator_integrity(root, config, snapshot_workspace_harness(root, task_root=TASK, strict=True))
    output = root / "build/private/worker.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}\n")
    # No task commands run here. Only the real package's integrity boundary is
    # tested; absent GPU quality evidence cannot become a task PASS.
    assert finish_validator_integrity(context, task_name=TASK.name)
    receipt = json.loads((root / "validation_integrity.json").read_text())
    assert receipt["status"] == "PASS" and receipt["raw_declared_sources_unchanged"] is True
    assert not validation_report_is_complete(root)
