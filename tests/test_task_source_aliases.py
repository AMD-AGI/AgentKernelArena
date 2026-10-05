"""Task copies must keep harness imports connected to declared editable sources."""

import logging
import runpy
import shutil
from unittest.mock import patch

import pytest
import yaml

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.preprocessing import setup_workspace


LOG = logging.getLogger(__name__)
SOURCE = "source/kernel.py"
ALIAS = "ut/kernel_src/kernel.py"


def _task(tmp_path, *, declaration=None):
    task = tmp_path / "task"
    (task / "source").mkdir(parents=True)
    (task / "ut/kernel_src").mkdir(parents=True)
    (task / SOURCE).write_text("def kernel(): return 1\n")
    (task / "source/support.py").write_text("VALUE = 1\n")
    (task / ALIAS).symlink_to("../../source/kernel.py")
    (task / "ut/kernel_src/support.py").symlink_to("../../source/support.py")
    (task / "entry.py").symlink_to(SOURCE)
    (task / "config.yaml").write_text(yaml.safe_dump({
        "task_type": "triton2triton",
        "source_file_path": [SOURCE] if declaration is None else declaration,
    }))
    return task


def _setup(task, tmp_path):
    return setup_workspace(str(task / "config.yaml"), tmp_path / "run", "test", LOG)


@pytest.mark.parametrize("declaration", [SOURCE, [SOURCE]])
def test_workspace_source_edits_reach_preserved_aliases(tmp_path, declaration):
    task = _task(tmp_path, declaration=declaration)
    workspace = _setup(task, tmp_path)
    # A resumed setup must also preserve the links it previously created.
    assert _setup(task, tmp_path) == workspace
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    (workspace / SOURCE).write_text("def kernel(): return 2\n")

    verify_workspace_harness(snapshot)
    for name in (ALIAS, "entry.py"):
        assert (workspace / name).is_symlink()
        assert runpy.run_path(str(workspace / name))["kernel"]() == 2
    assert runpy.run_path(str(task / SOURCE))["kernel"]() == 1


def test_snapshot_rejects_old_dereferencing_copy_even_with_identical_bytes(tmp_path):
    task = _task(tmp_path)
    workspace = tmp_path / "old_workspace"
    shutil.copytree(task, workspace)
    with pytest.raises(RuntimeError, match="source alias"):
        snapshot_workspace_harness(workspace, task_root=task)


@pytest.mark.parametrize("change", ["delete", "regular_file", "redirect", "absolute", "dangling"])
@pytest.mark.parametrize("before_snapshot", [False, True])
def test_broken_or_redirected_alias_is_rejected(tmp_path, change, before_snapshot):
    task = _task(tmp_path)
    workspace = _setup(task, tmp_path)
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    alias = workspace / ALIAS
    content = alias.read_text()
    alias.unlink()
    if change == "regular_file":
        alias.write_text(content)
    elif change == "redirect":
        (workspace / "stale.py").write_text(content)
        alias.symlink_to("../../stale.py")
    elif change == "absolute":
        alias.symlink_to(workspace / SOURCE)
    elif change == "dangling":
        alias.symlink_to("../../missing.py")

    with pytest.raises(RuntimeError, match="source alias"):
        if before_snapshot:
            snapshot_workspace_harness(workspace, task_root=task)
        else:
            verify_workspace_harness(snapshot)


def test_parent_redirection_cannot_preserve_apparent_alias_text(tmp_path):
    task = _task(tmp_path)
    workspace = _setup(task, tmp_path)
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    (workspace / "source").rename(tmp_path / "outside")
    (workspace / "source").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(RuntimeError, match="source alias"):
        verify_workspace_harness(snapshot)


def test_snapshot_retains_alias_contract_if_original_task_later_changes(tmp_path):
    task = _task(tmp_path)
    workspace = _setup(task, tmp_path)
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    (task / ALIAS).unlink()
    (workspace / ALIAS).unlink()
    (workspace / ALIAS).write_text((workspace / SOURCE).read_text())
    with pytest.raises(RuntimeError, match="source alias"):
        verify_workspace_harness(snapshot)


def test_declared_source_can_itself_be_a_contained_symlink(tmp_path):
    task = _task(tmp_path)
    (task / SOURCE).rename(task / "implementation.py")
    (task / SOURCE).symlink_to("../implementation.py")
    workspace = _setup(task, tmp_path)
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    (workspace / "implementation.py").write_text("def kernel(): return 2\n")
    verify_workspace_harness(snapshot)
    assert runpy.run_path(str(workspace / ALIAS))["kernel"]() == 2
    assert SOURCE in snapshot.source_aliases


def test_undeclared_source_alias_still_has_content_protection(tmp_path):
    task = _task(tmp_path)
    workspace = _setup(task, tmp_path)
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    assert "ut/kernel_src/support.py" in snapshot.digests
    (workspace / "source/support.py").write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="support.py"):
        verify_workspace_harness(snapshot)


def test_contained_directory_alias_remains_connected_and_guarded(tmp_path):
    task = _task(tmp_path)
    (task / "imports").symlink_to("source", target_is_directory=True)
    workspace = _setup(task, tmp_path)
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    (workspace / SOURCE).write_text("def kernel(): return 2\n")
    verify_workspace_harness(snapshot)
    assert (workspace / "imports").is_symlink()
    assert runpy.run_path(str(workspace / "imports/kernel.py"))["kernel"]() == 2
    (workspace / "imports").unlink()
    shutil.copytree(workspace / "source", workspace / "imports")
    with pytest.raises(RuntimeError, match="source alias"):
        verify_workspace_harness(snapshot)


@pytest.mark.parametrize("target", ["absolute", "escape", "dangling", "cycle"])
def test_setup_rejects_nonrelocatable_or_broken_task_links(tmp_path, target):
    task = _task(tmp_path)
    link = task / "unsafe.py"
    if target == "absolute":
        link.symlink_to(task / SOURCE)
    elif target == "escape":
        (tmp_path / "outside.py").write_text("VALUE = 1\n")
        link.symlink_to("../outside.py")
    elif target == "cycle":
        link.symlink_to("unsafe.py")
    else:
        link.symlink_to("missing.py")
    with pytest.raises(ValueError, match="[Tt]ask symlink"):
        _setup(task, tmp_path)
    assert not (tmp_path / "run/task_test").exists()


def test_setup_does_not_silently_rebind_a_detached_existing_alias(tmp_path):
    task = _task(tmp_path)
    workspace = _setup(task, tmp_path)
    alias = workspace / ALIAS
    alias.unlink()
    alias.write_text("def kernel(): return 9\n")
    with pytest.raises(ValueError, match="Existing workspace task symlink differs"):
        _setup(task, tmp_path)
    assert runpy.run_path(str(alias))["kernel"]() == 9


def test_setup_ignores_stale_image_repo_cache_when_validating_links(tmp_path):
    task = _task(tmp_path)
    (task / "vendor").mkdir()
    (task / "vendor/stale.py").symlink_to("/unavailable/old-cache.py")
    config_path = task / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config.update(task_type="image_kernel", image_repo_path="/image/vendor", repo_subdir="vendor")
    config_path.write_text(yaml.safe_dump(config))
    with patch("src.preprocessing._ensure_repo_seeded_from_image", return_value=False):
        workspace = _setup(task, tmp_path)
    assert not (workspace / "vendor").exists()
    assert (workspace / ALIAS).is_symlink()


def test_materialization_copy_preserves_source_aliases(tmp_path):
    from src.tools.materialize_perf_helpers import _copy_task

    task = _task(tmp_path)
    workspace = _copy_task(task, tmp_path / "inspection", False)
    assert (workspace / ALIAS).is_symlink()
    snapshot_workspace_harness(workspace, task_root=task)


def test_quality_loop_restore_and_dual_gate_keep_each_source_connected(tmp_path):
    from agents.quality_loop.orchestrator import QualityLoop

    task = _task(tmp_path)
    candidate = tmp_path / "candidate"
    QualityLoop._replace_directory(candidate, task)
    (candidate / SOURCE).write_text("def kernel(): return 2\n")
    loop = QualityLoop.__new__(QualityLoop)
    loop.logger = LOG
    seen = []

    def check(workspace, *args):
        seen.append(runpy.run_path(str(workspace / ALIAS))["kernel"]())
        return True, None

    with (
        patch("agents.quality_loop.orchestrator.evaluate_compilation", return_value=(True, None)),
        patch("agents.quality_loop.orchestrator.evaluate_correctness", side_effect=check),
    ):
        assert loop._dual_correctness_gate("example", task, candidate, tmp_path / "gate")
    assert seen == [1, 2]
    assert (candidate / ALIAS).is_symlink()


def test_heldout_restoration_reaches_original_alias_only(tmp_path):
    from src.held_out.run_heldout_eval import evaluate_single_task

    task = _task(tmp_path)
    workspace = _setup(task, tmp_path)
    (workspace / SOURCE).write_text("def kernel(): return 2\n")
    seen = []

    def check(copied_workspace, *args):
        seen.append(runpy.run_path(str(copied_workspace / ALIAS))["kernel"]())
        return True, None

    with (
        patch("src.held_out.run_heldout_eval.apply_all_injections", return_value=True),
        patch("src.held_out.run_heldout_eval.evaluate_compilation", return_value=(True, None)),
        patch("src.held_out.run_heldout_eval.evaluate_correctness", side_effect=check),
        patch("src.held_out.run_heldout_eval.measure_baseline", return_value=[]),
        patch("src.held_out.run_heldout_eval.measure_performance", return_value=[]),
    ):
        evaluate_single_task(workspace, tmp_path / "heldout", {}, task, LOG)
    assert seen == [1, 2]
    assert runpy.run_path(str(workspace / ALIAS))["kernel"]() == 2


def test_heldout_rejects_detached_alias_before_creating_output(tmp_path):
    from src.held_out.run_heldout_eval import evaluate_single_task

    task = _task(tmp_path)
    workspace = tmp_path / "old_workspace"
    shutil.copytree(task, workspace)
    output = tmp_path / "heldout"
    with pytest.raises(RuntimeError, match="source alias"):
        evaluate_single_task(workspace, output, {}, task, LOG)
    assert not output.exists()
