"""Copy/restore operations must not write through links into another workspace."""

import logging
import os
import runpy
import shutil
from unittest.mock import patch

import pytest
import yaml

from agents.quality_loop.filesystem import (
    TreeChanges,
    apply_changes,
    diff_trees,
    snapshot_tree,
)
from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.held_out.run_heldout_eval import evaluate_single_task


LOG = logging.getLogger(__name__)
SOURCE = "source/kernel.py"
BASELINE = "def kernel(): return 1\n"
CANDIDATE = "def kernel(): return 2\n"


def _plain_task(tmp_path, *, declaration=None):
    task = tmp_path / "task"
    (task / "source").mkdir(parents=True)
    (task / SOURCE).write_text(BASELINE)
    (task / "config.yaml").write_text(yaml.safe_dump({
        "task_type": "triton2triton",
        "source_file_path": [SOURCE] if declaration is None else declaration,
    }))
    return task


@pytest.mark.parametrize("parent_link", [False, True])
@pytest.mark.parametrize("absolute", [False, True])
def test_heldout_rejects_new_links_before_any_kernel_restoration(
    tmp_path, parent_link, absolute
):
    task = _plain_task(tmp_path, declaration=["source/first.py", SOURCE])
    (task / "source/first.py").write_text("FIRST = 1\n")
    workspace = tmp_path / "submitted"
    shutil.copytree(task, workspace)
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    assert not snapshot.source_aliases
    (workspace / "source/first.py").write_text("FIRST = 2\n")
    (workspace / SOURCE).write_text(CANDIDATE)

    if parent_link:
        target = workspace / "implementation" if absolute else tmp_path / "external"
        (workspace / "source").rename(target)
        link = workspace / "source"
        helper = target / "kernel.py"
    else:
        helper = workspace / "implementation.py" if absolute else tmp_path / "external.py"
        helper.write_text(CANDIDATE)
        link = workspace / SOURCE
        link.unlink()
        target = helper
    link.symlink_to(target if absolute else os.path.relpath(target, link.parent))
    # Canonical-alias verification alone does not cover this optimizer-created link.
    verify_workspace_harness(snapshot)

    output = tmp_path / "heldout"
    with (
        patch("src.held_out.run_heldout_eval.apply_all_injections") as inject,
        patch("src.held_out.run_heldout_eval.evaluate_compilation") as compile_kernel,
        pytest.raises(ValueError, match="Unsafe held-out kernel restoration path"),
    ):
        evaluate_single_task(workspace, output, {}, task, LOG)
    inject.assert_not_called()
    compile_kernel.assert_not_called()
    assert helper.read_text() == CANDIDATE
    assert (workspace / "source/first.py").read_text() == "FIRST = 2\n"
    # In the leaf-link case this is a private regular file preceding the bad
    # destination: it must not be partially restored before validation fails.
    if not parent_link:
        assert (output / "orig/source/first.py").read_text() == "FIRST = 2\n"


@pytest.mark.parametrize("declaration", [SOURCE, [SOURCE]])
def test_heldout_restores_a_relative_declared_source_alias_only_in_orig(
    tmp_path, declaration
):
    task = _plain_task(tmp_path, declaration=declaration)
    (task / SOURCE).rename(task / "implementation.py")
    (task / SOURCE).symlink_to("../implementation.py")
    workspace = tmp_path / "submitted"
    shutil.copytree(task, workspace, symlinks=True)
    (workspace / "implementation.py").write_text(CANDIDATE)
    seen = []

    def correctness(copied_workspace, *args):
        seen.append(runpy.run_path(str(copied_workspace / SOURCE))["kernel"]())
        return True, None

    with (
        patch("src.held_out.run_heldout_eval.apply_all_injections", return_value=True),
        patch("src.held_out.run_heldout_eval.evaluate_compilation", return_value=(True, None)),
        patch("src.held_out.run_heldout_eval.evaluate_correctness", side_effect=correctness),
        patch("src.held_out.run_heldout_eval.measure_baseline", return_value=[]),
        patch("src.held_out.run_heldout_eval.measure_performance", return_value=[]),
    ):
        evaluate_single_task(workspace, tmp_path / "heldout", {}, task, LOG)
    assert seen == [1, 2]
    assert (workspace / "implementation.py").read_text() == CANDIDATE
    for root in (workspace, tmp_path / "heldout/orig", tmp_path / "heldout/opt"):
        assert (root / SOURCE).is_symlink()
        assert str((root / SOURCE).readlink()) == "../implementation.py"


@pytest.mark.parametrize("change", ["add", "modify", "delete", "replace_with_file"])
def test_quality_loop_applies_alias_node_changes_without_touching_referents(tmp_path, change):
    source, destination = tmp_path / "source", tmp_path / "destination"
    for root in (source, destination):
        (root / "ut").mkdir(parents=True)
        (root / "first.py").write_text("FIRST = 1\n")
        (root / "second.py").write_text("SECOND = 2\n")
        if change != "add":
            (root / "ut/alias.py").symlink_to("../first.py")
    before = snapshot_tree(source)
    alias = source / "ut/alias.py"
    if change != "add":
        alias.unlink()
    if change in {"add", "modify"}:
        alias.symlink_to("../second.py")
    elif change == "replace_with_file":
        alias.write_text("REPLACEMENT = 3\n")

    changes = diff_trees(before, snapshot_tree(source))
    assert changes.paths == ("ut/alias.py",)
    apply_changes(source, destination, changes)

    copied = destination / "ut/alias.py"
    if change == "delete":
        assert not copied.exists() and not copied.is_symlink()
    elif change == "replace_with_file":
        assert not copied.is_symlink()
        assert copied.read_text() == "REPLACEMENT = 3\n"
    else:
        assert copied.is_symlink()
        assert str(copied.readlink()) == "../second.py"
        assert copied.resolve() == destination / "second.py"
    for root in (source, destination):
        assert (root / "first.py").read_text() == "FIRST = 1\n"
        assert (root / "second.py").read_text() == "SECOND = 2\n"


@pytest.mark.parametrize("side,delete", [("source", False), ("destination", False), ("destination", True)])
def test_quality_loop_rejects_escaping_parents(tmp_path, side, delete):
    source, destination, outside = [tmp_path / name for name in ("source", "destination", "outside")]
    for root in (source, destination, outside):
        (root / "nested").mkdir(parents=True)
        (root / "nested/kernel.py").write_text(root.name)
    redirected = source if side == "source" else destination
    shutil.rmtree(redirected / "nested")
    (redirected / "nested").symlink_to(outside / "nested", target_is_directory=True)
    changes = TreeChanges(
        added=(), modified=() if delete else ("nested/kernel.py",),
        deleted=("nested/kernel.py",) if delete else (),
    )
    with pytest.raises(ValueError, match="unsafe quality_loop path"):
        apply_changes(source, destination, changes)
    assert (outside / "nested/kernel.py").read_text() == "outside"


def test_quality_loop_can_delete_an_escaping_leaf_link_without_touching_target(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("UNCHANGED = True\n")
    (destination / "alias.py").symlink_to(outside)
    apply_changes(source, destination, TreeChanges((), (), ("alias.py",)))
    assert not (destination / "alias.py").is_symlink()
    assert outside.read_text() == "UNCHANGED = True\n"


@pytest.mark.parametrize("absolute", [False, True])
def test_quality_loop_rejects_nonrelocatable_added_aliases(tmp_path, absolute):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    target = source / "kernel.py" if absolute else tmp_path / "outside.py"
    target.write_text("UNCHANGED = True\n")
    (source / "alias.py").symlink_to(target if absolute else "../outside.py")
    with pytest.raises(ValueError, match="unsafe quality_loop symlink"):
        apply_changes(source, destination, TreeChanges(("alias.py",), (), ()))
    assert not (destination / "alias.py").exists()
    assert target.read_text() == "UNCHANGED = True\n"
