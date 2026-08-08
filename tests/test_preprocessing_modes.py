import logging
import stat
from pathlib import Path

from src.perf_helper_materialization import MARK_END, MARK_START
from src.preprocessing import _make_workspace_tree_owner_mutable, setup_workspace


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def test_setup_workspace_makes_sealed_task_copy_mutable_without_touching_source(
    tmp_path: Path,
) -> None:
    task = tmp_path / "sealed_task"
    scripts = task / "scripts"
    scripts.mkdir(parents=True)
    config = task / "config.yaml"
    runner = scripts / "task_runner.py"
    executable = task / "run.sh"
    config.write_text("task_type: triton2triton\n", encoding="utf-8")
    runner.write_text(
        f"{MARK_START}\nold generated block\n{MARK_END}\n",
        encoding="utf-8",
    )
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    for path in (config, runner):
        path.chmod(0o444)
    executable.chmod(0o555)
    scripts.chmod(0o555)
    task.chmod(0o555)

    run_directory = tmp_path / "run"
    workspace = setup_workspace(
        str(config),
        run_directory,
        "20260808_000000",
        logging.getLogger(__name__),
        task_name="triton2triton/example",
    )

    assert _mode(task) == 0o555
    assert _mode(scripts) == 0o555
    assert _mode(config) == 0o444
    assert _mode(runner) == 0o444
    assert _mode(executable) == 0o555

    assert _mode(workspace) & stat.S_IWUSR
    assert _mode(workspace / "scripts") & stat.S_IWUSR
    assert _mode(workspace / "config.yaml") & stat.S_IWUSR
    assert _mode(workspace / "scripts/task_runner.py") & stat.S_IWUSR
    assert _mode(workspace / "run.sh") == 0o755
    materialized = (workspace / "scripts/task_runner.py").read_text(encoding="utf-8")
    assert "old generated block" not in materialized
    assert "def _measure_cuda_event_fallback" in materialized


def test_workspace_mode_normalization_never_follows_symlinks(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    target = external / "target.py"
    target.write_text("protected = True\n", encoding="utf-8")
    target.chmod(0o444)
    external.chmod(0o555)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "external-link").symlink_to(external, target_is_directory=True)
    (workspace / "target-link.py").symlink_to(target)

    _make_workspace_tree_owner_mutable(workspace)

    assert (workspace / "external-link").is_symlink()
    assert (workspace / "target-link.py").is_symlink()
    assert _mode(external) == 0o555
    assert _mode(target) == 0o444
