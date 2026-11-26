"""eval_only agent: installs given solution files, then the framework evaluates them.

CPU fixtures only; timings are synthetic, not GPU evidence. Landlock enforcement
runs only where the kernel offers Landlock ABI 3 or later.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import yaml

from agents.eval_only import launch_agent as eval_only
from src import agent_sandbox as sandbox
from src.task_run import run_task_v2
from tests.test_task_session_v2 import RUNNER

ROOT = Path(__file__).resolve().parents[1]
V3_TASK = "triton/activation/silu_and_mul/fp16_d3200"
FIXTURE_TASK = "suite/protocol_fixture"
GOOD = "def compute():\n    return 1 + 2\n"
WRONG = "def compute():\n    return 4\n"


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def cpu_runtime(monkeypatch):
    monkeypatch.setattr("src.task_runtime._runtime_identity", lambda: {
        "gpu_arch": "gfx950", "gpu_name": "CPU protocol fixture, no GPU", "torch": "fixture"})
    monkeypatch.setenv("AGENT_KERNEL_ARENA_PYTHON", sys.executable)


def tree_package(tmp_path):
    """A schema-v2 fixture whose candidate is the solution/ directory, like a v3 task."""
    root = tmp_path / "task-package"
    (root / "solution").mkdir(parents=True)
    (root / "evaluate.py").write_text(RUNNER.replace('pathlib.Path("kernel.py")', 'pathlib.Path("solution/kernel.py")'))
    (root / "solution" / "kernel.py").write_text("def compute():\n    return 3\n")
    (root / "solution" / "baseline_helper.py").write_text("UNUSED = True\n")
    config = {"schema_version": 2,
              "candidate": {"language": "hip", "editable": [{"path": "solution", "scope": "tree"}],
                            "entrypoints": [{"file": "solution/kernel.py", "kind": "function", "symbol": "compute"}]},
              "evaluation": {"runner": [sys.executable, "evaluate.py"]}}
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def provide(source, task_id, files):
    for relative, text in files.items():
        path = source / task_id / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return source


def run(tmp_path, path, source, mode="none"):
    config = {"agent": {"template": "eval_only", "source_dir": str(source)}, "agent_sandbox": mode}
    return run_task_v2(eval_config=config, agent=SimpleNamespace(value="eval_only"),
                       agent_launcher=eval_only.launch_agent, task_name=FIXTURE_TASK,
                       task_config_dir=str(path), run_directory=tmp_path / "run", timestamp="fixture",
                       logger=logging.getLogger("test_eval_only_agent"))


def report(workspace):
    return yaml.safe_load((workspace / "task_result.yaml").read_text())


def manifest(workspace):
    (path,) = workspace.glob(".eval-only-*/manifest.json")
    return json.loads(path.read_text())


def test_a_correct_solution_is_installed_and_accepted(tmp_path, cpu_runtime):
    source = provide(tmp_path / "solutions", FIXTURE_TASK, {"solution/kernel.py": GOOD,
                                                           "solution/nested/helper.py": "X = 1\n"})
    complete, workspace = run(tmp_path, tree_package(tmp_path), source)
    result = report(workspace)
    assert complete and result["candidate_accepted"] is True
    assert result["agent_execution"]["status"] == "COMPLETED"
    assert result["agent_execution"]["candidate_changed"] is True
    # The solution files are the complete solution/ tree: the initial helper is gone.
    assert result["evaluated_candidate_sources"] == {"solution/kernel.py": sha256(GOOD),
                                                     "solution/nested/helper.py": sha256("X = 1\n")}
    record = manifest(workspace)
    assert record["task_id"] == FIXTURE_TASK and record["replaced_trees"] == ["solution"]
    assert {row["path"]: row["sha256"] for row in record["files"]} == result["evaluated_candidate_sources"]


def test_a_wrong_solution_is_evaluated_and_rejected(tmp_path, cpu_runtime):
    source = provide(tmp_path / "solutions", FIXTURE_TASK, {"solution/kernel.py": WRONG})
    _, workspace = run(tmp_path, tree_package(tmp_path), source)
    result = report(workspace)
    assert result["agent_execution"]["status"] == "COMPLETED"
    assert result["pass_correctness"] is False
    assert result["candidate_accepted"] is False


def test_a_missing_solution_fails_the_agent_and_leaves_the_initial_candidate(tmp_path, cpu_runtime):
    source = tmp_path / "solutions"
    source.mkdir()
    _, workspace = run(tmp_path, tree_package(tmp_path), source)
    result = report(workspace)
    assert result["agent_execution"]["status"] == "FAILED"
    assert "no solution files for this task" in result["agent_execution"]["error"]
    assert result["agent_execution"]["candidate_changed"] is False
    assert result["candidate_changed_from_initial"] is False
    assert not list(workspace.glob(".eval-only-*"))


def test_a_file_scope_solution_replaces_only_that_file(tmp_path, cpu_runtime):
    from tests.test_task_run_v2 import package
    source = provide(tmp_path / "solutions", FIXTURE_TASK, {"kernel.py": GOOD})
    _, workspace = run(tmp_path, package(tmp_path), source)
    result = report(workspace)
    assert result["candidate_accepted"] is True
    assert result["evaluated_candidate_sources"] == {"kernel.py": sha256(GOOD)}
    assert manifest(workspace)["replaced_trees"] == []


@pytest.mark.parametrize("case,expected", [
    ("outside", "evaluate.py: not an editable path"),
    ("directory_outside", "notes/: not an editable path"),
    ("symlink_file", "solution/kernel.py: symlinks are not accepted"),
    ("symlink_directory", "solution/linked: symlinks are not accepted"),
    ("runtime_directory", "solution/__pycache__/: __pycache__ is not evaluated"),
    ("no_entry_point", "lacks the task's entry point file(s): solution/kernel.py"),
    ("empty", "no solution files for this task in"),
    ("task_symlink", "the task's solution directory is a symlink"),
])
def test_unacceptable_solutions_are_refused_before_the_workspace_changes(tmp_path, case, expected):
    workspace = tmp_path / "workspace"
    shutil.copytree(tree_package(tmp_path).parent, workspace)
    before = {p.relative_to(workspace).as_posix(): p.read_bytes() for p in workspace.rglob("*") if p.is_file()}
    source = tmp_path / "solutions"
    task = source / FIXTURE_TASK
    outside = tmp_path / "elsewhere.py"
    outside.write_text(GOOD)
    if case == "task_symlink":
        (tmp_path / "real").mkdir()
        task.parent.mkdir(parents=True)
        task.symlink_to(tmp_path / "real")
    else:
        (task / "solution").mkdir(parents=True)
    if case != "symlink_file" and case not in ("empty", "task_symlink", "no_entry_point"):
        (task / "solution" / "kernel.py").write_text(GOOD)
    if case == "outside":
        (task / "evaluate.py").write_text("print('replaced harness')\n")
    elif case == "directory_outside":
        (task / "notes").mkdir()
    elif case == "symlink_file":
        (task / "solution" / "kernel.py").symlink_to(outside)
    elif case == "symlink_directory":
        (task / "solution" / "linked").symlink_to(tmp_path)
    elif case == "runtime_directory":
        (task / "solution" / "__pycache__").mkdir()
        (task / "solution" / "__pycache__" / "kernel.cpython-312.pyc").write_bytes(b"\0")
    elif case == "no_entry_point":
        (task / "solution" / "other.py").write_text(GOOD)
    eval_config = {"agent": {"template": "eval_only", "source_dir": str(source)}, "_task_id": FIXTURE_TASK}
    with pytest.raises(eval_only.SolutionError) as error:
        eval_only.launch_agent(eval_config=eval_config, task_config_dir=str(workspace / "config.yaml"),
                                workspace=str(workspace))
    assert expected in str(error.value)
    after = {p.relative_to(workspace).as_posix(): p.read_bytes() for p in workspace.rglob("*") if p.is_file()}
    assert after == before


def test_the_installer_never_follows_a_symlink_out_of_the_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    shutil.copytree(tree_package(tmp_path).parent, workspace)
    outside = tmp_path / "outside"
    shutil.move(str(workspace / "solution"), outside)
    (workspace / "solution").symlink_to(outside)
    source = provide(tmp_path / "solutions", FIXTURE_TASK, {"solution/kernel.py": GOOD})
    eval_config = {"agent": {"template": "eval_only", "source_dir": str(source)}, "_task_id": FIXTURE_TASK}
    with pytest.raises(eval_only.SolutionError, match="refusing a symlink in the workspace at solution"):
        eval_only.launch_agent(eval_config=eval_config, task_config_dir=str(workspace / "config.yaml"),
                                workspace=str(workspace))
    assert sorted(p.name for p in outside.iterdir()) == ["baseline_helper.py", "kernel.py"]
    assert "return 3" in (outside / "kernel.py").read_text()


def test_a_task_id_cannot_leave_the_source_directory(tmp_path):
    from src.task_spec import TaskConfigError
    eval_config = {"agent": {"template": "eval_only", "source_dir": str(tmp_path)}, "_task_id": "../escape"}
    with pytest.raises(TaskConfigError, match="normalized task-relative path"):
        eval_only.launch_agent(eval_config=eval_config, task_config_dir=str(tree_package(tmp_path)),
                                workspace=str(tmp_path))


def test_a_v3_task_accepts_files_under_solution_only(tmp_path):
    task_file = ROOT / "tasks" / V3_TASK / "task.yaml"
    workspace = tmp_path / "workspace"
    (workspace / "solution").mkdir(parents=True)
    (workspace / "solution" / "kernel.py").write_text("# initial\n")
    source = provide(tmp_path / "solutions", V3_TASK, {"solution/kernel.py": GOOD, "solution/helpers.py": "Y = 2\n"})
    eval_config = {"agent": {"template": "eval_only", "source_dir": str(source)},
                   "_task_id": V3_TASK, "_target_language": "triton"}
    summary = eval_only.launch_agent(eval_config=eval_config, task_config_dir=str(task_file),
                                      workspace=str(workspace))
    assert "solution/helpers.py, solution/kernel.py" in summary
    assert sorted(p.name for p in (workspace / "solution").iterdir()) == ["helpers.py", "kernel.py"]
    record = manifest(workspace)
    assert record["target_language"] == "triton" and record["replaced_trees"] == ["solution"]
    provide(source, V3_TASK, {"reference.py": "def reference(x): return x\n"})
    with pytest.raises(eval_only.SolutionError, match="reference.py: not an editable path"):
        eval_only.launch_agent(eval_config=eval_config, task_config_dir=str(task_file), workspace=str(workspace))


def test_configuration_requires_a_source_directory_and_resolves_it_in_the_checkout(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="agent.source_dir"):
        eval_only._load_agent_config({"agent": {"template": "eval_only"}})
    with pytest.raises(ValueError, match="timeout_seconds"):
        eval_only._load_agent_config({"agent": {"source_dir": "x", "timeout_seconds": 0}})
    config = eval_only._load_agent_config({"agent": {"source_dir": "solutions/one"}})
    (tmp_path / "solutions" / "one").mkdir(parents=True)
    assert eval_only.source_root(config, repo_root=tmp_path) == (tmp_path / "solutions" / "one").resolve()
    assert eval_only.source_root({"source_dir": str(tmp_path / "solutions")}) == (tmp_path / "solutions").resolve()
    with pytest.raises(eval_only.SolutionError, match="not a directory"):
        eval_only.source_root({"source_dir": "missing"}, repo_root=tmp_path)


landlock = pytest.mark.skipif(sandbox.landlock_abi() < sandbox.MIN_ABI,
                              reason="needs Linux with Landlock ABI 3 or later")


@landlock
def test_the_installer_runs_confined_and_attests_the_policy(tmp_path, cpu_runtime, monkeypatch):
    # tmp_path stands in for the checkout: the run, its state and the solution are inside it.
    monkeypatch.setattr(sandbox, "repository_root", lambda: tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "agent_sandbox.py").write_text("protected\n")  # the wrapper's self-test file
    home = tmp_path.parent / f"{tmp_path.name}-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    source = provide(tmp_path / "solutions", FIXTURE_TASK, {"solution/kernel.py": GOOD})
    _, workspace = run(tmp_path, tree_package(tmp_path), source, mode="landlock")
    result = report(workspace)
    assert result["agent_execution"]["status"] == "COMPLETED", result["agent_execution"]
    assert result["agent_sandbox"]["status"] == "enforced" and result["agent_sandbox"]["launches"] == 1
    assert result["candidate_accepted"] is True
