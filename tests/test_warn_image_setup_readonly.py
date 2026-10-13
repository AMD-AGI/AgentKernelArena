"""The image tasks' setup must leave the materialized baseline untouched."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.task_spec import load_task_spec


ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "image_kernel/mi300x_sglang_hip_pa_decode",
    "image_kernel/mi300x_sglang_hip_pa_ragged",
)


def copied_setup(source: Path, destination: Path) -> None:
    for name in ("config.yaml", "scripts/setup_task.py", "scripts/task_adapter.py"):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / name).read_bytes())
    for name in (
        "aiter/csrc/cpp_itfs/pa/pa_kernels.cuh",
        "aiter/csrc/cpp_itfs/pa/pa_ragged.cuh",
        "aiter/csrc/include/aiter_enum.h",
    ):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"// synthetic declared image source\n")


def tree_bytes(root: Path) -> dict[str, bytes | None]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def run_setup(workspace: Path, command: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env.pop("PYTHONPYCACHEPREFIX", None)
    return subprocess.run(
        [sys.executable, *command[1:]], cwd=workspace, env=env,
        capture_output=True, text=True, timeout=30, check=False,
    )


@pytest.mark.parametrize("task_id", TASKS)
def test_declared_image_setup_is_read_only_without_bytecode_environment(tmp_path, task_id):
    source = ROOT / "tasks" / task_id
    setup = load_task_spec(source / "config.yaml", task_id=task_id).to_mapping()["workspace"]["setup"]
    assert setup == [["python3", "-B", "scripts/setup_task.py"]]

    workspace = tmp_path / "workspace"
    copied_setup(source, workspace)
    before = tree_bytes(workspace)
    completed = run_setup(workspace, setup[0])
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "setup: PASS" in completed.stdout
    assert tree_bytes(workspace) == before
    assert not list(workspace.rglob("__pycache__"))


@pytest.mark.parametrize("task_id", TASKS)
def test_plain_python_setup_would_write_import_bytecode(tmp_path, task_id):
    source = ROOT / "tasks" / task_id
    workspace = tmp_path / "workspace"
    copied_setup(source, workspace)
    before = tree_bytes(workspace)
    completed = run_setup(workspace, ["python3", "scripts/setup_task.py"])
    assert completed.returncode == 0, completed.stdout + completed.stderr
    after = tree_bytes(workspace)
    assert all(after[name] == contents for name, contents in before.items())
    assert any(name.startswith("scripts/__pycache__/task_adapter.") and name.endswith(".pyc")
               for name in after.keys() - before.keys())
