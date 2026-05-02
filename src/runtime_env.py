# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Helpers for keeping subprocesses on the same Python environment."""
import os
import sys
from pathlib import Path


PYTHON_ENV_VAR = "AGENT_KERNEL_ARENA_PYTHON"


def _absolute_path(path: Path) -> Path:
    """Return an absolute path without resolving symlinks."""
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return Path(os.path.abspath(path))


def _path_without(path_value: str, remove_dir: Path) -> list[str]:
    """Return PATH entries without duplicates of remove_dir."""
    entries: list[str] = []
    for raw_entry in path_value.split(os.pathsep):
        if not raw_entry:
            continue
        if _absolute_path(Path(raw_entry)) == remove_dir:
            continue
        entries.append(raw_entry)
    return entries


def build_subprocess_env(python_path: str | None = None) -> dict[str, str]:
    """Build an environment where bare python/pytest use this run's interpreter."""
    env = os.environ.copy()
    selected_python = _absolute_path(Path(python_path or sys.executable))
    python_bin = selected_python.parent

    if python_bin.exists():
        path_entries = _path_without(env.get("PATH", ""), python_bin)
        env["PATH"] = os.pathsep.join([str(python_bin), *path_entries])
        env[PYTHON_ENV_VAR] = str(selected_python)

    return env


def apply_subprocess_python_path(python_path: str | None = None) -> str:
    """Apply build_subprocess_env() to the current process and return the Python path."""
    env = build_subprocess_env(python_path)
    os.environ.update(env)
    fallback = str(_absolute_path(Path(python_path or sys.executable)))
    return env.get(PYTHON_ENV_VAR, fallback)
