# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Evaluate a given implementation: install solution files, then let Arena score them.

No model, network access or credentials are involved. The run config names a
directory of solution files::

    agent:
      template: eval_only
      source_dir: solutions/my_kernels   # relative to the repository root, or absolute

It holds one directory per task, ``<source_dir>/<task_id>/``, laid out like the
task workspace: every file sits at a path the task declares editable (for a
schema-v3 kernel task, under ``solution/``). The launcher reads and validates
them, then installs them through a short stdlib-only process started under the
agent sandbox, so the only process that writes the workspace is confined to it.
A solution file replaces the file at its path; files given under a
directory (``tree``) declaration become that directory's complete contents.
Editable paths without solution files keep their initial contents.

A task whose solution is missing or invalid fails here with the reason. The
framework then evaluates the unchanged initial candidate, as after any agent
failure, and reports ``agent_execution.status: FAILED`` with
``candidate_changed: false``; such a result is not the solution's.

The launcher records what it installed (paths, sizes, SHA-256 digests) in
``.eval-only-*/manifest.json`` in the workspace and in the run log. The final
result's ``evaluated_candidate_sources`` uses the same digests.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tempfile
from typing import Any

import yaml

from agents import register_agent
from src.agent_sandbox import POLICY_ENV, repository_root, wrap_command
from src.process_control import stop_process_group
from src.task_spec import TaskConfigError, TaskSpec, load_task_spec, relative_path, resolve_task_path

AGENT_NAME = "eval_only"
MANIFEST_VERSION = 1
# Directories the evaluator ignores inside an editable tree (see TaskSession);
# a file there would be installed but never evaluated, so it is refused.
RUNTIME_DIRECTORIES = frozenset({"__pycache__", ".pytest_cache", ".git", "build"})

# Runs as ``python -I -c`` under the agent sandbox, which denies it the
# repository checkout, so it must not import anything from this repository.
# stdin: {"replace": [tree paths], "files": {path: base64}}; argv[1]: workspace.
# It refuses symlinks on the way to a destination and prints the digests it wrote.
INSTALLER = r'''
import base64, hashlib, json, os, shutil, stat, sys

root = os.path.realpath(sys.argv[1])
plan = json.load(sys.stdin)


def target(relative):
    path = root
    for part in relative.split("/"):
        path = os.path.join(path, part)
        if os.path.islink(path):
            raise SystemExit("refusing a symlink in the workspace at " + os.path.relpath(path, root))
    return path


for tree in plan["replace"]:
    path = target(tree)
    if os.path.lexists(path):
        if not os.path.isdir(path):
            raise SystemExit("editable directory is not a directory: " + tree)
        for name in os.listdir(path):
            child = os.path.join(path, name)
            if os.path.isdir(child) and not os.path.islink(child):
                shutil.rmtree(child)
            else:
                os.unlink(child)
written = {}
for relative, encoded in sorted(plan["files"].items()):
    data = base64.b64decode(encoded)
    path = target(relative)
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    if os.path.lexists(path):
        if os.path.isdir(path) and not os.path.islink(path):
            raise SystemExit("a directory occupies the solution file path " + relative)
        os.unlink(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
    written[relative] = hashlib.sha256(data).hexdigest()
print(json.dumps(written, sort_keys=True))
'''


class SolutionError(RuntimeError):
    """The solution files for a task are missing or not acceptable."""


def _load_agent_config(eval_config: dict[str, Any]) -> dict[str, Any]:
    with Path(__file__).with_name("agent_config.yaml").open() as f:
        config = yaml.safe_load(f) or {}
    overrides = eval_config.get("agent") or {}
    if not isinstance(overrides, dict):
        raise ValueError("agent must be a mapping")
    for key in ("source_dir", "timeout_seconds"):
        if key in overrides:
            config[key] = overrides[key]
    source = config.get("source_dir")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("agent.source_dir must name the directory of solution files")
    timeout = config.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("agent.timeout_seconds must be a positive integer")
    return config


def source_root(config: dict[str, Any], repo_root: Path | None = None) -> Path:
    """The solution directory; a relative ``source_dir`` resolves against the repository root."""
    path = Path(config["source_dir"]).expanduser()
    if not path.is_absolute():
        path = (repository_root() if repo_root is None else repo_root) / path
    if not path.is_dir():
        raise SolutionError(f"agent.source_dir is not a directory: {config['source_dir']}")
    return path.resolve()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect(task_directory: Path, spec: TaskSpec) -> dict[str, bytes]:
    """Read the solution files of one task, refusing anything outside its editable paths."""
    if task_directory.is_symlink():
        raise SolutionError(f"the task's solution directory is a symlink: {task_directory}")
    if not task_directory.is_dir():
        raise SolutionError(f"no solution files for this task: {task_directory} does not exist")
    files: dict[str, bytes] = {}
    errors: list[str] = []
    editable = spec.candidate.editable
    for directory, names, file_names in os.walk(task_directory, followlinks=False):
        for name in sorted(names + file_names):
            path = Path(directory, name)
            relative = path.relative_to(task_directory).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                errors.append(f"{relative}: symlinks are not accepted")
                continue
            if stat.S_ISDIR(mode):
                if name in RUNTIME_DIRECTORIES:
                    errors.append(f"{relative}/: {name} is not evaluated; remove it")
                elif not any(edit.contains(relative) or PurePosixPath(edit.path).is_relative_to(relative)
                             for edit in editable):
                    errors.append(f"{relative}/: not an editable path of this task")
                continue
            if not stat.S_ISREG(mode):
                errors.append(f"{relative}: not a regular file")
                continue
            try:
                relative_path(relative, "solution file")
            except TaskConfigError as exc:
                errors.append(str(exc))
                continue
            if not any(edit.contains(relative) for edit in editable):
                errors.append(f"{relative}: not an editable path of this task")
                continue
            files[relative] = path.read_bytes()
        names[:] = [name for name in names if not Path(directory, name).is_symlink()
                    and name not in RUNTIME_DIRECTORIES]
    if errors:
        allowed = ", ".join(f"{edit.path}{'/' if edit.scope == 'tree' else ''}" for edit in editable)
        raise SolutionError(f"rejected solution ({'; '.join(errors)}); editable paths: {allowed}")
    if not files:
        raise SolutionError(f"no solution files for this task in {task_directory}")
    return files


def plan(files: dict[str, bytes], spec: TaskSpec, workspace: Path) -> dict[str, Any]:
    """The installation: trees to empty and files to write; checks the entry points stay present."""
    replace = sorted(edit.path for edit in spec.candidate.editable
                     if edit.scope == "tree" and any(edit.contains(path) for path in files))
    missing = []
    for entry in spec.candidate.entrypoints:
        if entry.file in files:
            continue
        if any(PurePosixPath(tree) in PurePosixPath(entry.file).parents for tree in replace):
            missing.append(entry.file)
        elif not resolve_task_path(workspace, entry.file).is_file():
            missing.append(entry.file)
    if missing:
        raise SolutionError("the solution lacks the task's entry point file(s): " + ", ".join(sorted(missing)))
    return {"replace": replace, "files": {path: base64.b64encode(data).decode("ascii")
                                          for path, data in sorted(files.items())}}


def install(installation: dict[str, Any], workspace: Path, timeout_seconds: int,
            logger: logging.Logger) -> dict[str, str]:
    """Run the installer under the agent sandbox, when one is active; return the digests it wrote."""
    env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", POLICY_ENV) if key in os.environ}
    command = [sys.executable, "-I", "-c", INSTALLER, str(workspace)]
    process = subprocess.Popen(
        wrap_command(command, env),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=workspace, env=env, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(json.dumps(installation), timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        stop_process_group(process)
        raise TimeoutError(f"installing the solution took longer than {timeout_seconds}s")
    except BaseException:
        stop_process_group(process)
        raise
    if process.returncode != 0:
        detail = (stderr or stdout or "").strip().splitlines()
        raise SolutionError(f"installing the solution failed (exit {process.returncode}): "
                              + (detail[-1] if detail else "no output"))
    if stderr.strip():
        logger.warning("Eval-only installer stderr: %s", stderr.strip())
    return json.loads(stdout.strip().splitlines()[-1])


def verify(files: dict[str, bytes], installation: dict[str, Any], workspace: Path) -> None:
    """Check the workspace holds exactly the given bytes at the given paths."""
    for path, data in files.items():
        installed = resolve_task_path(workspace, path, must_exist=True)
        if installed.is_symlink() or not installed.is_file() or _sha256(installed.read_bytes()) != _sha256(data):
            raise SolutionError(f"the installed file differs from the solution: {path}")
    for tree in installation["replace"]:
        root = resolve_task_path(workspace, tree, must_exist=True)
        present = {p.relative_to(workspace).as_posix() for p in root.rglob("*") if not p.is_dir()}
        extra = sorted(present - set(files))
        if extra:
            raise SolutionError(f"unexpected files in {tree}/ after installation: {', '.join(extra)}")


def write_manifest(workspace: Path, record: dict[str, Any]) -> Path:
    directory = Path(tempfile.mkdtemp(prefix=".eval-only-", dir=workspace))
    path = directory / "manifest.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


@register_agent(AGENT_NAME)
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str) -> str:
    """Install the solution files of this task into its workspace.

    Args:
        eval_config: Evaluator settings passed from main (agent settings, ``_task_id``, ``_target_language``)
        task_config_dir: The task's task.yaml (schema v3) or config.yaml (e2e, schema v2)
        workspace: Workspace directory whose candidate the framework evaluates afterwards

    Returns:
        str: A summary of the installed files

    Raises:
        SolutionError: The solution for this task is missing or not acceptable.
    """
    logger = logging.getLogger(__name__)
    config = _load_agent_config(eval_config)
    task_id = eval_config.get("_task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("the framework did not provide _task_id")
    relative_path(task_id, "task id")
    root = Path(workspace).resolve(strict=True)
    spec = load_task_spec(Path(task_config_dir), task_id=task_id,
                          target_language=eval_config.get("_target_language"))
    source = source_root(config)
    task_directory = source / task_id
    logger.info("Eval-only: task %s from %s", task_id, task_directory)
    files = collect(task_directory, spec)
    installation = plan(files, spec, root)
    written = install(installation, root, config["timeout_seconds"], logger)
    expected = {path: _sha256(data) for path, data in files.items()}
    if written != expected:
        raise SolutionError("the installer reported digests that differ from the solution")
    verify(files, installation, root)
    record = {
        "version": MANIFEST_VERSION,
        "task_id": task_id,
        "target_language": spec.target_language,
        "source_dir": config["source_dir"],
        "replaced_trees": installation["replace"],
        "files": [{"path": path, "bytes": len(files[path]), "sha256": expected[path]} for path in sorted(files)],
    }
    manifest = write_manifest(root, record)
    for row in record["files"]:
        logger.info("Eval-only installed %s (%d bytes, sha256 %s)", row["path"], row["bytes"], row["sha256"])
    logger.info("Eval-only manifest: %s", manifest)
    return f"Installed {len(files)} solution file(s) for {task_id}: " + ", ".join(sorted(files))
