# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Utilities for evaluator: command execution and file I/O.
"""
import ast
import os
import shutil
import subprocess
import logging
import yaml
import shlex
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List
from .testcases import TestCaseResult
from .runtime_env import PYTHON_ENV_VAR, build_subprocess_env


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str)]
    return []


def _is_docstring_statement(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _is_not_implemented_exception(expression: Optional[ast.expr]) -> bool:
    if isinstance(expression, ast.Call):
        expression = expression.func
    if isinstance(expression, ast.Name):
        return expression.id == "NotImplementedError"
    return (
        isinstance(expression, ast.Attribute)
        and expression.attr == "NotImplementedError"
        and isinstance(expression.value, ast.Name)
        and expression.value.id == "builtins"
    )


def _is_unimplemented_target_stub(function: ast.AST) -> bool:
    """Match only a no-op body followed by a direct NotImplementedError raise."""
    body = list(getattr(function, "body", []))
    meaningful_statements: List[ast.stmt] = []
    for index, statement in enumerate(body):
        if index == 0 and _is_docstring_statement(statement):
            continue
        if isinstance(statement, ast.Pass):
            continue
        meaningful_statements.append(statement)

    return (
        len(meaningful_statements) == 1
        and isinstance(meaningful_statements[0], ast.Raise)
        and _is_not_implemented_exception(meaningful_statements[0].exc)
    )


def inspect_target_definitions(
    workspace: Path,
    task_config: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """Return missing and unimplemented declared top-level Python targets.

    This intentionally does not walk into function bodies.  An implemented
    target may use a conditional ``NotImplementedError`` for an unsupported
    shape without being classified as an unimplemented submission.  The task
    contract requires a Python ``def`` for each target; assignment aliases are
    not treated as target definitions.
    """
    target_names = set(_string_list(task_config.get("target_kernel_functions")))
    if not target_names:
        return [], []

    found_names = set()
    stub_names = set()
    for configured_path in _string_list(task_config.get("source_file_path")):
        source_path = Path(configured_path)
        if not source_path.is_absolute():
            source_path = Path(workspace) / source_path
        if not source_path.is_file() or source_path.suffix != ".py":
            continue
        try:
            module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        except (OSError, UnicodeError, SyntaxError):
            # Compilation reports missing, unreadable, and invalid source files;
            # this guard is deliberately limited to recognized starter stubs.
            continue

        # The last top-level definition is the one bound by the module at run
        # time.  Nested methods/functions are intentionally excluded.
        definitions = {}
        for statement in module.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions[statement.name] = statement
        for target_name in target_names:
            function = definitions.get(target_name)
            if function is None:
                continue
            found_names.add(target_name)
            if _is_unimplemented_target_stub(function):
                stub_names.add(target_name)

    return sorted(target_names - found_names), sorted(stub_names)


def find_unimplemented_target_stubs(
    workspace: Path,
    task_config: Dict[str, Any],
) -> List[str]:
    """Compatibility helper returning only declared starter stubs."""
    _, stub_names = inspect_target_definitions(workspace, task_config)
    return stub_names


def _replace_leading_token(command: str, token: str, replacement: str) -> str:
    leading_len = len(command) - len(command.lstrip())
    leading = command[:leading_len]
    stripped = command[leading_len:]
    if stripped == token or stripped.startswith(f"{token} "):
        return f"{leading}{replacement}{stripped[len(token):]}"
    return command


def normalize_python_command(command: str, python_path: str) -> str:
    """Route bare Python tooling commands through the selected interpreter."""
    normalized = command
    normalized = _replace_leading_token(normalized, "python3", python_path)
    normalized = _replace_leading_token(normalized, "python", python_path)
    normalized = _replace_leading_token(normalized, "pytest", f"{python_path} -m pytest")
    return normalized


def run_command(
    command: str,
    workspace: Path,
    timeout: int = 300,
    logger: Optional[logging.Logger] = None,
    docker_container: Optional[str] = None,
    extra_env: Optional[dict] = None,
) -> Tuple[bool, str, str]:
    """
    Run a shell command in the workspace directory.

    When ``docker_container`` is provided the command is executed inside the
    named Docker container via ``docker exec``.  The workspace path is
    assumed to be identical on host and inside the container (bind-mounted).

    Args:
        command: Shell command to execute
        workspace: Working directory
        timeout: Command timeout in seconds
        logger: Optional logger for output
        docker_container: If set, run the command inside this Docker container
        extra_env: Optional env vars applied to THIS subprocess only (merged over
            the inherited env). Used e.g. to scope AITER_REBUILD=1 to a single
            build step without leaking it into the parent process / later tasks.

    Returns:
        Tuple of (success: bool, stdout: str, stderr: str)
    """
    log = logger or logging.getLogger(__name__)

    try:
        env = build_subprocess_env()
        if extra_env:
            env.update({str(k): str(v) for k, v in extra_env.items()})
        if docker_container:
            # When running inside a Docker container we can't rewrite "python3" to
            # the host interpreter path — skip normalize_python_command and wrap
            # the original command in `docker exec` instead.
            escaped = command.replace("'", "'\\''")
            abs_workspace = Path(workspace).resolve()
            command_to_run = (
                f"docker exec -w {abs_workspace} {docker_container} "
                f"bash -c '{escaped}'"
            )
            log.info(f"Running in Docker [{docker_container}]: {command_to_run[:200]}")
        else:
            python_path = env.get(PYTHON_ENV_VAR)
            quoted_python = shlex.quote(python_path) if python_path else None
            command_to_run = normalize_python_command(command, quoted_python) if quoted_python else command
            log.info(f"Running command: {command_to_run}")
            if command_to_run != command:
                log.info(f"Original command: {command}")

        log.info(f"Working directory: {workspace}")

        result = subprocess.run(
            command_to_run,
            shell=True,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        if result.returncode == 0:
            log.info(f"Command succeeded")
            if result.stdout:
                log.debug(f"STDOUT: {result.stdout[:500]}")  # Log first 500 chars
            return True, result.stdout, result.stderr
        else:
            log.warning(f"Command failed with exit code {result.returncode}")
            if result.stderr:
                log.warning(f"STDERR: {result.stderr[:500]}")
            return False, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        log.error(f"Command timed out after {timeout} seconds")
        return False, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        log.error(f"Command execution failed: {e}")
        return False, "", str(e)


def checkout_aiter(
    commit: str,
    docker_container: str,
    aiter_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Pinned-commit checkout of the aiter repo.

    aiter_path resolution order:
      1. explicit ``aiter_path`` argument
      2. ``AKA_AITER_PATH`` env var
    No baked-in default — if neither is provided, returns False with a clear
    error message. (Reviewer note: previously hardcoded to /sgl-workspace/aiter.)
    """
    if aiter_path is None:
        aiter_path = os.environ.get("AKA_AITER_PATH")
    if not aiter_path:
        log = logger or logging.getLogger(__name__)
        log.error(
            "aiter path is not configured: pass aiter_path explicitly or set "
            "the AKA_AITER_PATH env var to the absolute path of the aiter repo."
        )
        return False

    log = logger or logging.getLogger(__name__)

    # Detect if we're already inside the container (no docker CLI available)
    inside_container = not shutil.which("docker")

    if not inside_container:
        # Verify container is running
        check = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", docker_container],
            capture_output=True, text=True,
        )
        if check.returncode != 0 or "true" not in check.stdout.lower():
            log.error(f"Docker container '{docker_container}' is not running")
            return False

    # Checkout the requested commit.
    # Always reset + clean to avoid stale files conflicting with new commit
    # (e.g. rope.py file coexisting with rope/ directory after branch switch).
    # Also clear __pycache__ to avoid stale bytecode.
    checkout_cmd = (
        f"cd {aiter_path} && git reset --hard && git clean -fd"
        f" && git checkout --quiet {commit}"
        f" && find . -name __pycache__ -type d -exec rm -rf {{}} + 2>/dev/null; true"
    )
    if inside_container:
        result = subprocess.run(
            ["bash", "-c", checkout_cmd],
            capture_output=True, text=True, timeout=60,
        )
    else:
        result = subprocess.run(
            ["docker", "exec", docker_container, "bash", "-c", checkout_cmd],
            capture_output=True, text=True, timeout=60,
        )
    if result.returncode != 0:
        log.warning(f"git checkout {commit[:12]} failed, trying hard reset")
        reset_cmd = f"cd {aiter_path} && git reset --hard && git clean -fd && git checkout {commit}"
        if inside_container:
            result = subprocess.run(
                ["bash", "-c", reset_cmd],
                capture_output=True, text=True, timeout=60,
            )
        else:
            result = subprocess.run(
                ["docker", "exec", docker_container, "bash", "-c", reset_cmd],
                capture_output=True, text=True, timeout=60,
            )
        if result.returncode != 0:
            log.error(f"Failed to checkout aiter {commit[:12]}: {result.stderr[:300]}")
            return False

    log.info(f"aiter checked out to {commit[:12]} in {docker_container}")
    return True
