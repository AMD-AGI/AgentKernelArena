# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Utilities for evaluator: command execution and file I/O.
"""
import subprocess
import logging
import yaml
from pathlib import Path
from typing import Tuple, Optional, List
from .testcases import TestCaseResult


def run_command(
    command: str,
    workspace: Path,
    timeout: int = 300,
    logger: Optional[logging.Logger] = None,
    docker_container: Optional[str] = None,
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

    Returns:
        Tuple of (success: bool, stdout: str, stderr: str)
    """
    log = logger or logging.getLogger(__name__)

    try:
        if docker_container:
            escaped = command.replace("'", "'\\''")
            abs_workspace = Path(workspace).resolve()
            command = (
                f"docker exec -w {abs_workspace} {docker_container} "
                f"bash -c '{escaped}'"
            )
            log.info(f"Running in Docker [{docker_container}]: {command[:200]}")
        else:
            log.info(f"Running command: {command}")
        log.info(f"Working directory: {workspace}")

        result = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout
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
    aiter_path: str = "/sgl-workspace/aiter",
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Checkout a specific aiter commit inside the Docker container.

    Returns True on success, False on failure (container not running, git error).
    """
    log = logger or logging.getLogger(__name__)

    # Verify container is running
    check = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", docker_container],
        capture_output=True, text=True,
    )
    if check.returncode != 0 or "true" not in check.stdout.lower():
        log.error(f"Docker container '{docker_container}' is not running")
        return False

    # Checkout the requested commit (skip fetch — container may lack network)
    checkout_cmd = f"cd {aiter_path} && git checkout --quiet {commit} 2>&1"
    result = subprocess.run(
        ["docker", "exec", docker_container, "bash", "-c", checkout_cmd],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        log.warning(f"git checkout {commit[:12]} failed, trying hard reset")
        reset_cmd = f"cd {aiter_path} && git reset --hard && git clean -fd && git checkout {commit}"
        result = subprocess.run(
            ["docker", "exec", docker_container, "bash", "-c", reset_cmd],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            log.error(f"Failed to checkout aiter {commit[:12]}: {result.stderr[:300]}")
            return False

    log.info(f"aiter checked out to {commit[:12]} in {docker_container}")
    return True

