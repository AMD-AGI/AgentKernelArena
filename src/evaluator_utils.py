# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Utilities for evaluator: command execution and file I/O.
"""
import subprocess
import logging
import yaml
import shlex
from pathlib import Path
from typing import Tuple, Optional, List
from .testcases import TestCaseResult
from .runtime_env import PYTHON_ENV_VAR, build_subprocess_env


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
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, str, str]:
    """
    Run a shell command in the workspace directory.
    
    Args:
        command: Shell command to execute
        workspace: Working directory
        timeout: Command timeout in seconds
        logger: Optional logger for output
        
    Returns:
        Tuple of (success: bool, stdout: str, stderr: str)
    """
    log = logger or logging.getLogger(__name__)
    
    try:
        env = build_subprocess_env()
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
