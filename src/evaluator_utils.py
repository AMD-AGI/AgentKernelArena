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

