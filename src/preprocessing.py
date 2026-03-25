# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# This script will setup environment tools and dependencies. It will also provide duplicated workspace for the agent
import os
import shutil
import subprocess
import logging
from pathlib import Path

import yaml
from typing import Any, Optional


def _resolve_gfx_arch(target_gpu_model: str) -> str | None:
    """
    Look up the gfx architecture token (e.g. 'gfx942') for a given GPU model
    name (e.g. 'MI300') from default_cheatsheet.yaml.

    Returns None if the GPU model is not found.
    """
    cheatsheet_path = (
        Path(__file__).resolve().parent / "prompts" / "cheatsheet" / "default_cheatsheet.yaml"
    )
    try:
        config = yaml.safe_load(cheatsheet_path.read_text()) or {}
    except Exception:
        return None

    arch_map = config.get("architecture", {})
    gpu_key = str(target_gpu_model)
    entry = (
        arch_map.get(gpu_key)
        or arch_map.get(gpu_key.upper())
        or arch_map.get(gpu_key.lower())
    )
    if isinstance(entry, dict):
        return entry.get("gfx_arch")
    return None


def setup_rocm_env(target_gpu_model: str, logger: logging.Logger) -> None:
    """
    Set PYTORCH_ROCM_ARCH (and related env vars) based on config.yaml's
    target_gpu_model so that torch.utils.cpp_extension.load() and hipcc
    compile for the correct GPU architecture.

    Should be called once at the start of main(), before any task is launched.
    """
    gfx_arch = _resolve_gfx_arch(target_gpu_model)
    if not gfx_arch:
        logger.warning(
            f"Could not resolve gfx arch for GPU model '{target_gpu_model}'. "
            "PYTORCH_ROCM_ARCH will not be set; PyTorch will fall back to its built-in arch list."
        )
        return

    os.environ["PYTORCH_ROCM_ARCH"] = gfx_arch
    logger.info(f"Set PYTORCH_ROCM_ARCH={gfx_arch} (from target_gpu_model={target_gpu_model})")


def check_environment() -> None:
    # check hipcc, rocprof-compute
    if "hipcc" not in os.environ["PATH"]:
        raise ValueError("hipcc is not in the PATH")
    if "rocprof-compute" not in os.environ["PATH"]:
        raise ValueError("rocprof-compute is not in the PATH")
    pass


def _extract_repo_name(repo_url: str) -> str:
    """Extract repository name from URL (e.g. 'https://github.com/ROCm/rocPRIM.git' -> 'rocPRIM')."""
    # Remove trailing slashes and .git suffix
    url = repo_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    # Extract last path component
    return url.rsplit("/", 1)[-1]


def _ensure_repo_cloned(repo_url: str, target_dir: Path, logger: logging.Logger) -> tuple[Path, bool]:
    """
    Ensure repo is cloned to target_dir. Skip if already exists.
    
    Args:
        repo_url: Git repository URL
        target_dir: Directory to clone into
        logger: Logger instance

    Returns:
        (Path to the repository directory, whether a fresh clone was performed)
    """
    if (target_dir / ".git").exists():
        logger.info(f"Repository already exists at {target_dir}, skipping clone")
        return target_dir, False

    if target_dir.exists():
        logger.warning(
            f"Path {target_dir} exists but is not a git repository; removing and re-cloning"
        )
        if target_dir.is_dir():
            shutil.rmtree(target_dir)
        else:
            target_dir.unlink()

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cloning {repo_url} into {target_dir}")
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--single-branch",
                "--no-tags",
                repo_url,
                str(target_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"git clone failed: {(e.stderr or '').strip()}") from e

    return target_dir, True


def _normalize_post_clone_install_commands(raw: Any) -> list[str]:
    """Parse post_clone_install from YAML into a list of non-empty shell command strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, list):
        return [c.strip() for c in raw if isinstance(c, str) and c.strip()]
    raise ValueError("post_clone_install must be a string or a list of strings")


def _run_post_clone_install(
    commands: list[str],
    cwd: Path,
    logger: logging.Logger,
) -> None:
    """
    Run shell commands (e.g. apt install) after clone. Task authors control these commands;
    they run with shell=True in the repo root by default.
    """
    for i, cmd in enumerate(commands):
        logger.info("=" * 60)
        logger.info(f"post_clone_install [{i + 1}/{len(commands)}] (cwd={cwd})")
        logger.info(cmd)
        logger.info("=" * 60)
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                check=True,
                cwd=str(cwd),
                text=True,
                capture_output=True,
                timeout=7200,
            )
            if proc.stdout:
                logger.info(proc.stdout.rstrip())
            if proc.stderr:
                logger.info(proc.stderr.rstrip())
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout or "").strip()
            logger.error(f"post_clone_install failed with exit code {e.returncode}\n{err}")
            raise RuntimeError(
                f"post_clone_install step {i + 1} failed: {cmd[:120]}..."
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"post_clone_install timed out: {cmd[:120]}...") from e


def _maybe_post_clone_install(
    task_config: dict,
    repo_path: Path,
    did_clone: bool,
    logger: logging.Logger,
) -> None:
    """
    If task config defines post_clone_install, run shell commands to install OS / tooling deps.

    Keys:
      post_clone_install: str | list[str] — shell commands (e.g. apt-get install cmake)
      post_clone_install_mode: "after_clone" | "every_setup" (default: after_clone)
        - after_clone: only run when a new git clone just completed
        - every_setup: run on every setup_workspace (use guarded commands, e.g. `command -v cmake || ...`)
    """
    commands = _normalize_post_clone_install_commands(task_config.get("post_clone_install"))
    if not commands:
        return

    mode = task_config.get("post_clone_install_mode", "after_clone")
    if mode not in ("after_clone", "every_setup"):
        raise ValueError(
            "post_clone_install_mode must be 'after_clone' or 'every_setup' "
            f"(got {mode!r})"
        )
    if mode == "after_clone" and not did_clone:
        logger.info(
            "Skipping post_clone_install (repository already present; "
            "post_clone_install_mode=after_clone). "
            "Use post_clone_install_mode=every_setup to run on every setup, "
            "or guarded commands that no-op when deps exist."
        )
        return

    logger.info("Running post_clone_install (system packages / environment)")
    _run_post_clone_install(commands, repo_path, logger)


def setup_repo_from_config(
    task_config_dir: str, workspace_path: Path, logger: logging.Logger
) -> Optional[Path]:
    """
    If task has repo_url, ensure repo exists in the workspace and return its path.

    Returns None if the task does not specify repo_url.
    """
    with open(task_config_dir, "r") as f:
        task_config = yaml.safe_load(f) or {}
    repo_url = task_config.get("repo_url")
    if not repo_url:
        return None
    if not isinstance(repo_url, str) or not repo_url.strip():
        raise ValueError(f"Invalid repo_url in {task_config_dir}: {repo_url!r}")
    repo_subdir = task_config.get("repo_subdir") or _extract_repo_name(repo_url)
    repo_dir = workspace_path / repo_subdir
    _, did_clone = _ensure_repo_cloned(repo_url, repo_dir, logger)
    _maybe_post_clone_install(task_config, repo_dir, did_clone, logger)
    return repo_dir


def _sanitize_task_name(task_name: str) -> str:
    """Convert a task name like 'hip2hip/gpumode/SiLU' to 'hip2hip_gpumode_SiLU' for use in directory names."""
    return task_name.replace("/", "_")


def is_task_complete(run_directory: Path, task_name: str, timestamp: str) -> bool:
    """
    Check if a task is already completed.

    Args:
        run_directory: Run-level directory (e.g., workspace_MI300_cursor/run_20250115_143022/)
        task_name: Full task name (e.g., "hip2hip/gpumode/SiLU")
        timestamp: Timestamp string used in task directory name

    Returns:
        True if task directory exists and task_result.yaml exists, False otherwise
    """
    sanitized = _sanitize_task_name(task_name)
    task_dir = run_directory / f"{sanitized}_{timestamp}"
    result_file = task_dir / "task_result.yaml"
    return result_file.exists()


def setup_workspace(task_config_dir: str, run_directory: Path, timestamp: str, logger: logging.Logger,
                    task_name: str = "") -> Path:
    """
    Setup workspace for agent execution by duplicating task directory.

    For tasks with repo_url:
      1. Clone repo into tasks/ directory (if not already cloned)
      2. Optionally run post_clone_install (see task config)
      3. Copy entire task folder (including repo) to workspace

    Args:
        task_config_dir: Path to task's config.yaml
        run_directory: Run-level directory (e.g., workspace_MI300_cursor/run_20250115_143022/)
        timestamp: Timestamp string for unique workspace naming
        logger: Logger instance
        task_name: Full task name (e.g., "hip2hip/gpumode/SiLU") for unique directory naming

    Returns:
        Path to the created workspace directory
    """
    task_config_path = Path(task_config_dir)
    task_folder = task_config_path.parent

    # Load task config
    with open(task_config_path, "r") as f:
        task_config = yaml.safe_load(f) or {}

    # 1. Clone repo into tasks/ directory if needed (only once, reused by all runs)
    repo_url = task_config.get("repo_url")
    if repo_url:
        repo_subdir = task_config.get("repo_subdir") or _extract_repo_name(repo_url)
        repo_in_tasks = task_folder / repo_subdir
        _, did_clone = _ensure_repo_cloned(repo_url, repo_in_tasks, logger)
        _maybe_post_clone_install(task_config, repo_in_tasks, did_clone, logger)

    # 2. Create workspace directory
    if task_name:
        new_folder_name = f"{_sanitize_task_name(task_name)}_{timestamp}"
    else:
        new_folder_name = f"{task_folder.name}_{timestamp}"
    workspace_path = run_directory / new_folder_name
    workspace_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created workspace directory: {workspace_path}")

    # 3. Copy entire task folder (including cloned repo) to workspace
    for item in task_folder.iterdir():
        dst = workspace_path / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)

    logger.info(f"Copied task folder content from {task_folder} to {workspace_path}")

    return workspace_path
