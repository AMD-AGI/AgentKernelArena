# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# This script will setup environment tools and dependencies. It will also provide duplicated workspace for the agent
import os
import shutil
import subprocess
import logging
from pathlib import Path

import yaml
from typing import Any, Optional

from src.perf_helper_materialization import materialize_perf_helpers_in_workspace


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


def _detect_gfx_arch_from_rocminfo() -> str | None:
    """Detect the actual GPU gfx arch from rocminfo (e.g. 'gfx950')."""
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith("Name:") and "gfx" in stripped:
                    arch = stripped.split("Name:")[-1].strip()
                    if arch.startswith("gfx"):
                        return arch
    except Exception:
        pass
    return None


_ROCM_ARCH_ENV_VARS = ("PYTORCH_ROCM_ARCH", "AMDGPU_TARGETS", "GPU_TARGETS")


def setup_rocm_env(target_gpu_model: str, logger: logging.Logger) -> None:
    """
    Set the ROCm GPU-arch environment for correct compilation. Exports
    all three of:
        - ``PYTORCH_ROCM_ARCH``  (PyTorch / torch.utils.cpp_extension)
        - ``AMDGPU_TARGETS``     (CMake / HIP)
        - ``GPU_TARGETS``        (CMake / HIP)

    All three are exported together regardless of how the arch was
    resolved, so CMake-based HIP builds always see the same arch as
    PyTorch — including on the common case where ``rocminfo`` succeeds.

    Resolution priority:
        1. Auto-detect from ``rocminfo`` (most reliable — uses actual
           hardware).
        2. Fall back to cheatsheet lookup from ``target_gpu_model``.
        3. Leave the environment unchanged if neither works.
    """
    detected_arch = _detect_gfx_arch_from_rocminfo()
    if detected_arch:
        gfx_arch = detected_arch
        source = "auto-detected from rocminfo"
    else:
        gfx_arch = _resolve_gfx_arch(target_gpu_model)
        if not gfx_arch:
            logger.warning(
                f"Could not resolve gfx arch for GPU model '{target_gpu_model}'. "
                f"None of {_ROCM_ARCH_ENV_VARS} will be set; PyTorch and CMake "
                "will fall back to their built-in arch lists."
            )
            return
        source = f"from target_gpu_model={target_gpu_model}"

    for var in _ROCM_ARCH_ENV_VARS:
        os.environ[var] = gfx_arch
    logger.info(
        f"Set {', '.join(f'{v}={gfx_arch}' for v in _ROCM_ARCH_ENV_VARS)} ({source})"
    )


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


def _ensure_repo_seeded_from_image(
    image_path: Path,
    target_dir: Path,
    logger: logging.Logger,
    exclude_paths: tuple[str, ...] = (),
) -> bool:
    """Seed the repo working tree from an in-image source tree (image_kernel tasks).

    Unlike ``_ensure_repo_cloned`` (which fetches a fresh shallow copy from a git
    URL), this copies a tree that already ships inside the container image — so
    submodules, third-party headers, and any prebuilt build cache are present and
    no network/clone/submodule step is needed. ``.git`` and any task-declared
    disposable cache paths are excluded during the copy so multi-GPU runs do not
    duplicate gigabytes of history or rebuildable artifacts in every workspace.

    Copies once and skips if the target is already populated (reused by all runs).

    Returns:
        True if a fresh copy was performed, False if the target already existed.
    """
    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info(f"Image repo already seeded at {target_dir}, skipping copy")
        return False

    if not image_path.is_dir():
        raise RuntimeError(
            f"image_repo_path does not exist in the image (not a directory): {image_path}"
        )

    if target_dir.exists():
        # empty dir or stray file — clear it so copytree can create it cleanly
        if target_dir.is_dir():
            shutil.rmtree(target_dir)
        else:
            target_dir.unlink()

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Seeding repo from in-image path {image_path} -> {target_dir}")
    excluded = tuple(sorted({".git", *exclude_paths}))
    if excluded:
        logger.info("Excluding disposable image-repo paths: %s", list(excluded))
    # Prefer copy-on-write (reflink) — an in-image repo can be multi-GB, and CoW
    # makes seeding near-instant / space-free on filesystems that support it.
    # Fall back to a regular recursive copy otherwise. Excluded paths are never
    # copied in either path (rather than copied first and deleted afterwards).
    if not _reflink_copy_tree(image_path, target_dir, logger, excluded):
        shutil.copytree(
            image_path,
            target_dir,
            symlinks=True,
            ignore=_image_repo_copy_ignore(image_path, excluded),
        )
    return True


def _normalize_image_repo_excludes(raw: Any) -> tuple[str, ...]:
    """Validate task-declared relative paths that are safe to omit while seeding."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise ValueError("image_repo_exclude must be a string or a list of strings")

    normalized: set[str] = set()
    for value in raw:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("image_repo_exclude entries must be non-empty strings")
        path = Path(value.strip())
        if path.is_absolute() or ".." in path.parts or path.as_posix() in ("", "."):
            raise ValueError(
                f"image_repo_exclude entries must be safe relative paths: {value!r}"
            )
        normalized.add(path.as_posix().strip("/"))
    return tuple(sorted(normalized))


def _image_repo_copy_ignore(src: Path, exclude_paths: tuple[str, ...]):
    """Build a copytree ignore callback for exact repo-relative paths."""
    excluded = set(exclude_paths)

    def ignore(directory: str, names: list[str]) -> set[str]:
        relative_dir = Path(directory).relative_to(src)
        ignored: set[str] = set()
        for name in names:
            relative = (relative_dir / name).as_posix()
            if name == ".git" or relative in excluded:
                ignored.add(name)
        return ignored

    return ignore


def _reflink_copy_tree(
    src: Path,
    dst: Path,
    logger: logging.Logger,
    exclude_paths: tuple[str, ...] = (".git",),
) -> bool:
    """Copy top-level entries via ``cp -a --reflink=auto`` without excluded paths.

    ``--reflink=auto`` uses copy-on-write when the filesystem supports it and
    silently falls back to a normal copy otherwise, so this is safe everywhere.
    Nested exclusions require ``shutil.copytree`` and return False here; top-level
    exclusions (including ``.git``) are omitted from the cp argument list, so they
    are never temporarily materialized on quota-limited workspaces.
    Returns True on success, False if ``cp`` is unavailable or failed (caller
    then uses ``shutil.copytree``). ``dst`` must not already exist.
    """
    if shutil.which("cp") is None or any("/" in path for path in exclude_paths):
        return False
    excluded_names = set(exclude_paths)
    entries = [entry for entry in src.iterdir() if entry.name not in excluded_names]
    dst.mkdir()
    if not entries:
        return True
    try:
        subprocess.run(
            ["cp", "-a", "--reflink=auto", "--", *(str(entry) for entry in entries), str(dst)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError) as e:
        stderr = getattr(e, "stderr", "") or ""
        logger.warning(
            "reflink copy failed (%s%s); falling back to copytree",
            e,
            f": {stderr.strip()}" if stderr.strip() else "",
        )
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        return False
    return True


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
    image_repo_path = task_config.get("image_repo_path")
    repo_url = task_config.get("repo_url")
    if image_repo_path:
        if not isinstance(image_repo_path, str) or not image_repo_path.strip():
            raise ValueError(
                f"Invalid image_repo_path in {task_config_dir}: {image_repo_path!r}"
            )
        repo_subdir = task_config.get("repo_subdir") or Path(image_repo_path).name
        repo_dir = workspace_path / repo_subdir
        excludes = _normalize_image_repo_excludes(task_config.get("image_repo_exclude"))
        did_seed = _ensure_repo_seeded_from_image(
            Path(image_repo_path), repo_dir, logger, excludes
        )
        _maybe_post_clone_install(task_config, repo_dir, did_seed, logger)
        return repo_dir
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


def get_task_workspace_path(run_directory: Path, task_name: str, timestamp: str) -> Path:
    """Return the expected task workspace path for a run."""
    sanitized = _sanitize_task_name(task_name)
    return run_directory / f"{sanitized}_{timestamp}"


def is_task_complete(
    run_directory: Path,
    task_name: str,
    timestamp: str,
    agent_name: str | None = None,
) -> bool:
    """
    Check if a task is already completed.

    Args:
        run_directory: Run-level directory (e.g., workspace_MI300_cursor/run_20250115_143022/)
        task_name: Full task name (e.g., "hip2hip/gpumode/SiLU")
        timestamp: Timestamp string used in task directory name
        agent_name: Agent name. task_validator uses validation_report.yaml;
            all other agents use task_result.yaml.

    Returns:
        True if the expected completion report exists, False otherwise
    """
    task_dir = get_task_workspace_path(run_directory, task_name, timestamp)
    report_name = "validation_report.yaml" if agent_name == "task_validator" else "task_result.yaml"
    result_file = task_dir / report_name
    return result_file.exists()


def setup_workspace(task_config_dir: str, run_directory: Path, timestamp: str, logger: logging.Logger,
                    task_name: str = "") -> Path:
    """
    Setup workspace for agent execution by duplicating the task directory.

    Repo materialization depends on the task's source:
      - repo_url (repository tasks): clone once into a tasks/ CACHE (reused across
        runs to avoid re-cloning over the network), then copy into the workspace.
      - image_repo_path (image_kernel tasks): the in-image tree is already a single
        shared, read-only source, so it is seeded DIRECTLY into each run's
        workspace with NO tasks/ cache (avoids multi-GB per-task duplication and
        keeps the repo tree clean).

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

    # 1. Determine the repo subdir + source.
    #    Two sources are supported:
    #      - image_repo_path: an in-image source tree (image_kernel tasks)
    #      - repo_url:        a git URL (repository tasks)
    image_repo_path = task_config.get("image_repo_path")
    repo_url = task_config.get("repo_url")
    repo_subdir = None
    if image_repo_path:
        repo_subdir = task_config.get("repo_subdir") or Path(image_repo_path).name
    elif repo_url:
        repo_subdir = task_config.get("repo_subdir") or _extract_repo_name(repo_url)

    # repo_url (repository) tasks: keep the tasks/ clone CACHE. Cloning is a network
    # operation, so clone once and reuse across runs.
    # image_kernel tasks: do NOT cache. The in-image tree (image_repo_path) is
    # already a single shared, read-only source used by every image_kernel task, so
    # a per-task copy under tasks/ is pure duplication (multi-GB) and pollutes the
    # repo tree. It is seeded DIRECTLY into each run's workspace in step 4 instead.
    if repo_url and not image_repo_path:
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

    # 3. Copy the task folder to the workspace. For image_kernel the repo is seeded
    #    directly in step 4, so skip its subdir here — this also avoids copying any
    #    stale per-task cache left by an older version of this code.
    skip_names = {repo_subdir} if (image_repo_path and repo_subdir) else set()
    for item in task_folder.iterdir():
        if item.name in skip_names:
            continue
        dst = workspace_path / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)
    logger.info(f"Copied task folder content from {task_folder} to {workspace_path}")

    # 4. image_kernel: seed the in-image source tree DIRECTLY into the workspace
    #    (no tasks/ cache). One copy per run; nothing persisted under tasks/.
    if image_repo_path:
        repo_dir = workspace_path / repo_subdir
        excludes = _normalize_image_repo_excludes(task_config.get("image_repo_exclude"))
        did_seed = _ensure_repo_seeded_from_image(
            Path(image_repo_path), repo_dir, logger, excludes
        )
        _maybe_post_clone_install(task_config, repo_dir, did_seed, logger)

    materialize_perf_helpers_in_workspace(workspace_path, logger=logger)

    return workspace_path
