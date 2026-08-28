# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Shared machinery for the KernelForge-backed Arena agents.

``forge`` drives ``kernel-agents forge-loop`` and ``forge_rewrite`` drives
``kernel-agents forge-rewrite-by-flydsl``. Both resolve the same GPU identity,
prepare the same kind of git workspace, stream and hard-kill the same kind of
subprocess tree, and read the same ``__FORGE_RESULT__`` contract, so that part
lives here and each launcher only owns its own CLI and result handling.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

FORGE_RESULT_SENTINEL = "__FORGE_RESULT__"
KB_STATUS_FILE = "arena_forge_status.json"
FORGE_SHUTDOWN_MARGIN_SECONDS = 900

_GPU_TYPE_ALIASES = {
    # Arena historically accepts the family-style names below for the X SKUs.
    # KernelForge addresses KB records by exact hardware model, so normalize the
    # aliases before they become distinct, non-interoperable recipe identities.
    "mi300": "mi300x",
    "mi325": "mi325x",
}

# Task types whose name does not encode the FlyDSL/HIP/Triton target after a
# '2'. Without an entry the whole task_type string would be used as the fellow
# backend, producing a fellow KernelForge does not define.
_EXPLICIT_BACKEND = {
    "rewrite_by_flydsl": "flydsl",
}


def _normalize_gfx_arch(arch: str) -> str:
    """Normalize rocminfo/config variants to the Forge KB architecture token."""
    match = re.search(r"gfx[0-9a-f]+", str(arch or "").lower())
    if not match:
        raise ValueError(f"Invalid AMD GPU architecture: {arch!r}")
    return match.group(0)


def _resolve_gpu_arch(eval_config: dict[str, Any]) -> str:
    """Resolve the gfx arch for the forge run, reusing Arena's shared resolution.

    Priority mirrors ``src.preprocessing.setup_rocm_env`` so ``--gpu-target`` always
    matches the arch Arena actually compiles/runs for:

      1. the real hardware arch reported by ``rocminfo`` (most reliable); then
      2. the configured ``target_gpu_model`` looked up in the shared architecture
         map in ``default_cheatsheet.yaml`` (the single source of truth — covers
         MI300/MI325/MI355X, RDNA4->gfx1201, ...).

    An unknown model FAILS EXPLICITLY instead of silently assuming an arch: a
    mismatched arch (e.g. handing RDNA4 the gfx942 profile) produces invalid build
    flags, misleading agent guidance, and kernels tuned for the wrong ISA.
    """
    from src.preprocessing import _detect_gfx_arch_from_rocminfo, _resolve_gfx_arch

    detected = _detect_gfx_arch_from_rocminfo()
    if detected:
        return _normalize_gfx_arch(detected)

    model = str(eval_config.get("target_gpu_model", "")).strip()
    arch = _resolve_gfx_arch(model)
    if arch:
        return _normalize_gfx_arch(arch)

    raise ValueError(
        f"Cannot resolve a gfx arch for target_gpu_model={model!r}: it was not "
        "detected via rocminfo and is not defined under 'architecture' in "
        "src/prompts/cheatsheet/default_cheatsheet.yaml. Add the model there (with "
        "its gfx_arch) or run on the target GPU so rocminfo can report it."
    )


def _resolve_gpu_type(eval_config: dict[str, Any]) -> str:
    """Return Arena's hardware model in Forge's canonical KB token form."""
    raw = str(eval_config.get("target_gpu_model") or "").strip().lower()
    if not raw or not re.fullmatch(r"[a-z0-9][a-z0-9._+-]*", raw):
        raise ValueError(
            "target_gpu_model must be a non-empty hardware model token "
            f"for Forge KB identity; got {eval_config.get('target_gpu_model')!r}"
        )
    return _GPU_TYPE_ALIASES.get(raw, raw)


def _process_group_exists(pgid: int) -> bool:
    """Return whether a process group still has signalable members."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(
    process: subprocess.Popen,
    pgid: int,
    timeout: float,
) -> bool:
    """Wait for the complete group to exit, reaping its leader along the way."""
    deadline = time.monotonic() + timeout
    while True:
        process.poll()
        if not _process_group_exists(pgid):
            return True

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.1, remaining))


def _terminate_process_group(
    process: subprocess.Popen,
    logger: logging.Logger,
    term_timeout: float = 10,
    kill_timeout: float = 5,
) -> None:
    """Terminate the forge process and ALL its descendants (kernel-agents, claude, GPU).

    The subprocess is launched in its own session (``start_new_session=True``), so
    its PID is the process-group leader. Signalling the whole group (SIGTERM, then
    SIGKILL after a grace period) terminates the deep child tree; signalling only the
    leader would orphan those children, which would keep holding the GPU and could
    keep editing the workspace while Arena runs git checkout and final scoring.
    """
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return  # already exited

    def _signal_group(sig: int) -> bool:
        try:
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            return False  # group already gone

    if not _signal_group(signal.SIGTERM):
        return

    if _wait_for_process_group_exit(process, pgid, term_timeout):
        return

    logger.warning("Force killing forge process group (SIGKILL)")
    if not _signal_group(signal.SIGKILL):
        return

    if not _wait_for_process_group_exit(process, pgid, kill_timeout):
        logger.warning("Forge process group did not exit even after SIGKILL")


def _forge_max_hours(agent_config: dict[str, Any]) -> float:
    """Derive the KernelForge ``--max-hours`` budget from the run's timeout.

    ``timeout_seconds`` is the single time budget (the API/bootstrap patches it
    per-run, e.g. 115200 for a 32h run). ``--max-hours`` tracks it with a small
    margin so the run self-stops (BUDGET EXHAUSTED) just before the hard
    process-wait kill instead of being killed mid-iteration. A fixed hours value
    would ignore the per-run timeout and cap long runs early (a 32h run would
    stop at ~8h). The default timeout (29700s) yields ~8h.
    """
    timeout_s = float(agent_config.get("timeout_seconds", 3600))
    # KernelForge enforces a one-hour minimum. The Arena hard timeout remains the
    # final authority for shorter smoke runs.
    loop_seconds = max(
        3600.0,
        timeout_s - FORGE_SHUTDOWN_MARGIN_SECONDS,
    )
    return round(loop_seconds / 3600.0, 3)


def _repo_subdir_name(task_config: dict[str, Any]) -> str | None:
    """Best-effort name of the repo subdir a repository/image_kernel task lives in."""
    explicit = task_config.get("repo_subdir")
    if explicit:
        return str(explicit)
    image_repo_path = task_config.get("image_repo_path")
    if image_repo_path:
        return Path(str(image_repo_path)).name
    repo_url = task_config.get("repo_url")
    if repo_url:
        url = str(repo_url).rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]
        return url.rsplit("/", 1)[-1]
    return None


def _resolve_one_source_file(workspace: str, rel, task_config: dict[str, Any]) -> Path | None:
    """Resolve one source_file_path entry to an absolute workspace path.

    `source_file_path` entries may be given either workspace-relative (legacy
    snippet tasks copy the file to the workspace root) or repo-root-relative
    (repository / image_kernel tasks put the sources under a repo subdir).
    Resolution order:
      1. as given, relative to the workspace root (preserves legacy behavior);
      2. under the repo subdir (repo_subdir / image_repo_path / repo_url basename);
      3. a unique match anywhere in the workspace whose path ends with the given
         suffix (last-resort, ignores .git).
    Returns None if it cannot be resolved.
    """
    rel = str(rel)
    ws = Path(workspace)

    p = (ws / rel).resolve()
    if p.exists():
        return p

    subdir = _repo_subdir_name(task_config)
    if subdir:
        p2 = (ws / subdir / rel).resolve()
        if p2.exists():
            return p2

    tail = Path(rel)
    matches = [
        m for m in ws.rglob(tail.name)
        if str(m).endswith(rel) and ".git" not in m.parts
    ]
    if len(matches) == 1:
        return matches[0].resolve()

    return None


def _resolve_kernel_file(workspace: str, source_files: list, task_config: dict[str, Any]) -> Path:
    """Locate the anchor kernel file (source_file_path[0]); raise if not found."""
    p = _resolve_one_source_file(workspace, source_files[0], task_config)
    if p is None:
        raise RuntimeError(f"Kernel file not found in workspace: {Path(workspace) / str(source_files[0])}")
    return p


def _declared_editable_sources(task_config: dict[str, Any]) -> list[str]:
    """Return the complete ordered source allowlist for Forge edits."""
    primary = task_config.get("source_file_path") or []
    dependent = task_config.get("editable_sources") or []
    if not isinstance(primary, list) or not all(
        isinstance(path, str) and path.strip() for path in primary
    ):
        raise ValueError("source_file_path must be a list of non-empty paths")
    if not isinstance(dependent, list) or not all(
        isinstance(path, str) and path.strip() for path in dependent
    ):
        raise ValueError("editable_sources must be a list of non-empty paths")
    return list(
        dict.fromkeys(
            [
                *(path.strip() for path in primary),
                *(path.strip() for path in dependent),
            ]
        )
    )


def _resolve_all_source_files(
    workspace: str, source_files: list, task_config: dict[str, Any],
    logger: logging.Logger,
    *,
    strict: bool = False,
) -> list[Path]:
    """Resolve every declared editable source to an absolute workspace path.

    The first entry (anchor) must exist; extra entries that cannot be resolved
    are warned and skipped rather than failing the run (a task may list an
    optional/relocated file). Order-preserving, de-duplicated.
    """
    resolved: list[Path] = []
    workspace_root = Path(workspace).resolve()
    for i, rel in enumerate(source_files):
        p = _resolve_one_source_file(workspace, rel, task_config)
        if p is not None:
            try:
                p.relative_to(workspace_root)
            except ValueError as error:
                raise RuntimeError(
                    f"Editable source escapes the workspace: {p}"
                ) from error
            if p not in resolved:
                resolved.append(p)
        elif i == 0:
            raise RuntimeError(
                f"Anchor kernel file not found in workspace: {Path(workspace) / str(rel)}"
            )
        elif strict:
            raise RuntimeError(
                f"Producer editable source entry not found in workspace: {rel}"
            )
        else:
            logger.warning("forge: editable source entry not found, skipping: %s", rel)
    return resolved


def _strip_nested_git(workspace: str, logger: logging.Logger) -> None:
    """Remove any nested ``.git`` under the workspace (a cloned repo's own history).

    Repository tasks clone the upstream repo WITH its ``.git`` into the workspace.
    If left in place, forge's outer ``git init`` treats the repo dir as an
    embedded gitlink and does NOT track the files inside it — so the agent's edits
    to the real kernels are invisible to ``git add -u`` and keep/revert becomes a
    no-op. Stripping the nested ``.git`` lets the outer workspace git track the
    repo's files directly. Only the per-run workspace copy is touched; Arena's
    cached clone under ``tasks/`` is untouched. Never removes the workspace-root
    ``.git`` (which forge creates afterwards).
    """
    ws = Path(workspace).resolve()
    removed = 0
    for git_path in ws.rglob(".git"):
        if git_path.parent.resolve() == ws:
            continue  # never the outer workspace git (created later by forge)
        try:
            if git_path.is_dir():
                shutil.rmtree(git_path, ignore_errors=True)
            else:
                git_path.unlink()
            removed += 1
        except OSError as e:
            logger.warning(f"forge: failed to strip nested .git {git_path}: {e}")
    if removed:
        logger.info(
            f"forge: stripped {removed} nested .git so keep/revert tracks repo files"
        )


def _normalize_fellow_backend(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _infer_backend(task_config: dict[str, Any]) -> str:
    """Resolve the configured backend name that Arena forwards to KernelForge.

    Three task families need different signals:

      * Repository / image_kernel tasks ship a whole source tree, not a
        "<src>2<dst>" pair, so their explicit ``kernel_kind`` wins when present;
        otherwise ``repository_language`` describes the editable source language.
      * Task types that name their target explicitly (``rewrite_by_flydsl``) map
        through ``_EXPLICIT_BACKEND``.
      * Snippet tasks are "<source>2<target>" (triton2triton, cuda2hip,
        torch2hip, flydsl2flydsl, instruction2triton, ...); the optimized kernel
        is in the TARGET language, i.e. the part after the last '2'.

    Arena does not maintain KernelForge's supported-backend registry and does not
    substitute an unknown backend. KernelForge owns support validation.
    """
    task_type = _normalize_fellow_backend(task_config.get("task_type"))

    if task_type in ("image_kernel", "repository"):
        identity_config = task_config.get("kernel_identity") or {}
        if not isinstance(identity_config, dict):
            identity_config = {}
        kernel_kind = _normalize_fellow_backend(
            identity_config.get("kernel_kind")
            or task_config.get("kernel_kind")
            or ""
        )
        if kernel_kind:
            return kernel_kind

        repository_language = _normalize_fellow_backend(
            task_config.get("repository_language")
        )
        if repository_language:
            return repository_language
        raise ValueError(
            f"Task type {task_type!r} requires kernel_identity.kernel_kind, "
            "kernel_kind, or repository_language to select a Forge fellow"
        )

    explicit = _EXPLICIT_BACKEND.get(task_type)
    if explicit:
        return explicit

    target = task_type.rsplit("2", 1)[-1] if "2" in task_type else task_type
    if not target:
        raise ValueError("task_type is required to select a Forge fellow")
    return target


def _resolve_fellow(task_config: dict[str, Any], agent_config: dict[str, Any]) -> str:
    """Pick the fellow: explicit agent_config override wins, else inferred."""
    override = agent_config.get("fellow")
    if override:
        return str(override)
    return f"{_infer_backend(task_config)}-fellow"


def _task_kernel_identity(task_config: dict[str, Any]) -> dict[str, Any]:
    """Return optional task metadata forwarded to KernelForge."""
    value = task_config.get("kernel_identity") or {}
    if not isinstance(value, dict):
        raise ValueError("task kernel_identity configuration must be a mapping")
    return value


def _resolve_kernel_kind(task_config: dict[str, Any]) -> str:
    """Return the optional implementation kind supplied by the task."""
    identity_config = _task_kernel_identity(task_config)
    explicit = identity_config.get("kernel_kind", task_config.get("kernel_kind"))
    return str(explicit or "").strip().lower()


# Framework aliases whose identity forms the KB kernel-page slug component. Maps
# a path component to its canonical framework. Must match KernelForge's set and
# Hyperloom's _resolve_framework so a solution written here resolves to the SAME
# page a Hyperloom forge-loop reads. ``aiter_meta`` is aiter's C++/CK companion
# package and shares aiter's identity.
_FRAMEWORK_ALIASES = {
    "vllm": "vllm",
    "sglang": "sglang",
    "aiter": "aiter",
    "aiter_meta": "aiter",
}


def _resolve_framework(task_config: dict[str, Any]) -> str:
    """Return the optional source owner explicitly supplied by the task."""
    identity_config = _task_kernel_identity(task_config)
    explicit = str(
        identity_config.get("source_owner")
        or task_config.get("source_owner_framework")
        or ""
    ).strip()
    return _FRAMEWORK_ALIASES.get(explicit.lower(), explicit.lower()) if explicit else ""


def _logical_operator(task_config: dict[str, Any]) -> str:
    """Return the optional explicit logical identity."""
    identity_config = _task_kernel_identity(task_config)
    value = str(
        identity_config.get("logical_operator")
        or task_config.get("logical_operator")
        or ""
    ).strip()
    return _normalize_logical_operator(value)


def _normalize_logical_operator(value: str) -> str:
    """Match Hyperloom's balanced-template logical operation normalization."""
    raw = str(value or "").strip()
    if "<" not in raw:
        normalized = raw
    else:
        characters: list[str] = []
        depth = 0
        for character in raw:
            if character == "<":
                depth += 1
            elif character == ">":
                if depth > 0:
                    depth -= 1
            elif depth == 0:
                characters.append(character)
        normalized = "".join(characters).strip() or raw
    normalized = re.sub(r"\s*::\s*", "::", normalized)
    normalized = re.sub(r":{3,}", "::", normalized)
    return normalized.strip(": ")


def _git(workspace: str, *args: str, logger: logging.Logger) -> None:
    """Run a git command in the workspace, tolerating non-zero exit."""
    result = subprocess.run(
        ["git", *args], cwd=workspace, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        logger.debug(f"git {' '.join(args)} -> {result.returncode}: {result.stderr.strip()}")


def _git_required(workspace: str, *args: str) -> str:
    """Run a git query whose failure must reject the Forge result."""
    result = subprocess.run(
        ["git", *args], cwd=workspace, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Forge workspace integrity check failed: git {' '.join(args)} "
            f"returned {result.returncode}: {result.stderr.strip()}"
        )
    return result.stdout


def _capture_forge_edit_baseline(workspace: str) -> str:
    """Return the immutable commit Arena created before Forge starts editing."""
    baseline = _git_required(workspace, "rev-parse", "--verify", "HEAD").strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", baseline):
        raise RuntimeError(f"Invalid Forge workspace baseline commit: {baseline!r}")
    return baseline


def _verify_forge_edit_scope(
    workspace: str,
    baseline_commit: str,
    editable_sources: list[Path],
    logger: logging.Logger | None = None,
) -> None:
    """Enforce Arena's declared source allowlist before final scoring.

    KernelForge treats ``--source-files`` as orientation and KB metadata rather
    than an edit boundary. Arena owns that boundary: only files resolved from
    ``source_file_path`` plus ``editable_sources`` may differ from the initial
    workspace snapshot. The check includes committed, staged, unstaged, deleted,
    and renamed paths. Non-ignored untracked scratch files are discarded, matching
    Arena's harness guard: they did not exist at baseline and cannot influence the
    score after removal.
    """
    root = Path(workspace).resolve()
    allowed: set[str] = set()
    for source in editable_sources:
        resolved = Path(source).resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"Editable source escapes the Forge workspace: {resolved}"
            ) from error
        allowed.add(relative.as_posix())

    changed_output = _git_required(
        workspace,
        "diff",
        "--name-only",
        "--no-renames",
        "-z",
        baseline_commit,
        "--",
    )
    untracked_output = _git_required(
        workspace,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    changed = {path for path in changed_output.split("\0") if path}
    untracked = {path for path in untracked_output.split("\0") if path}
    for relative in sorted(untracked - allowed):
        scratch = root / relative
        try:
            scratch.unlink()
        except OSError as error:
            raise RuntimeError(
                f"Could not discard undeclared Forge scratch file: {relative}"
            ) from error
        if logger is not None:
            logger.warning(
                "Discarded undeclared Forge scratch file before scoring: %s",
                relative,
            )

    violations = sorted(changed - allowed)
    if violations:
        raise RuntimeError(
            "Forge changed files outside source_file_path/editable_sources; "
            f"Arena refuses to score this result: {violations}"
        )


# Build artifacts / regenerated reports / forge scaffolding must NOT be tracked:
# if they are, a validation or benchmark run that regenerates them dirties the
# tree and makes the loop's `git revert` fail — leaking a reverted (often broken)
# edit into the final tree. Only source is tracked, matching the loop's own
# `git add -u` philosophy.
#
# The rewrite pipeline's scratch directories are listed for a second reason:
# `_verify_forge_edit_scope` deletes untracked files that are not declared
# editable sources, which would remove the ported kernel before Arena reads it.
_GITIGNORE = """\
__pycache__/
*.pyc
*.pyo
*.so
*.o
*.hsaco
*.pt
build/
perf/
*_perf.yaml
performance_report.json
perf_report.json
forge_experiments/
forge_driver.py
forge_rewrite_ws/
.forge_rewrite/
.pytest_cache/
*.log
"""


def _init_git_workspace(workspace: str, logger: logging.Logger) -> None:
    """Initialize a git repo with an initial commit (required by forge-loop).

    Writes a .gitignore first so build artifacts and regenerated perf reports
    stay untracked — otherwise later tool runs dirty the tree and break the
    loop's keep/revert (git revert aborts on unstaged changes).
    """
    if not (Path(workspace) / ".git").exists():
        _git(workspace, "init", logger=logger)
    gitignore = Path(workspace) / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(_GITIGNORE)
    # Local identity so commits succeed without global git config.
    _git(workspace, "config", "user.email", "forge-loop@local", logger=logger)
    _git(workspace, "config", "user.name", "forge-loop", logger=logger)
    # Untrack anything already staged/committed that the .gitignore now excludes
    # (e.g. build/ created by Arena's baseline step before this init).
    _git(workspace, "rm", "-r", "--cached", "--quiet", ".", logger=logger)
    _git(workspace, "add", "-A", logger=logger)
    _git(workspace, "commit", "-m", "forge: initial workspace snapshot", logger=logger)


def _read_forge_result(result_json: Path, stdout: str) -> dict[str, Any] | None:
    """Read the structured result file, then fall back to the stdout sentinel."""
    try:
        result = json.loads(result_json.read_text())
        if isinstance(result, dict):
            return result
    except (OSError, json.JSONDecodeError):
        pass
    matches = re.findall(
        re.escape(FORGE_RESULT_SENTINEL)
        + r"(.*?)"
        + re.escape(FORGE_RESULT_SENTINEL),
        stdout,
        flags=re.DOTALL,
    )
    for payload in reversed(matches):
        try:
            result = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result
    return None


def run_forge_subprocess(
    cmd_parts: list[str],
    *,
    workspace: str,
    env: dict[str, str],
    timeout_seconds: int,
    logger: logging.Logger,
) -> tuple[subprocess.Popen, list[str], list[str], bool]:
    """Stream a KernelForge subprocess to the log and hard-kill it on timeout.

    Launched from the argv list with shell=False (no intermediate shell) and in a
    NEW SESSION so the KernelForge process and its Claude/GPU subprocesses all
    share one process group. On timeout we can then signal the ENTIRE group;
    terminating only the leader would leave those children alive, still holding
    the GPU / editing the workspace while Arena does final scoring.
    """
    process = subprocess.Popen(
        cmd_parts,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=workspace,
        env=env,
        bufsize=1,
        start_new_session=True,
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def read_stream(stream, sink, prefix, log_func):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                text = line.rstrip()
                if text:
                    sink.append(text)
                    log_func(f"{prefix} {text}")
        finally:
            stream.close()

    threads = [
        threading.Thread(
            target=read_stream,
            args=(process.stdout, stdout_lines, "[FORGE]", logger.info),
            daemon=True,
        ),
        threading.Thread(
            target=read_stream,
            args=(process.stderr, stderr_lines, "[FORGE STDERR]", logger.warning),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()

    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        logger.warning(
            f"Forge run timed out after {timeout_seconds}s; terminating process group"
        )
        _terminate_process_group(process, logger)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.error(
                "Forge process leader did not exit after process-group termination"
            )

    for thread in threads:
        thread.join(timeout=1)

    return process, stdout_lines, stderr_lines, timed_out


def forge_environment() -> dict[str, str]:
    """The environment every KernelForge subprocess inherits.

    All existing parent credentials are preserved; KernelForge independently
    decides whether optional external integrations are configured. The API key is
    dropped so the gateway auth path (ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN)
    is the only one available.
    """
    env = os.environ.copy()
    env["IS_SANDBOX"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.pop("ANTHROPIC_API_KEY", None)
    return env
