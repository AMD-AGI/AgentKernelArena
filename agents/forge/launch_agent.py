# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Forge agent — bridges AgentKernelArena to KernelForge's `forge-loop`.

KernelForge's autonomous optimization loop (baseline -> agent edit -> 5-stage
validate -> bench -> keep/revert) runs as a standalone, hard-killable subprocess
via `kernel-agents forge-loop`. This launcher adapts an Arena task workspace to
that loop's contract:

  1. Resolve the kernel file Arena copied into the workspace (task's
     ``source_file_path[0]``).
  2. Materialize a driver shim implementing the KernelForge driver contract
     (prints ``SNR: <db> dB`` for correctness and ``wall_ms: <ms>`` for bench).
  3. ``git init`` + initial commit the workspace (the loop uses git keep/revert).
  4. Generate a ``forge_program.md`` from the task prompt for agent guidance.
  5. Shell out to ``kernel-agents forge-loop`` (streaming output), which leaves
     the workspace at the best-kept kernel.

After this returns, Arena re-materializes its perf helpers and re-scores the
kernel with the task's own compile/correctness/performance commands.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from agents import register_agent

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
        return detected

    model = str(eval_config.get("target_gpu_model", "")).strip()
    arch = _resolve_gfx_arch(model)
    if arch:
        return arch

    raise ValueError(
        f"Cannot resolve a gfx arch for target_gpu_model={model!r}: it was not "
        "detected via rocminfo and is not defined under 'architecture' in "
        "src/prompts/cheatsheet/default_cheatsheet.yaml. Add the model there (with "
        "its gfx_arch) or run on the target GPU so rocminfo can report it."
    )


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
    """Terminate the forge-loop and ALL its descendants (kernel-agents, claude, GPU).

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

    logger.warning("Force killing forge loop process group (SIGKILL)")
    if not _signal_group(signal.SIGKILL):
        return

    if not _wait_for_process_group_exit(process, pgid, kill_timeout):
        logger.warning("Forge loop process group did not exit even after SIGKILL")


def _forge_max_hours(agent_config: dict[str, Any]) -> float:
    """Derive the forge-loop ``--max-hours`` budget from the run's timeout.

    ``timeout_seconds`` is the single time budget (the API/bootstrap patches it
    per-run, e.g. 115200 for a 32h run). ``--max-hours`` tracks it with a small
    margin so the loop self-stops (BUDGET EXHAUSTED) just before the hard
    process-wait kill instead of being killed mid-iteration. A fixed hours value
    would ignore the per-run timeout and cap long runs early (a 32h run would
    stop at ~8h). The default timeout (29700s) yields ~8h.
    """
    timeout_s = float(agent_config.get("timeout_seconds", 3600))
    return round(max(0.1, timeout_s / 3600.0 - 0.25), 3)


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


def _resolve_all_source_files(
    workspace: str, source_files: list, task_config: dict[str, Any],
    logger: logging.Logger,
) -> list[Path]:
    """Resolve EVERY source_file_path entry to an absolute workspace path.

    The first entry (anchor) must exist; extra entries that cannot be resolved
    are warned and skipped rather than failing the run (a task may list an
    optional/relocated file). Order-preserving, de-duplicated.
    """
    resolved: list[Path] = []
    for i, rel in enumerate(source_files):
        p = _resolve_one_source_file(workspace, rel, task_config)
        if p is not None:
            if p not in resolved:
                resolved.append(p)
        elif i == 0:
            raise RuntimeError(
                f"Anchor kernel file not found in workspace: {Path(workspace) / str(rel)}"
            )
        else:
            logger.warning("forge: source_file_path entry not found, skipping: %s", rel)
    return resolved


def _git_short_head(repo_dir: Path) -> str:
    """Return the short HEAD commit of a git repo dir, or "" (best-effort)."""
    if not (repo_dir / ".git").exists():
        return ""
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001 - best-effort
        return ""


def _repo_head_commit(workspace: str, task_config: dict[str, Any], logger: logging.Logger) -> str:
    """Return the short HEAD commit to record as the KB framework version.

    Order:
      1. the workspace repo subdir's own .git (repository tasks: a git clone) —
         read BEFORE ``_strip_nested_git`` removes it;
      2. the in-image source tree (``image_repo_path``) for image_kernel tasks,
         whose seeded workspace copy drops .git but whose in-image checkout still
         has one.
    Best-effort: returns "" when no commit can be read (KB then falls back to the
    installed package version / unknown).
    """
    candidates: list[Path] = []
    subdir = _repo_subdir_name(task_config)
    if subdir:
        candidates.append(Path(workspace) / subdir)
    image_repo_path = task_config.get("image_repo_path")
    if image_repo_path:
        candidates.append(Path(str(image_repo_path)))
    for repo_dir in candidates:
        commit = _git_short_head(repo_dir)
        if commit:
            logger.info(f"forge: recorded repo commit {commit} ({repo_dir.name}) for KB version")
            return commit
    return ""


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


# KernelForge fellows available to the single-fellow forge-loop path
# (see kernel_agents.fellows.base.build_single_fellow_prompt).
_VALID_FELLOW_BACKENDS = {"ck", "flydsl", "triton", "aiter", "hip", "hipblaslt"}


def _infer_backend(task_config: dict[str, Any]) -> str:
    """Infer the fellow backend (kernel language) from the task config.

    Two task families need different signals:

      * Repository / image_kernel tasks ship a whole source tree, not a
        "<src>2<dst>" pair, so their task_type carries no language. The kernel
        language is stated explicitly by ``repository_language`` — the
        authoritative signal. The source REPO (aiter / sglang) must NOT be used
        as the fellow: there is no kernel-building sglang fellow, and the aiter
        fellow only INTEGRATES prebuilt operators rather than editing kernel
        source, so an aiter-repo HIP kernel is optimized by the hip fellow.
      * Snippet tasks are "<source>2<target>" (triton2triton, cuda2hip,
        torch2hip, flydsl2flydsl, instruction2triton, ...); the optimized kernel
        is in the TARGET language, i.e. the part after the last '2'.

    A repo/image task that lacks a usable ``repository_language`` falls through
    to the generic heuristics (keyword scan, then triton) with a warning: a
    mislabeled fellow only degrades the run, it never hard-fails it.
    """
    task_type = str(task_config.get("task_type") or "").lower().strip()

    if task_type in ("image_kernel", "repository"):
        lang = str(task_config.get("repository_language") or "").lower().strip()
        if lang in _VALID_FELLOW_BACKENDS:
            return lang
        logging.getLogger(__name__).warning(
            "Task type %r has no usable 'repository_language' (got %r); falling "
            "back to generic backend inference. Set repository_language to one "
            "of %s to select the correct fellow.",
            task_type, lang, sorted(_VALID_FELLOW_BACKENDS),
        )

    target = task_type.rsplit("2", 1)[-1] if "2" in task_type else task_type
    if target in _VALID_FELLOW_BACKENDS:
        return target
    for backend in _VALID_FELLOW_BACKENDS:
        if backend in task_type:
            return backend
    return "triton"


def _resolve_fellow(task_config: dict[str, Any], agent_config: dict[str, Any]) -> str:
    """Pick the fellow: explicit agent_config override wins, else inferred."""
    override = agent_config.get("fellow")
    if override:
        return str(override)
    return f"{_infer_backend(task_config)}-fellow"


def _derive_workload_key(task_config_dir: str, arena_root: str) -> str:
    """Return the Arena task identity to pass as KernelForge's ``--workload-key``.

    Several Arena tasks can optimize the SAME kernel source under different
    benchmark regimes (e.g. ``aiter_pa_decode`` vs ``aiter_pa_ragged`` both edit
    ``pa_kernels.cuh``). They resolve to the same KernelForge kernel identity, so
    without a distinct workload key their KB experience would rank together and
    suppress / mis-seed each other. The task's path relative to ``tasks/`` (e.g.
    ``image_kernel/aiter_pa_decode``) is a stable, globally-unique identity for
    the workload; fall back to the task directory name if the relative path can't
    be computed.
    """
    task_dir = Path(task_config_dir).resolve().parent
    try:
        return str(task_dir.relative_to(Path(arena_root).resolve() / "tasks"))
    except ValueError:
        return task_dir.name


def _git(workspace: str, *args: str, logger: logging.Logger) -> None:
    """Run a git command in the workspace, tolerating non-zero exit."""
    result = subprocess.run(
        ["git", *args], cwd=workspace, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        logger.debug(f"git {' '.join(args)} -> {result.returncode}: {result.stderr.strip()}")


# Build artifacts / regenerated reports / forge scaffolding must NOT be tracked:
# if they are, a validation or benchmark run that regenerates them dirties the
# tree and makes the loop's `git revert` fail — leaking a reverted (often broken)
# edit into the final tree. Only source is tracked, matching the loop's own
# `git add -u` philosophy.
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
forge_program.md
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


def _write_program_md(task_config: dict[str, Any], target_funcs, gpu_arch: str, backend: str, dest: Path) -> None:
    """Generate a program.md for the forge agent from the Arena task prompt."""
    prompt_cfg = task_config.get("prompt") or {}
    instructions = prompt_cfg.get("instructions") or ""
    funcs = ", ".join(target_funcs) if target_funcs else "the target kernel"

    # Level-3 tasks (repository / image_kernel) give the agent a full source tree,
    # so the target kernel file is often only an ENTRY POINT: the code that governs
    # its performance may live in other files it includes/imports. Tell the agent
    # it may follow the implementation across the workspace and edit those files
    # too. Snippet tasks copy a single self-contained file, so the note is omitted
    # there to avoid pointing the agent at unrelated files.
    task_type = str(task_config.get("task_type") or "").strip().lower()
    scope_section = ""
    if task_type in ("repository", "image_kernel"):
        scope_section = f"""
## Kernel Scope (the target file may not be self-contained)
`{funcs}` lives in the target kernel file, but the code that governs its
performance may span OTHER files in this workspace (headers it includes, modules
it imports, dispatch/config layers, or JIT template sources). You are NOT limited
to the single target file:
1. Before editing, use Grep/Glob/Read to trace the real implementation across the
   repository (includes, imports, call sites, templates) so you edit where the
   performance-relevant code actually is.
2. You MAY edit any source file the kernel depends on when that is where the
   change belongs. The loop stages every tracked source edit, so a cross-file
   change is validated, benchmarked, and kept/reverted as one unit.
3. Do NOT edit the measurement files (the test driver / harness / perf helpers);
   the loop rejects edits to those.
"""

    dest.write_text(
        f"""# Program: optimize {funcs}

**GPU**: {gpu_arch}
**Backend**: {backend}

## Objective
Optimize the body of `{funcs}` for maximum performance on {gpu_arch} while
keeping numerical results correct (the loop gates on an SNR threshold).
{scope_section}
## Modification Rules
1. You MAY make multiple changes across several places in the kernel this
   iteration (not just one); group them under a clear hypothesis so the
   iteration's effect stays attributable.
2. Do NOT change the kernel's function signature or parameter list.
3. Do NOT remove imports or helper utilities in the file.
4. Do NOT edit task harness, test, scoring, or measurement files (`config.yaml`,
   `scripts/`, `test/`, `tests/`, `conftest.py`, `performance_utils_pytest.py`,
   `*_test.py`, `*_test.cpp`, `*_test.cu`, `*_test.hip`). The Arena runner
   hashes these files and rejects the score if they change.
5. Build, run, and verify your edit YOURSELF (compile, run the driver, profile)
   before finishing. The loop additionally runs a canonical correctness (SNR gate)
   and benchmark pass on your final kernel after you stop.

## Task instructions (from AgentKernelArena)
{instructions}
"""
    )


def _render_driver_shim(drivers_dir: str, workspace: str, task_config: str, arena_root: str) -> str:
    """Generate a forge_driver.py that bakes the task paths in (no env needed).

    The shim adds the shared adapter's dir to sys.path and delegates to
    ``arena_task_adapter.run``, passing workspace / task config / arena root as
    explicit arguments. KernelForge invokes this file as ``python forge_driver.py
    <args>`` (see kernel_agents.mcp_server.tools.{test,bench}), so a ``__main__``
    entry is all that is required.
    """
    return f"""\
#!/usr/bin/env python3
# Auto-generated by the AgentKernelArena forge launcher. Task paths are baked in
# below so the driver needs NO environment variables. It delegates to the shared
# arena_task_adapter, which reuses Arena's own correctness/performance eval.
import sys

sys.path.insert(0, {drivers_dir!r})
import arena_task_adapter as adapter

WORKSPACE = {workspace!r}
TASK_CONFIG = {task_config!r}
ARENA_ROOT = {arena_root!r}

if __name__ == "__main__":
    raise SystemExit(adapter.run(WORKSPACE, TASK_CONFIG, ARENA_ROOT, sys.argv[1:]))
"""


@register_agent("forge")
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str) -> str:
    """Run one KernelForge forge-loop over the Arena task workspace.

    Args:
        eval_config: Arena run config (includes target_gpu_model).
        task_config_dir: Path to the task's config.yaml (for source/target fields).
        workspace: Isolated task workspace Arena prepared; the kernel lives here.

    Returns:
        Combined streamed output of the forge-loop subprocess.
    """
    logger = logging.getLogger(__name__)

    forge_bin = shutil.which("kernel-agents")
    if not forge_bin:
        raise RuntimeError(
            "Command 'kernel-agents' not found. Install KernelForge "
            "(pip install -e KernelForge) so the forge-loop CLI is on PATH."
        )

    # Agent config
    config_path = Path(__file__).with_name("agent_config.yaml")
    with config_path.open("r") as f:
        agent_config = yaml.safe_load(f) or {}

    # Task config: locate the kernel file + target function(s).
    with open(task_config_dir, "r") as f:
        task_config = yaml.safe_load(f) or {}
    source_files = task_config.get("source_file_path") or []
    if not source_files:
        raise RuntimeError(f"Task config has no source_file_path: {task_config_dir}")
    kernel_file = _resolve_kernel_file(workspace, source_files, task_config)
    # Resolve the FULL editable surface (repository tasks list several files; the
    # entry file is often only a dispatcher). The anchor is all_source_files[0].
    all_source_files = _resolve_all_source_files(workspace, source_files, task_config, logger)
    target_funcs = task_config.get("target_kernel_functions") or []
    task_type = str(task_config.get("task_type") or "").strip()

    gpu_arch = _resolve_gpu_arch(eval_config)
    fellow = _resolve_fellow(task_config, agent_config)
    backend = fellow.split("-")[0]

    # Materialize the driver: prefer a task-shipped scripts/forge_driver.py
    # (used verbatim); otherwise generate a shim that delegates to the shared
    # arena_task_adapter with the task paths baked in — so the driver needs NO
    # environment variables.
    arena_root = str(Path(__file__).resolve().parents[2])
    driver_dest = Path(workspace) / "forge_driver.py"
    task_driver = Path(workspace) / "scripts" / "forge_driver.py"
    if task_driver.exists():
        shutil.copy2(task_driver, driver_dest)
        logger.info(f"Forge: using task-provided driver {task_driver}")
    else:
        drivers_dir = str(Path(__file__).parent / "drivers")
        workspace_task_config = Path(workspace) / "config.yaml"
        task_config_path = str(
            workspace_task_config if workspace_task_config.exists() else task_config_dir
        )
        driver_dest.write_text(_render_driver_shim(
            drivers_dir=drivers_dir,
            workspace=str(workspace),
            task_config=task_config_path,
            arena_root=arena_root,
        ))
        logger.info(f"Forge: generated driver shim -> {driver_dest} (task paths baked in)")

    # program.md for agent guidance.
    program_md = Path(workspace) / "forge_program.md"
    _write_program_md(task_config, target_funcs, gpu_arch, backend, program_md)

    # Repository / image_kernel tasks bring a cloned repo (with its own nested
    # .git) into the workspace. Capture the repo commit (for the KB version) FIRST,
    # then strip .git BEFORE git init so the outer workspace git tracks the repo's
    # files (otherwise keep/revert is a no-op on the real kernel). No-op / skipped
    # for single-file snippet tasks.
    repo_commit = ""
    if task_type.lower() in ("repository", "image_kernel"):
        repo_commit = _repo_head_commit(workspace, task_config, logger)
        _strip_nested_git(workspace, logger)

    # The loop needs a git repo for the keep/revert pattern.
    _init_git_workspace(workspace, logger)

    experiments_dir = Path(workspace) / "forge_experiments"
    result_json = experiments_dir / "forge_result.json"

    model = str(agent_config.get("model", "claude-opus-4-8"))
    permission_mode = str(agent_config.get("permission_mode", "acceptEdits"))

    # Build the forge-loop command. All task-specific config is passed as flags
    # (model / permission-mode / gpu-target / fellow / paths) so nothing relies
    # on forge-specific environment variables.
    cmd_parts = [
        forge_bin, "forge-loop",
        "--kernel", str(kernel_file),
        "--driver", str(driver_dest),
        "--workspace", str(workspace),
        "--experiments-dir", str(experiments_dir),
        "--result-json", str(result_json),
        "--snr-threshold", str(agent_config.get("snr_threshold", 30.0)),
        "--max-iters", str(agent_config.get("max_iters", 2)),
        "--max-hours", str(_forge_max_hours(agent_config)),
        "--gpu-target", gpu_arch,
        "--fellow", fellow,
        "--git-branch", "forge-optimize",
        "--program-md-file", str(program_md),
        "--model", model,
        "--permission-mode", permission_mode,
        # Task-shape awareness: lets forge-loop switch on single-file vs repo/
        # image_kernel and enable multi-file handling only where needed.
        "--task-type", task_type,
    ]
    # Full editable source set + target functions (repository tasks span several
    # files; the entry file is often only a dispatcher). forge-loop uses these
    # for PMC name hints, the in-session edit budget, and the agent prompt. The
    # anchor stays --kernel; for single-file tasks --source-files is just it.
    if all_source_files:
        cmd_parts += ["--source-files", ",".join(str(p) for p in all_source_files)]
    if target_funcs:
        cmd_parts += ["--target-functions", ",".join(str(f) for f in target_funcs)]
    # Repo commit as the KB framework version (used only when the installed
    # package version is unknown — e.g. a source-checkout framework).
    if repo_commit:
        cmd_parts += ["--framework-version", repo_commit]
    # Workload identity: distinguishes tasks that share one kernel source but
    # exercise different shape regimes, so their KB experience does not collide.
    cmd_parts += ["--workload-key", _derive_workload_key(task_config_dir, arena_root)]
    # shapes_json is only meaningful for per-kernel drivers that parse --shape.
    # The generic arena_task_adapter ignores --shape (the task's pytest owns its
    # shapes), so we omit it unless a task explicitly configures one — the
    # forge-loop CLI defaults to "{}" (one default-shape sweep).
    shapes_json = agent_config.get("shapes_json")
    if shapes_json:
        cmd_parts += ["--shapes-json", str(shapes_json)]
    # Human-readable rendering for the log only; the process is launched from the
    # argv list (cmd_parts) with shell=False, so no shell parsing is involved.
    cmd = " ".join(shlex.quote(p) for p in cmd_parts)

    # Environment: inherit the gateway auth (ANTHROPIC_BASE_URL/ANTHROPIC_AUTH_TOKEN)
    # from the parent. Only two non-inherited vars remain, both genuinely
    # environmental (not forge config): IS_SANDBOX (required by the claude CLI's
    # bypassPermissions under root) and PYTHONUNBUFFERED (streaming). Everything
    # else is a CLI flag or baked into the generated driver.
    env = os.environ.copy()
    env["IS_SANDBOX"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    # The gateway uses bearer auth; ensure x-api-key auth isn't picked instead.
    env.pop("ANTHROPIC_API_KEY", None)

    logger.info("Forge Preflight")
    logger.info(f"  forge bin:   {forge_bin}")
    logger.info(f"  kernel:      {kernel_file}")
    logger.info(f"  driver:      {driver_dest}")
    logger.info(f"  gpu target:  {gpu_arch}")
    logger.info(f"  model:       {model}")
    logger.info(f"  fellow:      {fellow} (inferred from task_type={task_config.get('task_type')!r})")
    logger.info(f"  budget:      {agent_config.get('max_iters')} iters / {_forge_max_hours(agent_config)}h")
    logger.info(f"  gateway:     {env.get('ANTHROPIC_BASE_URL', '<unset>')}")
    logger.info(f"Running command: {cmd}")
    logger.info("=" * 80)
    logger.info("Forge Output (streaming):")
    logger.info("=" * 80)

    timeout_seconds = int(agent_config.get("timeout_seconds", 3600))

    # Launch from the argv list with shell=False (no intermediate shell) and in a
    # NEW SESSION so the forge-loop, the kernel-agents process, and its Claude/GPU
    # subprocesses all share one process group. On timeout we can then signal the
    # ENTIRE group; terminating only the leader would leave those children alive,
    # still holding the GPU / editing the workspace while Arena does git checkout
    # and final scoring.
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
        threading.Thread(target=read_stream, args=(process.stdout, stdout_lines, "[FORGE]", logger.info), daemon=True),
        threading.Thread(target=read_stream, args=(process.stderr, stderr_lines, "[FORGE STDERR]", logger.warning), daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        logger.warning(f"Forge loop timed out after {timeout_seconds}s; terminating process group")
        _terminate_process_group(process, logger)

    for t in threads:
        t.join(timeout=1)

    logger.info("=" * 80)
    logger.info(f"Forge loop completed with exit code: {process.returncode}")
    logger.info("=" * 80)

    # Restore the workspace working tree to the loop's final (best-kept) state.
    # The loop runs on the 'forge-optimize' branch; ensure no partial/uncommitted
    # revert leaves the tree dirty before Arena re-scores.
    _git(workspace, "checkout", "--", ".", logger=logger)

    output = "\n".join(stdout_lines)
    if stderr_lines:
        output += "\n=== STDERR ===\n" + "\n".join(stderr_lines)
    return output
