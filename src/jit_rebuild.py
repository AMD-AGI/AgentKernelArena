# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Force JIT-compiled kernels to rebuild from the CURRENT source.

aiter ships a prebuilt in-tree ``.so`` in the image and, by default, loads it
WITHOUT checking whether the kernel source changed. An agent's edit to a HIP
``.cu``/``.cuh`` would then be silently ignored — the benchmark keeps running the
ORIGINAL kernel, correctness validates the original, and the speedup never moves.

This is a centralized safety net so individual task authors never have to
remember a per-task rebuild tweak. It is applied before every compile /
correctness / performance step (see ``src/evaluator.py`` + ``src/performance.py``),
so it holds identically for the baseline, any agent (cursor / claude / forge),
and the final re-score.

Two mechanisms, because aiter has two build paths:

1. ``AITER_REBUILD=1`` — makes aiter's ``@compile_ops`` PYBIND ops rebuild from
   source instead of loading the prebuilt in-tree ``.so``. Sufficient for most
   HIP ops (e.g. paged-attention ``compile_template_op``).

2. Stale-``.so`` deletion — CK ops whose module is produced by a ``gen_func``
   codegen path (e.g. ``mha_batch_prefill``) BYPASS the ``AITER_REBUILD`` trigger
   and keep serving the cached specialized ``.so`` even when the ``.cu`` changed.
   ``AITER_REBUILD=1`` alone does NOT rebuild them (empirically verified). So we
   additionally delete the task op's compiled ``.so`` from the workspace jit dir
   — but ONLY when the source is newer than the built ``.so`` (i.e. an edit
   happened), to avoid a pointless multi-minute CK rebuild on every eval step.

Scope: aiter HIP (C/C++) kernels only. Triton / Python kernels re-key their JIT
on the source and recompile on edit, so they are left untouched — and forcing an
aiter rebuild for a Triton task would trigger a slow, pointless C++ recompile.

Note: sglang (tvm-ffi) tasks are intentionally NOT handled here yet; that path
needs a kernel-name-agnostic cache-clear and is deferred.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# C/C++ (HIP/CUDA) kernel sources — these are the ones aiter ships prebuilt and
# can shadow with a stale .so. Triton/Python (.py) are not listed: their JIT
# keys by source and recompiles on edit.
_CPP_EXTS = (".cu", ".cuh", ".hip", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp")


def _task_paths(task_config: dict[str, Any]) -> list[str]:
    """Kernel/source + repo path strings used to detect the framework + language."""
    paths: list[str] = []
    sfp = task_config.get("source_file_path") or []
    if isinstance(sfp, str):
        sfp = [sfp]
    paths.extend(str(s) for s in sfp if s)
    for key in ("image_repo_path", "repo_url", "repo_subdir"):
        val = task_config.get(key)
        if val:
            paths.append(str(val))
    return paths


def force_jit_rebuild(
    task_config: dict[str, Any],
    logger: logging.Logger | None = None,
    workspace: str | os.PathLike | None = None,
) -> dict[str, str]:
    """Best-effort: make the task's JIT kernel recompile from the current source.

    Returns the environment overrides (e.g. ``{"AITER_REBUILD": "1"}``) that the
    CALLER must apply ONLY to the build subprocess it spawns for this task step
    (pass them as ``run_command(..., extra_env=...)``). They are deliberately NOT
    written to ``os.environ``: a worker process runs many tasks sequentially, so
    leaking ``AITER_REBUILD=1`` into the parent env would force pointless C++
    rebuilds for a later (e.g. Triton) aiter task. Returns ``{}`` for
    non-aiter / non-C/C++ tasks.

    Never raises — a rebuild-hint failure must not break evaluation. ``workspace``
    is required for the stale-``.so`` deletion (gen_func CK ops); without it only
    the ``AITER_REBUILD`` env override is returned.
    """
    lg = logger or log
    try:
        return apply_jit_rebuild(
            _task_paths(task_config),
            workspace=workspace,
            task_config=task_config,
            logger=lg,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort
        lg.debug("force_jit_rebuild skipped: %r", exc)
        return {}


def apply_jit_rebuild(
    paths,
    workspace: str | os.PathLike | None = None,
    task_config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, str]:
    """Framework-keyed rebuild forcing from a list of path strings.

    aiter: return ``{"AITER_REBUILD": "1"}`` so aiter rebuilds the kernel from
    source instead of loading the prebuilt in-tree ``.so``. The caller applies it
    to the build subprocess only (``run_command(extra_env=...)``); it is NOT
    written to ``os.environ`` so it cannot leak into a later task run by the same
    worker. Additionally clear the task op's stale compiled ``.so`` (gen_func CK
    ops that ignore ``AITER_REBUILD``) when a workspace + task_config are supplied.
    Returns ``{}`` when no rebuild forcing applies.
    """
    lg = logger or log
    strs = [str(p).lower() for p in paths if p]
    if not strs:
        return {}
    # Only C/C++ HIP kernels have the prebuilt-.so shadowing problem; forcing a
    # rebuild for a Triton (.py) task would just recompile aiter's C++ for nothing.
    if not any(s.endswith(_CPP_EXTS) for s in strs):
        return {}
    joined = " ".join(strs)

    if "aiter" not in joined:
        return {}

    if workspace is not None and task_config is not None:
        _clear_stale_aiter_op_so(Path(workspace), task_config, lg)
    # Explicit "1" (not setdefault): an inherited AITER_REBUILD=0 must still be
    # overridden when a rebuild is required. Scoped to the caller's subprocess.
    return {"AITER_REBUILD": "1"}


def _op_name_keys(task_config: dict[str, Any]) -> set[str]:
    """Op-name prefixes used to match aiter's specialized codegen ``.so`` files.

    aiter names a gen_func module's ``.so`` after the op (e.g. the
    ``mha_batch_prefill`` op -> ``mha_batch_prefill_bf16_...nsink.so``), so the
    task's ``target_kernel_functions`` give reliable prefixes. Short/generic
    names (< 5 chars) are dropped to avoid over-broad matches.
    """
    keys: set[str] = set()
    for fn in task_config.get("target_kernel_functions") or []:
        f = str(fn).strip().lstrip("_")
        if len(f) >= 5:
            keys.add(f)
    return keys


def _resolve_source_files(workspace: Path, task_config: dict[str, Any]) -> list[Path]:
    """Resolve the task's C/C++ source files to absolute paths in the workspace."""
    repo_subdir = task_config.get("repo_subdir")
    if not repo_subdir:
        for key in ("image_repo_path", "repo_url"):
            val = task_config.get(key)
            if val:
                name = str(val).rstrip("/")
                if name.endswith(".git"):
                    name = name[:-4]
                repo_subdir = name.rsplit("/", 1)[-1]
                break

    resolved: list[Path] = []
    sfp = task_config.get("source_file_path") or []
    if isinstance(sfp, str):
        sfp = [sfp]
    for s in sfp:
        rel = str(s)
        if not rel.lower().endswith(_CPP_EXTS):
            continue
        candidates: list[Path] = []
        if repo_subdir:
            candidates.append(workspace / repo_subdir / rel)
        candidates.append(workspace / rel)
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            matches = [m for m in workspace.rglob(Path(rel).name) if ".git" not in m.parts]
            found = matches[0] if len(matches) == 1 else None
        if found is not None:
            resolved.append(found)
    return resolved


def _clear_stale_aiter_op_so(
    workspace: Path, task_config: dict[str, Any], logger: logging.Logger
) -> None:
    """Delete the task op's compiled ``.so`` when its source was edited.

    Targets aiter's gen_func codegen variants (e.g. ``mha_batch_prefill_*.so``)
    that ignore ``AITER_REBUILD``. Never touches aiter's ``module_*`` base
    modules. Only deletes a ``.so`` older than the newest task source file, so an
    unedited baseline keeps its prebuilt ``.so`` (no pointless CK rebuild).
    """
    keys = _op_name_keys(task_config)
    if not keys:
        return

    max_src_mtime: float | None = None
    for src in _resolve_source_files(workspace, task_config):
        try:
            mt = src.stat().st_mtime
            max_src_mtime = mt if max_src_mtime is None else max(max_src_mtime, mt)
        except OSError:
            continue

    removed: list[str] = []
    for jit_dir in workspace.glob("**/aiter/jit"):
        if not jit_dir.is_dir():
            continue
        for so in jit_dir.rglob("*.so"):
            name = so.name
            if name.startswith("module_"):
                continue  # aiter's prebuilt base modules — not gen_func variants
            if not any(name.startswith(k) for k in keys):
                continue
            # Keep the .so unless the source is strictly newer (an edit happened).
            if max_src_mtime is not None:
                try:
                    if so.stat().st_mtime >= max_src_mtime:
                        continue
                except OSError:
                    pass
            try:
                so.unlink()
                removed.append(name)
            except OSError:
                pass

    if removed:
        logger.info(
            "force_jit_rebuild: source newer than build — cleared %d stale aiter "
            ".so to force rebuild: %s",
            len(removed),
            sorted(set(removed)),
        )
