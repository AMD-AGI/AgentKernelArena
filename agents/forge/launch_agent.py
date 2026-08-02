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

import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from agents import register_agent

_FORGE_RESULT_SENTINEL = "__FORGE_RESULT__"
_KB_STATUS_FILE = "arena_forge_status.json"
_PRODUCER_REQUIRED_FLAGS = {
    "--framework",
    "--operator-name",
    "--result-json",
    "--shapes-json",
    "--source-files",
    "--target-functions",
    "--workload-key",
}
_SOURCE_OWNERS = {"aiter", "sglang", "standalone", "vllm"}
_KERNEL_KIND_BACKENDS = {
    "aiter": "aiter",
    "ck": "ck",
    "flydsl": "flydsl",
    "hip": "hip",
    "hipblaslt": "hipblaslt",
    "torch": "triton",
    "triton": "triton",
}


def _kb_mode(agent_config: dict[str, Any]) -> str:
    """Return the declarative KB mode; ordinary evaluation is the default."""
    kb_config = agent_config.get("knowledge_base") or {}
    if not isinstance(kb_config, dict):
        raise ValueError("agent knowledge_base configuration must be a mapping")
    mode = str(kb_config.get("mode", "compatibility")).strip().lower()
    if mode not in {"compatibility", "producer"}:
        raise ValueError(
            "agent knowledge_base.mode must be 'compatibility' or 'producer'"
        )
    return mode


def _apply_run_kb_config(
    agent_config: dict[str, Any],
    eval_config: dict[str, Any],
) -> dict[str, Any]:
    """Overlay per-run KB intent without mutating the shared agent config."""
    run_agent = eval_config.get("agent") or {}
    if not isinstance(run_agent, dict):
        raise ValueError("run agent configuration must be a mapping")
    run_kb = run_agent.get("knowledge_base")
    if run_kb is None:
        return agent_config
    if not isinstance(run_kb, dict):
        raise ValueError("run agent.knowledge_base configuration must be a mapping")
    merged = dict(agent_config)
    base_kb = agent_config.get("knowledge_base") or {}
    if not isinstance(base_kb, dict):
        raise ValueError("agent knowledge_base configuration must be a mapping")
    merged["knowledge_base"] = {**base_kb, **run_kb}
    return merged


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
    kb_config = agent_config.get("knowledge_base") or {}
    margin_s = float(kb_config.get("finalization_margin_seconds", 900))
    if margin_s < 0:
        raise ValueError("knowledge_base.finalization_margin_seconds cannot be negative")
    if (
        _kb_mode(agent_config) == "producer"
        and timeout_s - margin_s < 3600
    ):
        raise ValueError(
            "Producer timeout_seconds must leave at least one hour for forge-loop "
            f"after the KB finalization margin ({timeout_s:g}s total, "
            f"{margin_s:g}s margin)"
        )
    # forge-loop enforces a one-hour minimum. Producer mode rejects an impossible
    # budget above; compatibility smoke runs retain the historical outer timeout.
    loop_seconds = max(3600.0, timeout_s - margin_s)
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


# KernelForge fellows available to the single-fellow forge-loop path
# (see kernel_agents.fellows.base.build_single_fellow_prompt).
_VALID_FELLOW_BACKENDS = {"ck", "flydsl", "triton", "aiter", "hip", "hipblaslt"}


def _kernel_kind_backend(kernel_kind: str) -> str:
    """Map an explicit implementation kind to its KernelForge fellow backend."""
    kind = str(kernel_kind or "").strip().lower().replace("-", "_")
    if (
        kind in {"aiter_ck", "ck"}
        or kind.startswith("ck_")
        or kind.endswith("_ck")
    ):
        return "ck"
    if "flydsl" in kind:
        return "flydsl"
    if "triton" in kind:
        return "triton"
    return _KERNEL_KIND_BACKENDS.get(kind, "")


def _infer_backend(task_config: dict[str, Any]) -> str:
    """Infer the fellow backend (kernel language) from the task config.

    Two task families need different signals:

      * Repository / image_kernel tasks ship a whole source tree, not a
        "<src>2<dst>" pair, so their task_type carries no language. A curated
        ``kernel_kind`` selects CK/FlyDSL/Triton semantics when present; otherwise
        ``repository_language`` describes the editable source language. The
        source REPO (aiter / sglang) must NOT be used as the fellow.
      * Snippet tasks are "<source>2<target>" (triton2triton, cuda2hip,
        torch2hip, flydsl2flydsl, instruction2triton, ...); the optimized kernel
        is in the TARGET language, i.e. the part after the last '2'.

    A repo/image task that lacks a usable ``repository_language`` falls through
    to the generic heuristics (keyword scan, then triton) with a warning: a
    mislabeled fellow only degrades the run, it never hard-fails it.
    """
    task_type = str(task_config.get("task_type") or "").lower().strip()

    if task_type in ("image_kernel", "repository"):
        kb_config = task_config.get("knowledge_base") or {}
        if not isinstance(kb_config, dict):
            kb_config = {}
        kernel_kind = str(
            kb_config.get("kernel_kind")
            or task_config.get("kernel_kind")
            or ""
        ).lower().strip().replace("-", "_")
        kind_backend = _kernel_kind_backend(kernel_kind)
        if kind_backend:
            return kind_backend

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


def _task_kb_config(task_config: dict[str, Any]) -> dict[str, Any]:
    """Return the task's canonical KB declaration."""
    value = task_config.get("knowledge_base") or {}
    if not isinstance(value, dict):
        raise ValueError("task knowledge_base configuration must be a mapping")
    return value


def _resolve_kernel_kind(
    task_config: dict[str, Any],
    backend: str,
    *,
    producer: bool,
) -> str:
    """Resolve the implementation kind without inventing producer metadata."""
    kb_config = _task_kb_config(task_config)
    explicit = kb_config.get("kernel_kind", task_config.get("kernel_kind"))
    if explicit:
        return str(explicit).strip().lower()
    if producer:
        raise ValueError(
            "KB producer task requires knowledge_base.kernel_kind; migrate the "
            "task instead of inferring implementation identity"
        )
    return ""


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


def _framework_from_path(path: str) -> str:
    """First (shallowest == owning package) known framework component in a path."""
    for comp in Path(path).parts:
        canon = _FRAMEWORK_ALIASES.get(comp.lower())
        if canon:
            return canon
    return ""


def _resolve_framework(
    task_config: dict[str, Any],
    kernel_file: str,
    source_files: list[Path] | None = None,
) -> str:
    """Framework identity for the KB slug: the package the kernel source lives in.

    Producer (this arena run) and consumer (a Hyperloom forge-loop) must resolve
    the same framework string for the same operator, or the kernel-page slug
    diverges and the solution is missed. Kernels live in different packages:
    some directly in ``vllm``/``sglang`` (e.g.
    ``.../vllm/model_executor/layers/fused_moe``), others in ``aiter`` /
    ``aiter_meta``. ``image_repo_path`` records the ORIGINAL package location but
    can point at a DEEP subdir, so we scan its full path (not its basename) for
    the owning framework, then fall back to the resolved kernel path.

    framework is a SOFT slug input: returns "" when unsure so the caller omits
    ``--framework`` and forge-loop infers it from the path (never raises, never
    guesses a wrong value). The serving stack a kernel is used FROM is NOT the
    framework; the framework is the package whose source is edited.
    """
    kb_config = _task_kb_config(task_config)
    explicit = str(
        kb_config.get("source_owner")
        or task_config.get("source_owner_framework")
        or ""
    ).strip()
    if explicit:
        return _FRAMEWORK_ALIASES.get(explicit.lower(), explicit.lower())

    candidates = [
        str(task_config.get("image_repo_path") or ""),
        *(str(path) for path in (source_files or [])),
        str(kernel_file or ""),
    ]
    source_origin = task_config.get("source_origin") or {}
    if isinstance(source_origin, dict):
        candidates.extend(
            [str(source_origin.get("repo") or ""), str(source_origin.get("path") or "")]
        )
    for base in candidates:
        framework = _framework_from_path(base)
        if framework:
            return framework
    return ""


def _logical_operator(task_config: dict[str, Any], *, producer: bool) -> str:
    """Return only explicitly declared logical identity."""
    kb_config = _task_kb_config(task_config)
    value = str(
        kb_config.get("logical_operator")
        or task_config.get("logical_operator")
        or ""
    ).strip()
    if producer and not value:
        raise ValueError(
            "KB producer task requires knowledge_base.logical_operator; task "
            "directory names and target symbols are not logical operator identity"
        )
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


def _json_scalar(value: Any, path: str) -> None:
    """Validate a workload value is stable JSON data."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return
    raise ValueError(f"{path} values must be JSON scalars, got {type(value).__name__}")


def _validate_shape(shape: Any, path: str) -> dict[str, Any]:
    if not isinstance(shape, dict) or not shape:
        raise ValueError(f"{path} must be a non-empty mapping")
    normalized: dict[str, Any] = {}
    for raw_key, value in shape.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError(f"{path} contains an empty dimension name")
        _json_scalar(value, f"{path}.{key}")
        normalized[key] = value
    return normalized


def _selector_schema(
    workload: dict[str, Any],
    parsed_cases: list[tuple[str, dict[str, Any]]],
) -> dict[str, str]:
    raw_schema = workload.get("selector_schema")
    if raw_schema is None:
        return {}
    if not isinstance(raw_schema, dict):
        raise ValueError("knowledge_base.workload.selector_schema must be a mapping")
    name = str(raw_schema.get("name") or "").strip()
    if name != "hyperloom-v1":
        raise ValueError("selector_schema.name must be 'hyperloom-v1'")
    fields = raw_schema.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError("selector_schema.fields must be a non-empty mapping")

    normalized: dict[str, str] = {}
    for raw_param, raw_selector in fields.items():
        param = str(raw_param).strip()
        selector = str(raw_selector).strip()
        if not param or not re.fullmatch(r"[A-Z][A-Z0-9]*", selector):
            raise ValueError(
                "selector_schema.fields must map parameter names to unambiguous "
                "uppercase selector keys without underscores"
            )
        if selector == "CASEID" or selector in normalized.values():
            raise ValueError(f"Duplicate or reserved selector key: {selector}")
        normalized[param] = selector

    case_param_keys = {
        key
        for _, params in parsed_cases
        for key in params
    }
    extra = sorted(set(normalized) - case_param_keys)
    if extra:
        raise ValueError(
            "selector_schema.fields references parameters absent from session cases: "
            f"{extra}"
        )
    missing_by_case = {
        case_id: sorted(set(normalized) - set(params))
        for case_id, params in parsed_cases
        if set(normalized) - set(params)
    }
    if missing_by_case:
        raise ValueError(
            "selector_schema.fields must be present in every session case: "
            f"{missing_by_case}"
        )
    return normalized


def _load_session_workload(
    workload: dict[str, Any],
    task_config_dir: str,
) -> dict[str, Any]:
    source_name = str(workload.get("source") or "").strip()
    source_path = Path(source_name)
    if not source_path.is_absolute():
        source_path = Path(task_config_dir).resolve().parent / source_path
    try:
        document = json.loads(source_path.read_text())
    except OSError as error:
        raise ValueError(f"Cannot read KB workload source {source_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid KB workload JSON {source_path}: {error}") from error
    cases = document.get("cases") if isinstance(document, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"KB workload source {source_path} has no non-empty cases list")

    parsed_cases: list[tuple[str, dict[str, Any]]] = []
    seen_case_ids: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"KB workload case {index} must be a mapping")
        case_id = str(case.get("id") or "").strip()
        if not case_id:
            raise ValueError(f"KB workload case {index} has no id")
        if case_id in seen_case_ids:
            raise ValueError(f"KB workload source contains duplicate case id {case_id!r}")
        seen_case_ids.add(case_id)
        parsed_cases.append(
            (case_id, _validate_shape(case.get("params"), f"workload case {case_id}.params"))
        )

    primary_case = str(workload.get("primary_case") or "").strip()
    if not primary_case:
        if len(parsed_cases) != 1:
            raise ValueError(
                "knowledge_base.workload.primary_case is required when the source "
                "contains multiple cases"
            )
        primary_case = parsed_cases[0][0]
    matches = [shape for case_id, shape in parsed_cases if case_id == primary_case]
    if len(matches) != 1:
        raise ValueError(
            f"KB workload primary_case {primary_case!r} does not select exactly one case"
        )
    field_mapping = _selector_schema(workload, parsed_cases)
    if not field_mapping:
        primary = matches[0]
        validation = [shape for _, shape in parsed_cases]
        return {"primary": primary, "validation": validation}

    selectors = {
        case_id: {
            "CASE_ID": case_id,
            **{
                selector: params[param]
                for param, selector in field_mapping.items()
            },
        }
        for case_id, params in parsed_cases
    }
    primary = selectors[primary_case]
    return {
        "primary": primary,
        "minimal": primary,
        "validation": list(selectors.values()),
    }


def _resolve_shapes(
    task_config: dict[str, Any],
    task_config_dir: str,
    agent_config: dict[str, Any],
    *,
    producer: bool,
) -> dict[str, Any] | None:
    """Resolve mode-appropriate shapes without leaking producer metadata."""
    if not producer:
        # Compatibility mode preserves only the historical agent-level override.
        # Task knowledge_base.workload is producer metadata and must not change
        # ordinary Forge benchmark selection.
        legacy = agent_config.get("shapes_json")
        if not legacy:
            return None
        try:
            parsed = json.loads(str(legacy))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid agent shapes_json: {error}") from error
        if not isinstance(parsed, dict):
            raise ValueError("agent shapes_json must decode to an object")
        return parsed

    kb_config = _task_kb_config(task_config)
    workload = kb_config.get("workload")
    if workload is None:
        raise ValueError(
            "KB producer task requires knowledge_base.workload with either "
            "'shapes' or a session-case 'source'"
        )
    if not isinstance(workload, dict):
        raise ValueError("knowledge_base.workload must be a mapping")
    if workload.get("source"):
        return _load_session_workload(workload, task_config_dir)

    raw_shapes = workload.get("shapes", workload)
    if not isinstance(raw_shapes, dict):
        raise ValueError("knowledge_base.workload.shapes must be a mapping")
    primary = _validate_shape(raw_shapes.get("primary"), "workload.shapes.primary")
    normalized: dict[str, Any] = {"primary": primary}
    if "minimal" in raw_shapes:
        normalized["minimal"] = _validate_shape(
            raw_shapes["minimal"], "workload.shapes.minimal"
        )
    if "validation" in raw_shapes:
        validation = raw_shapes["validation"]
        if not isinstance(validation, list) or not validation:
            raise ValueError("workload.shapes.validation must be a non-empty list")
        normalized["validation"] = [
            _validate_shape(shape, f"workload.shapes.validation[{index}]")
            for index, shape in enumerate(validation)
        ]
    return normalized


def _canonical_workload_key(shapes: dict[str, Any]) -> str:
    """Return an unambiguous deterministic key derived from the primary selector."""
    primary = _validate_shape(shapes.get("primary"), "shapes.primary")
    compute_selector = {
        key: primary[key]
        for key in sorted(primary)
        if key.upper() != "CASE_ID"
    }
    if not compute_selector:
        raise ValueError("KB workload primary shape has no compute dimensions")
    payload = json.dumps(
        compute_selector,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return f"shape-v2-{hashlib.sha256(payload).hexdigest()}"


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


def _write_program_md(
    task_config: dict[str, Any],
    target_funcs,
    gpu_arch: str,
    backend: str,
    editable_sources: list[Path],
    dest: Path,
) -> None:
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
        source_allowlist = "\n".join(
            f"- `{path}`" for path in editable_sources
        )
        scope_section = f"""
## Editable Source Scope
`{funcs}` lives in the target kernel file, but the code that governs its
performance may span OTHER files in this workspace (headers it includes, modules
it imports, dispatch/config layers, or JIT template sources).

The complete editable source allowlist is:
{source_allowlist}

You may inspect any dependency, but edit ONLY files in this allowlist. If an
additional dependency must be editable, stop and report that the task's
`editable_sources` declaration is incomplete. Do not edit measurement files.
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


def _check_producer_cli_capabilities(forge_bin: str) -> None:
    """Verify the executable on PATH exposes all required producer flags."""
    try:
        completed = subprocess.run(
            [forge_bin, "forge-loop", "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Cannot inspect kernel-agents forge-loop capabilities: {error}") from error
    help_text = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0:
        raise RuntimeError(
            "kernel-agents forge-loop --help failed while checking producer "
            f"capabilities (exit {completed.returncode}): {help_text.strip()}"
        )
    advertised = set(re.findall(r"(?<!\w)--[a-z][a-z0-9-]*", help_text))
    missing = sorted(_PRODUCER_REQUIRED_FLAGS - advertised)
    if missing:
        raise RuntimeError(
            "Installed kernel-agents does not support the Forge KB producer "
            f"contract; missing CLI flags: {', '.join(missing)}"
        )


def _producer_preflight(
    *,
    forge_bin: str,
    env: dict[str, str],
    logical_operator: str,
    kernel_kind: str,
    framework: str,
    shapes: dict[str, Any] | None,
    fellow: str,
    backend: str,
) -> None:
    """Fail before a long run when publication cannot satisfy its contract."""
    missing_credentials = [
        name for name in ("GBRAIN_BASE_URL", "GBRAIN_TOKEN") if not env.get(name)
    ]
    if missing_credentials:
        raise RuntimeError(
            "KB producer mode requires inherited credentials: "
            + ", ".join(missing_credentials)
        )
    if framework and framework not in _SOURCE_OWNERS:
        raise RuntimeError(
            f"Unknown KB source_owner {framework!r}; expected one of "
            f"{sorted(_SOURCE_OWNERS)}"
        )
    expected_backend = _kernel_kind_backend(kernel_kind)
    if kernel_kind and not expected_backend:
        raise RuntimeError(f"Unknown KB kernel_kind {kernel_kind!r}")
    missing_metadata = [
        name
        for name, value in (
            ("logical_operator", logical_operator),
            ("kernel_kind", kernel_kind),
            ("source_owner", framework),
            ("workload", shapes),
        )
        if not value
    ]
    if missing_metadata:
        raise RuntimeError(
            "KB producer task is missing required producer metadata: "
            + ", ".join(missing_metadata)
        )
    if not fellow.endswith("-fellow"):
        raise RuntimeError(f"Invalid KernelForge fellow {fellow!r}")
    fellow_backend = fellow.removesuffix("-fellow")
    if expected_backend != backend or expected_backend != fellow_backend:
        raise RuntimeError(
            "KB producer kernel_kind/fellow/backend mismatch: "
            f"kernel_kind={kernel_kind!r} requires {expected_backend!r}, "
            f"fellow={fellow!r}, backend={backend!r}"
        )
    _check_producer_cli_capabilities(forge_bin)


def _build_forge_command(
    *,
    forge_bin: str,
    kernel_file: Path,
    driver_dest: Path,
    workspace: str,
    experiments_dir: Path,
    result_json: Path,
    program_md: Path,
    agent_config: dict[str, Any],
    gpu_arch: str,
    fellow: str,
    task_type: str,
    source_files: list[Path],
    target_functions: list[str],
    logical_operator: str,
    framework: str,
    shapes: dict[str, Any] | None,
) -> list[str]:
    """Build argv without shell parsing so the producer contract is testable."""
    cmd = [
        forge_bin,
        "forge-loop",
        "--kernel",
        str(kernel_file),
        "--driver",
        str(driver_dest),
        "--workspace",
        str(workspace),
        "--experiments-dir",
        str(experiments_dir),
        "--result-json",
        str(result_json),
        "--snr-threshold",
        str(agent_config.get("snr_threshold", 30.0)),
        "--max-iters",
        str(agent_config.get("max_iters", 2)),
        "--max-hours",
        str(_forge_max_hours(agent_config)),
        "--gpu-target",
        gpu_arch,
        "--fellow",
        fellow,
        "--git-branch",
        "forge-optimize",
        "--program-md-file",
        str(program_md),
        "--model",
        str(agent_config.get("model", "claude-opus-4-8")),
        "--permission-mode",
        str(agent_config.get("permission_mode", "acceptEdits")),
        "--task-type",
        task_type,
    ]
    if source_files:
        cmd.extend(["--source-files", ",".join(str(path) for path in source_files)])
    if target_functions:
        cmd.extend(["--target-functions", ",".join(str(name) for name in target_functions)])
    if logical_operator:
        cmd.extend(["--operator-name", logical_operator])
    if framework:
        cmd.extend(["--framework", framework])
    if shapes:
        cmd.extend(
            [
                "--shapes-json",
                json.dumps(shapes, sort_keys=True, separators=(",", ":")),
                "--workload-key",
                _canonical_workload_key(shapes),
            ]
        )
    return cmd


def _read_forge_result(result_json: Path, stdout: str) -> dict[str, Any] | None:
    """Read the structured result file, then fall back to the stdout sentinel."""
    try:
        result = json.loads(result_json.read_text())
        if isinstance(result, dict):
            return result
    except (OSError, json.JSONDecodeError):
        pass
    matches = re.findall(
        re.escape(_FORGE_RESULT_SENTINEL)
        + r"(.*?)"
        + re.escape(_FORGE_RESULT_SENTINEL),
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


def _publication_status(result: dict[str, Any] | None, *, required: bool) -> dict[str, Any]:
    """Prove the current durable best has no pending remote publication."""
    def _authoritative(publication: Any) -> bool:
        if not isinstance(publication, dict) or "published_commit" not in publication:
            return False
        return (
            "pending_commit" in publication
            or publication.get("status") == "warm_start_existing"
        )

    status: dict[str, Any] = {
        "required": required,
        "published": False,
        "latest_best_published": False,
        "authoritative": False,
        "state": "result_missing" if result is None else "not_published",
        "reason": "",
        "best_commit": "",
        "published_commit": "",
        "pending_commit": "",
    }
    if result is None:
        status["reason"] = "forge_result_missing"
        return status

    best_commit = str(result.get("best_commit") or "")
    status["best_commit"] = best_commit
    top_level = result.get("remote_publication")
    publication = top_level if isinstance(top_level, dict) else None
    source = "remote_publication"
    kb_experience = result.get("kb_experience")
    top_level_authoritative = _authoritative(publication)
    if not top_level_authoritative and isinstance(kb_experience, dict):
        nested = kb_experience.get("publication")
        nested_authoritative = _authoritative(nested)
        if nested_authoritative or publication is None:
            publication = nested
            source = "kb_experience.publication"
    if publication is None:
        status["state"] = "schema_unsupported"
        status["reason"] = (
            "forge result has neither remote_publication nor "
            "kb_experience.publication"
        )
        return status

    status["source"] = source
    status["authoritative"] = _authoritative(publication)
    published_commit = str(publication.get("published_commit") or "")
    pending_commit = str(publication.get("pending_commit") or "")
    status["published_commit"] = published_commit
    status["pending_commit"] = pending_commit
    status["publication_state"] = str(publication.get("status") or "")
    latest_best_published = bool(
        status["authoritative"]
        and best_commit
        and published_commit == best_commit
        and not pending_commit
    )
    status["published"] = latest_best_published
    status["latest_best_published"] = latest_best_published
    status["state"] = (
        "published"
        if latest_best_published
        else str(publication.get("status") or "not_published")
    )
    if not best_commit:
        status["reason"] = "best_commit_missing"
    elif not status["authoritative"]:
        status["reason"] = "publication_schema_incomplete"
    elif pending_commit:
        status["reason"] = f"publication_pending:{pending_commit}"
    elif published_commit != best_commit:
        status["reason"] = (
            f"published_commit_mismatch:{published_commit or '<none>'}!={best_commit}"
        )

    if isinstance(kb_experience, dict):
        final_write = kb_experience.get("write")
        if isinstance(final_write, dict):
            status["final_write"] = {
                key: final_write[key]
                for key in ("written", "reason")
                if key in final_write
            }
    return status


def _validate_producer_outcome(
    *,
    returncode: int | None,
    timed_out: bool,
    forge_result: dict[str, Any] | None,
    kb_status: dict[str, Any],
) -> None:
    """Reject process and publication failures before Arena scoring."""
    if timed_out:
        raise RuntimeError("KB producer forge-loop timed out before completion")
    if returncode != 0:
        raise RuntimeError(f"KB producer forge-loop failed with exit code {returncode}")
    if forge_result is None:
        raise RuntimeError("KB producer forge-loop returned no structured result")
    if not kb_status.get("authoritative"):
        raise RuntimeError(
            "KB producer result lacks authoritative remote publication state"
        )
    if not kb_status.get("latest_best_published"):
        raise RuntimeError(
            "KB producer did not publish the latest local best: "
            f"{kb_status.get('state')} ({kb_status.get('reason') or 'no reason reported'})"
        )


def _write_forge_status(
    experiments_dir: Path,
    *,
    returncode: int | None,
    timed_out: bool,
    result: dict[str, Any] | None,
    kb_status: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "exit_code": returncode,
        "timed_out": timed_out,
        "kb": kb_status,
        "experiment_id": (result or {}).get("experiment_id", ""),
        "improved": bool((result or {}).get("improved", False)),
        "best_iteration": (result or {}).get("best_iteration", 0),
    }
    experiments_dir.mkdir(parents=True, exist_ok=True)
    (experiments_dir / _KB_STATUS_FILE).write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary


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
    agent_config = _apply_run_kb_config(agent_config, eval_config)
    producer = _kb_mode(agent_config) == "producer"

    # Task config: locate the kernel file + target function(s).
    with open(task_config_dir, "r") as f:
        task_config = yaml.safe_load(f) or {}
    source_files = task_config.get("source_file_path") or []
    if not isinstance(source_files, list) or not source_files:
        raise RuntimeError(f"Task config has no source_file_path: {task_config_dir}")
    kernel_file = _resolve_kernel_file(workspace, source_files, task_config)
    editable_source_entries = _declared_editable_sources(task_config)
    # Resolve the complete edit allowlist. source_file_path[0] remains the anchor;
    # editable_sources adds dependent files without changing anchor semantics.
    all_source_files = _resolve_all_source_files(
        workspace,
        editable_source_entries,
        task_config,
        logger,
        strict=producer,
    )
    target_funcs = task_config.get("target_kernel_functions") or []
    if not isinstance(target_funcs, list):
        raise ValueError("target_kernel_functions must be a list")
    target_funcs = [str(name).strip() for name in target_funcs if str(name).strip()]
    if producer and not target_funcs:
        raise ValueError(
            "KB producer task requires concrete target_kernel_functions"
        )
    task_type = str(task_config.get("task_type") or "").strip()

    gpu_arch = _resolve_gpu_arch(eval_config)
    fellow = _resolve_fellow(task_config, agent_config)
    backend = fellow.split("-")[0]
    logical_operator = _logical_operator(task_config, producer=producer)
    kernel_kind = _resolve_kernel_kind(task_config, backend, producer=producer)
    shapes = _resolve_shapes(
        task_config,
        task_config_dir,
        agent_config,
        producer=producer,
    )
    framework = _resolve_framework(task_config, str(kernel_file), all_source_files)

    # Preserve all existing parent credentials. Producer mode validates gbrain
    # credentials but never manufactures or renames credential variables.
    env = os.environ.copy()
    env["IS_SANDBOX"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.pop("ANTHROPIC_API_KEY", None)
    if producer:
        _producer_preflight(
            forge_bin=forge_bin,
            env=env,
            logical_operator=logical_operator,
            kernel_kind=kernel_kind,
            framework=framework,
            shapes=shapes,
            fellow=fellow,
            backend=backend,
        )

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
    _write_program_md(
        task_config,
        target_funcs,
        gpu_arch,
        backend,
        all_source_files,
        program_md,
    )

    # Repository / image_kernel tasks bring a cloned repo (with its own nested
    # .git) into the workspace. Strip it before outer git init so keep/revert
    # tracks the real source files.
    if task_type.lower() in ("repository", "image_kernel"):
        _strip_nested_git(workspace, logger)

    # The loop needs a git repo for the keep/revert pattern.
    _init_git_workspace(workspace, logger)

    experiments_dir = Path(workspace) / "forge_experiments"
    result_json = experiments_dir / "forge_result.json"
    result_json.unlink(missing_ok=True)

    model = str(agent_config.get("model", "claude-opus-4-8"))
    cmd_parts = _build_forge_command(
        forge_bin=forge_bin,
        kernel_file=kernel_file,
        driver_dest=driver_dest,
        workspace=workspace,
        experiments_dir=experiments_dir,
        result_json=result_json,
        program_md=program_md,
        agent_config=agent_config,
        gpu_arch=gpu_arch,
        fellow=fellow,
        task_type=task_type,
        source_files=all_source_files,
        target_functions=target_funcs,
        logical_operator=logical_operator if producer else "",
        framework=framework,
        shapes=shapes,
    )
    # Human-readable rendering for the log only; the process is launched from the
    # argv list (cmd_parts) with shell=False, so no shell parsing is involved.
    cmd = " ".join(shlex.quote(p) for p in cmd_parts)

    logger.info("Forge Preflight")
    logger.info(f"  KB mode:     {'producer' if producer else 'compatibility'}")
    logger.info(f"  forge bin:   {forge_bin}")
    logger.info(f"  kernel:      {kernel_file}")
    logger.info(f"  driver:      {driver_dest}")
    logger.info(f"  gpu target:  {gpu_arch}")
    logger.info(f"  model:       {model}")
    logger.info(f"  fellow:      {fellow} (inferred from task_type={task_config.get('task_type')!r})")
    logger.info(f"  operator:    {logical_operator or '<compatibility inference>'}")
    logger.info(f"  kernel kind: {kernel_kind}")
    logger.info(f"  source owner:{framework or '<unknown>'}")
    logger.info(
        "  workload:    %s",
        _canonical_workload_key(shapes) if shapes else "<forge default>",
    )
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

    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        logger.warning(f"Forge loop timed out after {timeout_seconds}s; terminating process group")
        _terminate_process_group(process, logger)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.error("Forge process leader did not exit after process-group termination")

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
    forge_result = _read_forge_result(result_json, "\n".join(stdout_lines))
    kb_status = _publication_status(forge_result, required=producer)
    _write_forge_status(
        experiments_dir,
        returncode=process.returncode,
        timed_out=timed_out,
        result=forge_result,
        kb_status=kb_status,
    )
    logger.info(
        "Forge result: exit=%s timed_out=%s KB=%s (%s)",
        process.returncode,
        timed_out,
        kb_status["state"],
        kb_status["reason"] or "no error",
    )
    if producer:
        _validate_producer_outcome(
            returncode=process.returncode,
            timed_out=timed_out,
            forge_result=forge_result,
            kb_status=kb_status,
        )
    return output
