# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Workspace integrity guard for task harness files.

Agents should optimize kernels, not the measurement harness.  This module
records a digest snapshot of task-owned harness files before an agent runs and
verifies that those files are unchanged before scoring.

Note that the protected set is defined by naming patterns, not only by location, so a
file an agent creates anywhere in the workspace can fall inside it.  Such files are
discarded rather than treated as tampering; see ``verify_workspace_harness``.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from .perf_helper_materialization import configured_performance_entrypoints


_HARNESS_DIRS = {
    "script",
    "scripts",
    "test",
    "tests",
}
_HARNESS_FILE_NAMES = {
    "_aka_benchmark.py",
    "config.yaml",
    "config.yml",
    "conftest.py",
    "hip_graph_benchmark.hpp",
    "performance_utils_pytest.py",
}
_HARNESS_FILE_SUFFIXES = (
    "_test.py",
    "_test.cpp",
    "_test.cu",
    "_test.hip",
    "_harness.py",
)
_IGNORED_RUNTIME_DIRS = {
    ".git",
    ".task-venv",
    ".validator_torch_extensions",
    ".venv",
    "__pycache__",
}


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """Immutable digest snapshot of protected workspace files."""

    root: Path
    digests: dict[str, str]


def _is_protected_path(rel: Path) -> bool:
    parts = set(rel.parts[:-1])
    name = rel.name
    if parts & _HARNESS_DIRS:
        return True
    if name in _HARNESS_FILE_NAMES:
        return True
    return name.endswith(_HARNESS_FILE_SUFFIXES)


def _iter_protected_files(root: Path) -> Iterable[Path]:
    configured_entrypoints = {
        path.resolve() for path in configured_performance_entrypoints(root)
    }
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if set(rel.parts) & _IGNORED_RUNTIME_DIRS:
            continue
        if _is_protected_path(rel) or path.resolve() in configured_entrypoints:
            yield path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _task_config(root: Path) -> dict:
    for name in ("config.yaml", "config.yml"):
        path = root / name
        if not path.is_file():
            continue
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except (OSError, yaml.YAMLError):
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def _string_list(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _editable_entrypoint_targets(root: Path) -> dict[Path, set[str]]:
    """Return co-located benchmark/source files and their editable functions.

    A small number of ROCmBench tasks intentionally keep the Triton kernel and
    pytest harness in one configured file. Protecting that whole entrypoint
    would make the optimization task impossible, so its kernel implementation
    surface is excluded from the integrity digest while benchmark tests and
    task-owned data remain protected.
    """

    config = _task_config(root)
    targets = {
        name.strip()
        for configured in _string_list(config.get("target_kernel_functions"))
        for name in configured.split(",")
        if name.strip()
    }
    if not targets:
        return {}
    source_paths = set()
    for configured in _string_list(config.get("source_file_path")):
        path = (root / configured).resolve()
        if path.is_file():
            source_paths.add(path)
    entrypoints = {
        path.resolve() for path in configured_performance_entrypoints(root)
    }
    # Legacy instruction2triton tasks embed the editable Triton target in the
    # configured pytest/performance entrypoint but leave source_file_path empty.
    # Treat that entrypoint as the implied source for this family only. The AST
    # digest still protects tests, ordinary helpers, constants, and executable
    # harness statements; only imports and Triton target/helper nodes are masked.
    if config.get("task_type") == "instruction2triton" and not source_paths:
        source_paths.update(entrypoints)
    return {
        path: targets
        for path in source_paths & entrypoints
        if path.suffix == ".py"
    }


def _is_triton_jit_function(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "triton"
            and target.attr == "jit"
        ):
            return True
        if isinstance(target, ast.Name) and target.id == "jit":
            return True
    return False


def _sha256_python_harness(path: Path, editable_targets: set[str]) -> str:
    """Hash the harness portion of a co-located Python kernel entrypoint.

    ROCmBench keeps editable Triton code and pytest harnesses in one module.
    Imports, declared target functions, and Triton JIT helpers are legitimate
    optimization surface, so omit their complete AST nodes.  Test/benchmark
    functions, ordinary Python helpers, module constants, and executable
    statements remain in the digest.
    """

    try:
        tree = ast.parse(path.read_text())
    except (OSError, UnicodeDecodeError, SyntaxError):
        return "invalid-python:" + _sha256(path)

    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
        and not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in editable_targets
        )
        and not _is_triton_jit_function(node)
    ]
    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _protected_digests(root: Path) -> dict[str, str]:
    editable_entrypoints = _editable_entrypoint_targets(root)
    digests = {}
    for path in sorted(_iter_protected_files(root)):
        resolved = path.resolve()
        if resolved in editable_entrypoints:
            digest = _sha256_python_harness(path, editable_entrypoints[resolved])
        else:
            digest = _sha256(path)
        digests[str(path.relative_to(root))] = digest
    return digests


def describe_workspace_harness(root: Path) -> dict[str, object]:
    """Return trusted, non-secret facts about the active harness guard.

    Task validators run inside the materialized task workspace and cannot inspect
    the framework source tree that applies the guard.  Expose the effective path
    boundary so validation prompts do not have to infer it from task-local files
    or look for a guard manifest that intentionally does not live in the task.
    """

    root = Path(root)
    editable_entrypoints = _editable_entrypoint_targets(root)
    return {
        "enforced_during_optimization": True,
        "protected_paths": sorted(
            str(path.relative_to(root)) for path in _iter_protected_files(root)
        ),
        "editable_entrypoint_targets": {
            str(path.relative_to(root)): sorted(targets)
            for path, targets in sorted(
                editable_entrypoints.items(), key=lambda item: str(item[0])
            )
        },
    }


def snapshot_workspace_harness(root: Path) -> WorkspaceSnapshot:
    """Capture digests for task-owned harness and test files."""

    root = Path(root)
    digests = _protected_digests(root)
    return WorkspaceSnapshot(root=root, digests=digests)


def verify_workspace_harness(snapshot: WorkspaceSnapshot, logger=None) -> None:
    """Reject tampering with protected harness files; discard ones the agent added.

    Editing or deleting a harness file the task shipped is harness hacking and the score
    is refused.  A file the agent *created* is a different case: the baseline harness ran
    without it, so deleting it restores exactly the state that was measured, and no score
    can have been influenced by it.  Discarding it is therefore as safe as rejecting the
    run, and it does not throw away hours of legitimate kernel work because the agent left
    a scratch file whose name happened to end in ``_test.py``.

    Deletions are always logged: a silent removal would be worse than a hard failure.
    """

    def _scan() -> dict[str, str]:
        # Preserve the editable-body masking used for colocated kernel/harness
        # files. A raw SHA here would reject legitimate target-function edits.
        return _protected_digests(snapshot.root)

    before = snapshot.digests
    current = _scan()

    discarded = sorted(rel for rel in current if rel not in before)
    for rel in discarded:
        (snapshot.root / rel).unlink()
        message = (
            f"Discarded agent-created file matching a protected harness pattern: {rel}. "
            "It did not exist when the baseline was measured, so it cannot contribute to "
            "the score; scratch files must not use harness/test naming."
        )
        if logger is not None:
            logger.warning(message)

    if discarded:
        current = _scan()

    modified = sorted(
        rel for rel, digest in before.items()
        if rel in current and current[rel] != digest
    )
    deleted = sorted(rel for rel in before if rel not in current)
    if not (modified or deleted):
        return
    details = []
    if modified:
        details.append(f"modified={modified}")
    if deleted:
        details.append(f"deleted={deleted}")
    if discarded:
        details.append(f"discarded={discarded}")
    raise RuntimeError(
        "Protected test/harness files changed during agent execution; "
        "kernel score is rejected to prevent harness hacking: "
        + "; ".join(details)
    )
