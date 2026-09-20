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
from dataclasses import dataclass, field
from pathlib import Path
import symtable
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
class PythonHarnessPolicy:
    """Baseline-owned binding boundary for a co-located Python harness."""

    editable_functions: frozenset[str]
    protected_bindings: frozenset[str]


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """Immutable digest snapshot of protected workspace files."""

    root: Path
    digests: dict[str, str]
    python_policies: dict[str, PythonHarnessPolicy] = field(default_factory=dict)


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
    # harness statements; only kernel-only imports and implementation nodes are masked.
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


def _target_decorator_helper_names(
    tree: ast.Module, editable_targets: set[str]
) -> set[str]:
    """Return top-level helper names called by declared target decorators.

    Autotune and heuristic configuration is often kept in an ordinary helper
    such as ``get_autotune_config()`` rather than written inline in the target
    decorator.  Follow calls from the target's decorators by exactly one hop;
    calls made by those helpers are deliberately not traversed.
    """

    helper_names: set[str] = set()
    for node in tree.body:
        if not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in editable_targets
        ):
            continue
        for decorator in node.decorator_list:
            for child in ast.walk(decorator):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    helper_names.add(child.func.id)
    return helper_names


def _editable_python_functions(tree: ast.Module, editable_targets: set[str]) -> set[str]:
    helpers = _target_decorator_helper_names(tree, editable_targets)
    return editable_targets | {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (node.name in helpers or _is_triton_jit_function(node))
    }


def _python_harness_policy(path: Path, editable_targets: set[str]) -> PythonHarnessPolicy:
    """Freeze harness globals before editable code can redefine its own boundary.

    Python's symbol table distinguishes a test's local ``x`` from the global
    timer/import it uses. This leaves kernel-only imports editable without
    allowing them to rebind protected helpers, constants, imports or builtins.
    """
    tree = ast.parse(path.read_text())
    editable = _editable_python_functions(tree, editable_targets)
    protected = ast.Module(body=[
        node for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
        and not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name in editable)
    ], type_ignores=[])
    symbols = symtable.symtable(ast.unparse(protected), str(path), "exec")
    bindings = {symbol.get_name() for symbol in symbols.get_symbols()}

    def collect_globals(table):
        for child in table.get_children():
            bindings.update(symbol.get_name() for symbol in child.get_symbols()
                            if symbol.is_global())
            collect_globals(child)

    collect_globals(symbols)
    return PythonHarnessPolicy(frozenset(editable), frozenset(bindings - editable))


def _protected_import(
    node: ast.Import | ast.ImportFrom, bindings: frozenset[str],
) -> ast.Import | ast.ImportFrom | None:
    """Retain only import aliases that can replace a protected module binding."""
    if isinstance(node, ast.ImportFrom) and (
        node.module == "__future__" or any(alias.name == "*" for alias in node.names)
    ):
        # Future imports change the surrounding module's semantics; wildcard
        # imports can replace any global. Neither has a kernel-only boundary.
        return node
    names = [alias for alias in node.names
             if (alias.asname or (alias.name.split(".")[0]
                                 if isinstance(node, ast.Import) else alias.name)) in bindings]
    if not names:
        return None
    if isinstance(node, ast.Import):
        return ast.Import(names=names)
    return ast.ImportFrom(module=node.module, names=names, level=node.level)


def _global_imports(node: ast.AST, bindings: frozenset[str]) -> list[ast.AST]:
    """Keep imports in editable scopes that explicitly overwrite harness globals.

    An ordinary function-local import stays editable. ``global timer; import
    payload as timer`` binds the module variable and must remain protected,
    including when nested inside an editable implementation helper.
    """
    scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    nodes, nested = [], []

    def walk_scope(current):
        for child in ast.iter_child_nodes(current):
            if isinstance(child, scopes):
                nested.append(child)
            else:
                nodes.append(child)
                walk_scope(child)

    walk_scope(node)
    globals_ = frozenset(name for child in nodes if isinstance(child, ast.Global)
                         for name in child.names) & bindings
    imports = [projected for child in nodes
               if isinstance(child, (ast.Import, ast.ImportFrom))
               and (projected := _protected_import(child, globals_)) is not None]
    for child in nested:
        imports.extend(_global_imports(child, bindings))
    return imports


def _sha256_python_harness(
    path: Path, editable_targets: set[str], policy: PythonHarnessPolicy | None = None,
) -> str:
    """Hash the harness portion of a co-located Python kernel entrypoint.

    ROCmBench keeps editable Triton code and pytest harnesses in one module.
    Kernel-only imports, declared target functions, Triton JIT helpers, and top-level
    helpers called directly by target decorators are legitimate optimization
    surface, so omit their complete AST nodes.  Test/benchmark functions,
    unrelated ordinary Python helpers, module constants, and executable
    statements remain in the digest. Import bindings used by the original
    harness are frozen even if the candidate moves them into an editable scope.
    """

    try:
        tree = ast.parse(path.read_text())
        if policy is None:
            policy = _python_harness_policy(path, editable_targets)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return "invalid-python:" + _sha256(path)

    editable = policy.editable_functions | (
        _editable_python_functions(tree, editable_targets) - policy.protected_bindings
    )
    protected, global_imports = [], []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            projected = _protected_import(node, policy.protected_bindings)
            if projected is not None:
                protected.append(projected)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in editable:
            imports = _global_imports(node, policy.protected_bindings)
            if imports:
                global_imports.append((node.name, [ast.dump(item, include_attributes=False)
                                                    for item in imports]))
        else:
            protected.append(node)
    tree.body = protected
    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    canonical += repr(global_imports)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _task_input_paths(root: Path, task_root: Path) -> set[str]:
    """Freeze non-editable files shipped by the task, including data and oracles.

    Use the task package to identify inputs, not files generated by baseline
    compilation or agent preparation. Repository caches are not task packages;
    the existing harness rules still apply inside those trees.
    """
    if not task_root.is_dir():
        raise FileNotFoundError(f"Original task directory is unavailable: {task_root}")
    config = _task_config(root)
    editable = {
        (root / name).resolve()
        for key in ("source_file_path", "target_file_path", "editable_sources")
        for name in _string_list(config.get(key))
    }
    if config.get("task_type") == "instruction2triton" and not editable:
        editable.update(
            path.resolve() for path in configured_performance_entrypoints(root)
        )

    repo_subdir = config.get("repo_subdir")
    if not repo_subdir:
        origin = config.get("image_repo_path") or config.get("repo_url")
        if origin:
            repo_subdir = str(origin).rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    runtime_dirs = _IGNORED_RUNTIME_DIRS | {"build", "logs", "perf", ".pytest_cache"}
    output_names = {
        "task_result.yaml", "validation_report.yaml", ".validation_complete",
        "baseline_perf.yaml", "optimized_perf.yaml",
    }
    protected = set()
    for path in task_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(task_root)
        if set(rel.parts[:-1]) & runtime_dirs or rel.name in output_names:
            continue
        if repo_subdir and rel.is_relative_to(repo_subdir):
            continue
        candidate = root / rel
        if candidate.resolve() not in editable:
            protected.add(rel.as_posix())
    return protected


def _protected_digests(
    root: Path, extra_paths: Iterable[str] = (),
    python_policies: dict[str, PythonHarnessPolicy] | None = None,
) -> dict[str, str]:
    editable_entrypoints = _editable_entrypoint_targets(root)
    digests = {}
    paths = set(_iter_protected_files(root))
    paths.update(
        root / relative for relative in extra_paths if (root / relative).is_file()
    )
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved in editable_entrypoints:
            # Configured entrypoints are resolved, while the filesystem scan
            # can also encounter an in-workspace symlink to the same module.
            policy = (python_policies or {}).get(str(resolved.relative_to(root.resolve())))
            digest = _sha256_python_harness(path, editable_entrypoints[resolved], policy)
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


def snapshot_workspace_harness(
    root: Path, *, task_root: Path | None = None
) -> WorkspaceSnapshot:
    """Capture harness digests and, when supplied, immutable task-package inputs.

    Callers running optimization must pass the original task directory. Keep the
    snapshot outside the agent workspace and verify it before final evaluation.
    """

    root = Path(root)
    task_inputs = (
        _task_input_paths(root, Path(task_root)) if task_root is not None else set()
    )
    missing = sorted(
        relative for relative in task_inputs if not (root / relative).is_file()
    )
    if missing:
        raise RuntimeError(f"Task inputs missing before agent execution: {missing}")
    policies = {}
    for path, targets in _editable_entrypoint_targets(root).items():
        try:
            policies[str(path.relative_to(root.resolve()))] = _python_harness_policy(path, targets)
        except (OSError, UnicodeDecodeError, SyntaxError):
            # Invalid Python is protected byte-for-byte by the digest fallback.
            continue
    digests = _protected_digests(root, task_inputs, policies)
    return WorkspaceSnapshot(root=root, digests=digests, python_policies=policies)


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
        # Recheck the original paths even when their names do not match a
        # harness pattern (e.g. session_cases.json or a reference module).
        return _protected_digests(snapshot.root, snapshot.digests, snapshot.python_policies)

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
