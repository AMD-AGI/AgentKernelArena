# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Workspace integrity guard for task harness files.

Agents should optimize kernels, not the measurement harness.  This module
records a digest snapshot of task-owned harness files before an agent runs and
verifies that those files are unchanged before scoring.

Note that the protected set is defined by naming patterns, not only by location, so a
file an agent creates anywhere in the workspace can fall inside it. Optimization
discards these files before evaluation; validation rejects them after its audit.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
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
    source_aliases: dict[str, tuple[str, str]] = field(default_factory=dict)
    symlink_protected_sources: tuple[str, ...] = ()
    path_states: dict[str, dict[str, str]] = field(default_factory=dict)
    digest_modes: dict[str, str] = field(default_factory=dict)


class HarnessIntegrityError(RuntimeError):
    """A strict failure with actual observations available for an archive."""

    def __init__(self, message: str, verification: dict):
        super().__init__(message)
        self.verification = verification


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
    trusted_inputs = _trusted_evaluation_input_paths(root)
    partial_sources = _editable_python_function_bodies(root)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if trusted_inputs:
            if (rel.as_posix() in trusted_inputs or path.resolve() in configured_entrypoints
                    or path in partial_sources):
                yield path
            continue
        if set(rel.parts) & _IGNORED_RUNTIME_DIRS:
            continue
        if (_is_protected_path(rel) or path.resolve() in configured_entrypoints
                or path in partial_sources):
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


def _sha256_python_harness(path: Path, editable_targets: set[str]) -> str:
    """Hash the harness portion of a co-located Python kernel entrypoint.

    ROCmBench keeps editable Triton code and pytest harnesses in one module.
    Imports, declared target functions, Triton JIT helpers, and top-level
    helpers called directly by target decorators are legitimate optimization
    surface, so omit their complete AST nodes.  Test/benchmark functions,
    unrelated ordinary Python helpers, module constants, and executable
    statements remain in the digest.
    """

    try:
        tree = ast.parse(path.read_text())
    except (OSError, UnicodeDecodeError, SyntaxError):
        return "invalid-python:" + _sha256(path)

    decorator_helpers = _target_decorator_helper_names(tree, editable_targets)
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
        and not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in editable_targets | decorator_helpers
        )
        and not _is_triton_jit_function(node)
    ]
    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _editable_source_paths(root: Path) -> set[Path]:
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
    return editable


def _editable_python_function_bodies(root: Path) -> dict[Path, set[str]]:
    """Opt in to body-only edits while freezing source interfaces and launchers."""
    policy = _task_config(root).get("harness_protection", {})
    configured = policy.get("editable_python_function_bodies", {}) if isinstance(policy, dict) else {}
    if not isinstance(configured, dict):
        raise RuntimeError("editable_python_function_bodies must be a mapping")
    editable = _editable_source_paths(root)
    result = {}
    for relative, names in configured.items():
        if not isinstance(relative, str):
            raise RuntimeError("Invalid editable Python function-body contract")
        path = root / relative
        if (Path(relative).is_absolute()
                or ".." in Path(relative).parts or path.suffix != ".py"
                or path.is_symlink() or not path.is_file() or path.resolve() not in editable
                or not path.resolve().is_relative_to(root.resolve())
                or not isinstance(names, list) or not names
                or any(not isinstance(name, str) or not name.isidentifier() for name in names)):
            raise RuntimeError("Invalid editable Python function-body contract")
        result[path] = set(names)
    return result


def _sha256_python_function_bodies(path: Path, names: set[str]) -> str:
    try:
        tree = ast.parse(path.read_text())
    except (OSError, UnicodeDecodeError, SyntaxError):
        return "invalid-python:" + _sha256(path)
    found = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            found.append(node.name)
            node.body = [ast.Pass()]
    if sorted(found) != sorted(names):
        raise RuntimeError("Editable Python functions are missing or duplicated: " + str(path))
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode("utf-8")).hexdigest()


def _task_input_paths(
    root: Path, task_root: Path, *, include_symlinks: bool = False
) -> set[str]:
    """Freeze non-editable files shipped by the task, including data and oracles.

    Use the task package to identify inputs, not files generated by baseline
    compilation or agent preparation. Repository caches are not task packages;
    the existing harness rules still apply inside those trees.
    """
    if not task_root.is_dir():
        raise FileNotFoundError(f"Original task directory is unavailable: {task_root}")
    config = _task_config(root)
    editable = _editable_source_paths(root)
    trusted = isinstance(config.get("trusted_evaluation"), dict)
    policy = config.get("harness_protection", {})
    complete_inputs = trusted or (isinstance(policy, dict) and policy.get("freeze_task_inputs") is True)
    outputs = policy.get("runtime_outputs", []) if isinstance(policy, dict) else []
    if (not isinstance(outputs, list) or any(not isinstance(value, str) or not value
            or Path(value).is_absolute() or ".." in Path(value).parts or value == "." for value in outputs)):
        raise RuntimeError("harness_protection runtime_outputs must be task-relative paths")

    repo_subdir = config.get("repo_subdir")
    if not repo_subdir:
        origin = config.get("image_repo_path") or config.get("repo_url")
        if origin:
            repo_subdir = str(origin).rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    runtime_dirs = _IGNORED_RUNTIME_DIRS | {"build", "logs", "perf", ".pytest_cache"}
    output_names = {
        "task_result.yaml", "validation_report.yaml", ".validation_complete",
        "validation_integrity.json", "baseline_perf.yaml", "optimized_perf.yaml",
    }
    protected = set()
    for path in task_root.rglob("*"):
        if not path.is_file() and not (include_symlinks and path.is_symlink()):
            continue
        rel = path.relative_to(task_root)
        if any(rel.is_relative_to(value) for value in outputs):
            continue
        if complete_inputs:
            # A trusted package can contain immutable helpers/data in nested
            # directories named build or logs. Only top-level runtime outputs
            # and Python bytecode caches are outside its input contract.
            runtime_output = (
                rel.parts[0] in runtime_dirs | {".validator_audit"}
                or "__pycache__" in rel.parts[:-1]
                or (len(rel.parts) == 1 and rel.name in output_names)
            )
        else:
            runtime_output = bool(set(rel.parts[:-1]) & runtime_dirs) or rel.name in output_names
        if runtime_output:
            continue
        if repo_subdir and rel.is_relative_to(repo_subdir):
            continue
        candidate = root / rel
        if candidate.resolve() not in editable or (include_symlinks and path.is_symlink()):
            protected.add(rel.as_posix())
    return protected


def _trusted_evaluation_input_paths(root: Path) -> set[str]:
    """Protect an opted-in package's complete non-editable input closure.

    Ordinary optimization also supplies task_root, but validators and direct
    snapshot callers may only have the materialized workspace. The descriptor
    must provide the same immutable boundary in both cases. This includes
    helpers imported by the declared contract, not just its named entrypoints.
    """
    config = _task_config(root)
    descriptor = config.get("trusted_evaluation")
    if descriptor is None:
        policy = config.get("harness_protection", {})
        if isinstance(policy, dict) and policy.get("freeze_task_inputs") is True:
            return _task_input_paths(root, root)
        return set()
    if (not isinstance(descriptor, dict)
            or type(descriptor.get("schema_version")) is not int
            or descriptor["schema_version"] != 1):
        raise RuntimeError("Unsupported trusted_evaluation harness contract")
    references = descriptor.get("reference_sources", {})
    if not isinstance(references, dict):
        raise RuntimeError("trusted_evaluation reference_sources must be a mapping")
    required = [
        descriptor.get("case_manifest", "cases.json"),
        descriptor.get("contract_file", "ut/evaluation_contract.py"),
        descriptor.get("source_guard", "ut/source_guard.py"),
        *references.values(),
    ]
    if "fixture_manifest" in descriptor:
        required.append(descriptor["fixture_manifest"])
    if "case_parallelism" in descriptor:
        from src.case_parallelism import verify_helpers
        verify_helpers(root, config)
        required.extend(["ut/case_parallel.py", "ut/gpu_binding.py"])
    protected = _task_input_paths(root, root)
    editable = _editable_source_paths(root)
    for value in required:
        if not isinstance(value, str) or not value:
            raise RuntimeError("Trusted evaluation input paths must be nonempty strings")
        relative = Path(value)
        path = root / relative
        if (relative.is_absolute() or ".." in relative.parts
                or not path.resolve().is_relative_to(root.resolve())
                or not path.is_file() or path.is_symlink()):
            raise RuntimeError(f"Missing or unsafe trusted evaluation input: {value}")
        if path.resolve() in editable:
            raise RuntimeError(f"Trusted evaluation input is declared editable: {value}")
        protected.add(relative.as_posix())
    return protected


def _editable_source_aliases(task_root: Path) -> dict[str, tuple[str, str]]:
    """Record task-owned links to editable files, separately from their bytes."""
    task_root = task_root.resolve()
    editable = _editable_source_paths(task_root)
    aliases = {}
    for path in task_root.rglob("*"):
        if not path.is_symlink():
            continue
        target = path.resolve()
        if target not in editable and not (
            target.is_dir() and any(source.is_relative_to(target) for source in editable)
        ):
            continue
        relative = path.relative_to(task_root).as_posix()
        link = path.readlink()
        if link.is_absolute() or not target.is_relative_to(task_root) or not target.exists():
            raise RuntimeError(f"Invalid editable source alias in task: {relative}")
        aliases[relative] = (str(link), target.relative_to(task_root).as_posix())
    return aliases


def _symlink_protected_sources(task_root: Path) -> tuple[str, ...]:
    """Record canonical editable paths for tasks that forbid new source links.

    Shipped relative aliases remain governed by the existing alias contract.
    Protect their canonical targets so a new link cannot redirect the actual
    implementation, even when it keeps the original alias text unchanged.
    """
    task_root = task_root.resolve()
    policy = _task_config(task_root).get("harness_protection", {})
    if (
        not isinstance(policy, dict)
        or policy.get("reject_new_source_symlinks") is not True
    ):
        return ()
    sources = _editable_source_paths(task_root)
    for source in sources:
        if not source.is_relative_to(task_root) or not source.is_file():
            raise RuntimeError(
                "Source symlink protection requires existing task-contained "
                f"editable sources: {source}"
            )
    return tuple(sorted(
        source.relative_to(task_root).as_posix() for source in sources
    ))


def _verify_source_path_types(root: Path, sources: Iterable[str]) -> None:
    root = root.resolve()
    for relative in sources:
        source = root / relative
        # Checking every component also catches a regular source directory
        # replaced by a symlink to an otherwise identical tree.
        components = [source]
        components.extend(
            parent for parent in source.parents
            if parent != root and root in parent.parents
        )
        if any(path.is_symlink() for path in components) or not source.is_file():
            raise RuntimeError(
                f"Editable source path changed or missing: {relative}; "
                "new source symlinks are forbidden by harness_protection. "
                "Kernel score is rejected."
            )


def _verify_source_aliases(root: Path, aliases: dict[str, tuple[str, str]]) -> None:
    root = root.resolve()
    for relative, (link, target) in aliases.items():
        path = root / relative
        try:
            intact = (
                path.is_symlink()
                and str(path.readlink()) == link
                and path.resolve(strict=True) == root / target
            )
        except (OSError, RuntimeError):
            intact = False
        if not intact:
            raise RuntimeError(
                f"Editable source alias changed or missing: {relative}; "
                f"expected a task-contained symlink to {target}. Kernel score is rejected."
            )


def verify_task_source_aliases(root: Path, task_root: Path) -> None:
    """Reject detached source aliases when re-evaluating a prior workspace."""
    _verify_source_aliases(Path(root), _editable_source_aliases(Path(task_root)))
    _verify_source_path_types(Path(root), _symlink_protected_sources(Path(task_root)))


def _protected_digests(root: Path, extra_paths: Iterable[str] = ()) -> dict[str, str]:
    editable_entrypoints = _editable_entrypoint_targets(root)
    partial_sources = _editable_python_function_bodies(root)
    digests = {}
    paths = set(_iter_protected_files(root))
    paths.update(
        root / relative for relative in extra_paths if (root / relative).is_file()
    )
    for path in sorted(paths):
        resolved = path.resolve()
        if path in partial_sources:
            digest = _sha256_python_function_bodies(path, partial_sources[path])
        elif resolved in editable_entrypoints:
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
        "symlink_protected_sources": list(_symlink_protected_sources(root)),
        "editable_python_function_bodies": {
            str(path.relative_to(root)): sorted(names)
            for path, names in sorted(_editable_python_function_bodies(root).items())
        },
    }


def _validation_protected_paths(root: Path) -> set[str]:
    paths = {str(path.relative_to(root)) for path in _iter_protected_files(root)}
    config = _task_config(root)
    policy = config.get("harness_protection", {})
    if (isinstance(config.get("trusted_evaluation"), dict)
            or isinstance(policy, dict) and policy.get("freeze_task_inputs") is True):
        # Reuse the complete-input boundary and its runtime-output exclusions.
        # New shadow modules and aliases, including dangling links, are inputs.
        paths.update(_task_input_paths(root, root, include_symlinks=True))
    else:
        paths.update(
            str(path.relative_to(root)) for path in root.rglob("*")
            if path.is_symlink() and _is_protected_path(path.relative_to(root))
            and not set(path.relative_to(root).parts) & _IGNORED_RUNTIME_DIRS
        )
    return paths


def _path_states(root: Path, paths: Iterable[str]) -> dict[str, dict[str, str]]:
    """Record lexical types and link text without reading link targets."""
    names = {"."}
    for relative in paths:
        path = Path(relative)
        names.add(path.as_posix())
        names.update(parent.as_posix() for parent in path.parents)
    result = {}
    for relative in sorted(names):
        path = root / relative
        if path.is_symlink():
            result[relative] = {"kind": "symlink", "target": str(path.readlink())}
        elif path.is_file():
            result[relative] = {"kind": "file"}
        elif path.is_dir():
            result[relative] = {"kind": "directory"}
        elif path.exists():
            result[relative] = {"kind": "other"}
        else:
            result[relative] = {"kind": "missing"}
    return result


def _digest_modes(root: Path, paths: Iterable[str]) -> dict[str, str]:
    entrypoints = _editable_entrypoint_targets(root)
    partial = _editable_python_function_bodies(root)
    return {
        relative: (
            "masked_python_function_bodies_sha256" if root / relative in partial
            else "masked_python_harness_ast_sha256" if (root / relative).resolve() in entrypoints
            else "file_sha256"
        )
        for relative in paths
    }


def snapshot_workspace_harness(
    root: Path, *, task_root: Path | None = None, strict: bool = False
) -> WorkspaceSnapshot:
    """Capture harness digests and immutable task-package inputs.

    Callers running optimization must pass the original task directory. Keep the
    snapshot outside the agent workspace and verify it before final evaluation.
    trusted_evaluation packages additionally protect their non-editable inputs
    when only the materialized workspace is available.
    """

    root = Path(root)
    original_root = Path(task_root) if task_root is not None else root
    source_aliases = _editable_source_aliases(original_root)
    symlink_protected_sources = _symlink_protected_sources(original_root)
    _verify_source_aliases(root, source_aliases)
    _verify_source_path_types(root, symlink_protected_sources)
    task_inputs = (
        _task_input_paths(root, Path(task_root)) if task_root is not None else set()
    )
    if strict:
        task_inputs.update(_validation_protected_paths(root))
    missing = sorted(
        relative for relative in task_inputs
        if not (root / relative).is_file()
        and not (strict and (root / relative).is_symlink() and (root / relative).is_dir())
    )
    if missing:
        raise RuntimeError(f"Task inputs missing before agent execution: {missing}")
    digests = _protected_digests(root, task_inputs)
    return WorkspaceSnapshot(
        root=root, digests=digests, source_aliases=source_aliases,
        symlink_protected_sources=symlink_protected_sources,
        path_states=_path_states(root, task_inputs | set(digests) | set(source_aliases) | set(symlink_protected_sources)) if strict else {},
        digest_modes=_digest_modes(root, digests) if strict else {},
    )


def _verify_validation_harness(snapshot: WorkspaceSnapshot) -> dict:
    before = snapshot.digests
    observed = None
    states = None
    errors = []
    added = []
    changed_paths = []
    changed_config = []
    config_digests = {}
    try:
        if not snapshot.path_states:
            raise RuntimeError("Strict verification requires a strict pre-agent snapshot")
        states = _path_states(snapshot.root, snapshot.path_states)
        changed_paths = sorted(
            name for name, value in snapshot.path_states.items()
            if states.get(name) != value
        )
        if changed_paths:
            # A changed directory/link may redirect reads. Retain its topology,
            # but do not follow it or invent a completed after-digest scan.
            errors.append("Protected path types or aliases changed: " + ", ".join(changed_paths))
        else:
            _verify_source_aliases(snapshot.root, snapshot.source_aliases)
            _verify_source_path_types(snapshot.root, snapshot.symlink_protected_sources)
            # Never derive an after-scan's masking/output policy from changed
            # task configuration and label it with the original digest modes.
            config_digests = {
                name: _sha256(snapshot.root / name)
                for name in ("config.yaml", "config.yml") if name in before
            }
            changed_config = sorted(
                name for name, digest in config_digests.items() if before[name] != digest
            )
            if changed_config:
                raise RuntimeError("Protected configuration changed; after-scan withheld: " + ", ".join(changed_config))
            paths = _validation_protected_paths(snapshot.root) | set(before)
            states.update(_path_states(snapshot.root, paths))
            added = sorted(paths - set(snapshot.path_states))
            if any(states.get(name, {}).get("kind") == "symlink" for name in added):
                errors.append("New protected symlinks are forbidden")
            else:
                observed = _protected_digests(snapshot.root, paths)
    except (OSError, RuntimeError, ValueError) as exc:
        errors.append(str(exc))
    modified = sorted(set(changed_config) | {
        name for name, digest in before.items()
        if observed is not None and name in observed and observed[name] != digest
    })
    deleted = sorted(
        name for name in before
        if states is not None and states.get(name, {}).get("kind") == "missing"
    )
    result = {
        "status": "FAIL" if errors or added or modified or deleted else "PASS",
        "added_file_policy": "reject_without_discard",
        "before_digests": before, "after_digests": observed,
        "after_config_digests": config_digests,
        "digest_modes": snapshot.digest_modes,
        "before_path_states": snapshot.path_states, "after_path_states": states,
        "source_aliases": snapshot.source_aliases,
        "added": added, "modified": modified, "deleted": deleted,
        "changed_paths": changed_paths, "errors": errors, "files_discarded": [],
    }
    if result["status"] != "PASS":
        raise HarnessIntegrityError(
            "Post-validator integrity failed: "
            + "; ".join([*errors, f"added={added}", f"modified={modified}", f"deleted={deleted}"]),
            result,
        )
    return result


def verify_workspace_harness(
    snapshot: WorkspaceSnapshot, logger=None, *, reject_added: bool = False
) -> dict | None:
    """Reject tampering with protected harness files; discard ones the agent added.

    Editing or deleting a harness file the task shipped is harness hacking and the score
    is refused.  A file the agent *created* is a different case: the baseline harness ran
    without it, so deleting it restores exactly the state that was measured, and no score
    can have been influenced by it.  Discarding it is therefore as safe as rejecting the
    run, and it does not throw away hours of legitimate kernel work because the agent left
    a scratch file whose name happened to end in ``_test.py``.

    ``reject_added=True`` is the post-validator policy: added inputs may already
    have influenced the audit and remain in place as failure evidence.

    Deletions are always logged: a silent removal would be worse than a hard failure.
    """

    if reject_added:
        return _verify_validation_harness(snapshot)

    _verify_source_aliases(snapshot.root, snapshot.source_aliases)
    _verify_source_path_types(snapshot.root, snapshot.symlink_protected_sources)

    def _scan() -> dict[str, str]:
        # Preserve the editable-body masking used for colocated kernel/harness
        # files. A raw SHA here would reject legitimate target-function edits.
        # Recheck the original paths even when their names do not match a
        # harness pattern (e.g. session_cases.json or a reference module).
        return _protected_digests(snapshot.root, snapshot.digests)

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
