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
from typing import Iterable

import yaml

from .perf_helper_materialization import configured_performance_entrypoints
from .task_spec import EditScope, TaskSpec, resolve_task_path


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
    task_spec: TaskSpec | None = None
    initial_symbols: dict[str, frozenset[str]] = field(default_factory=dict)


_V2_OUTPUT_NAMES = {
    "task_result.yaml", "validation_report.yaml", ".validation_complete",
    "baseline_perf.yaml", "optimized_perf.yaml",
}
_V2_RUNTIME_DIRS = _IGNORED_RUNTIME_DIRS | {"build", "logs", "perf", ".pytest_cache"}


def _v2_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if (path.is_file() and not set(relative.parts[:-1]) & _V2_RUNTIME_DIRS
                and path.name not in _V2_OUTPUT_NAMES):
            yield path


def _v2_protected_paths(root: Path, spec: TaskSpec) -> set[str]:
    protected = set()
    for path in _v2_files(root):
        relative = path.relative_to(root)
        edits = [edit for edit in spec.candidate.editable if edit.contains(relative.as_posix())]
        if (not edits or edits[0].scope == "symbols" or _is_protected_path(relative)):
            protected.add(relative.as_posix())
    return protected


def _top_level_names(path: Path) -> frozenset[str]:
    names = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(item.id for target in targets for item in ast.walk(target) if isinstance(item, ast.Name))
    return frozenset(names)


_COMPILER_DECORATORS = frozenset({"triton.jit", "triton.autotune", "triton.heuristics"})
_TEST_LIFECYCLE_NAMES = frozenset({
    "setup_module", "teardown_module", "setup_function", "teardown_function",
    "setup_class", "teardown_class", "setup_method", "teardown_method",
})
_MODULE_INTROSPECTION_HOOKS = frozenset({"__getattr__", "__dir__"})


def _import_bindings(tree: ast.Module) -> dict[str, str]:
    bindings = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                if alias.name != "*":
                    bindings[alias.asname or alias.name] = node.module + "." + alias.name
    return bindings


def _qualified_import(node: ast.AST, bindings: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, ast.Attribute):
        parent = _qualified_import(node.value, bindings)
        return parent + "." + node.attr if parent else None
    return None


def _passive_expression(node: ast.AST | None) -> bool:
    # Defaults, annotations and class attributes execute while definitions are
    # imported. They must not register fixtures through calls/walrus expressions.
    # Function bodies remain candidate implementation, not a Python sandbox.
    return node is None or not any(isinstance(item, (
        ast.Call, ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom,
        ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    )) for item in ast.walk(node))


def _definition_calls(node: ast.AST):
    if isinstance(node, ast.Lambda):
        # Lambda bodies are deferred implementation; their defaults are not.
        for value in [*node.args.defaults, *node.args.kw_defaults]:
            if value is not None:
                yield from _definition_calls(value)
        return
    if isinstance(node, ast.Call):
        yield node
    for child in ast.iter_child_nodes(node):
        yield from _definition_calls(child)


def _config_factory_body_allowed(node: ast.FunctionDef, bindings: dict[str, str],
                                 protected_functions: set[str], bound_names: set[str]) -> bool:
    """Allow editable configuration constructors, not arbitrary import-time code.

    Existing ROCmBench contracts explicitly let the agent tune these helpers.
    Their bodies may build configuration data and branch on protected predicates;
    they cannot acquire the arbitrary-call privilege of protected functions.
    """
    if node.decorator_list:
        return False
    locals_ = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)
               and isinstance(item.ctx, ast.Store)}
    locals_.update(arg.arg for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs])
    locals_.update(arg.arg for arg in [node.args.vararg, node.args.kwarg] if arg)
    for statement in node.body:
        for item in ast.walk(statement):
            if isinstance(item, ast.stmt) and not isinstance(item, (
                    ast.Return, ast.Assign, ast.AnnAssign, ast.If, ast.Pass)):
                if not (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant)):
                    return False
            if isinstance(item, ast.Assign) and not all(isinstance(t, ast.Name) for t in item.targets):
                return False
            if isinstance(item, ast.AnnAssign) and not isinstance(item.target, ast.Name):
                return False
            if isinstance(item, (ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom)):
                return False
        for call in _definition_calls(statement):
            unshadowed = {key: value for key, value in bindings.items() if key not in locals_}
            imported = _qualified_import(call.func, unshadowed)
            protected = (isinstance(call.func, ast.Name)
                         and call.func.id in protected_functions - locals_)
            builtin_range = (isinstance(call.func, ast.Name) and call.func.id == "range"
                             and "range" not in bound_names | locals_)
            if not (protected or builtin_range or imported in {
                    "triton.Config", "triton.cdiv", "triton.next_power_of_2", "itertools.product"}):
                return False
    return True


def _validate_editable_definition(node: ast.AST, bindings: dict[str, str], *,
                                  initial_names: frozenset[str], definition_names: set[str],
                                  configuration_factories: set[str],
                                  new_helper: bool, method: bool = False) -> None:
    name = node.name
    if new_helper and not method and name in _MODULE_INTROSPECTION_HOOKS:
        # Pytest inspects modules through getattr/dir during collection. These
        # bodies therefore run automatically even without a fixture decorator.
        # Ordinary instance methods with the same names remain implementation.
        raise ValueError(f"New module helper {name!r} is an automatic introspection hook")
    if new_helper and (name.startswith(("test", "Test", "pytest_"))
                       or name in _TEST_LIFECYCLE_NAMES):
        raise ValueError(f"New helper {name!r} is a test or test lifecycle hook")
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        qualified = _qualified_import(target, bindings)
        plain_method = (method and isinstance(decorator, ast.Name)
                        and decorator.id in {"staticmethod", "classmethod", "property"}
                        and decorator.id not in bindings
                        and decorator.id not in initial_names | definition_names)
        if qualified not in _COMPILER_DECORATORS and not plain_method:
            raise ValueError(f"Unsupported decorator on editable definition {name!r}: "
                             f"{ast.unparse(decorator)}; only compiler decorators are allowed")
        # A compiler wrapper must not hide a fixture call in its arguments.
        if any((_qualified_import(item, bindings) or "").split(".")[0]
               in {"pytest", "unittest", "pluggy"} for item in ast.walk(decorator)):
            raise ValueError(f"Test environment reference in decorator on {name!r}")
        if isinstance(decorator, ast.Call):
            for value in [*decorator.args, *[kw.value for kw in decorator.keywords]]:
                if any(isinstance(item, ast.NamedExpr) for item in ast.walk(value)):
                    raise ValueError(f"Decorator on {name!r} contains a binding expression")
                for call in _definition_calls(value):
                    imported = _qualified_import(call.func, bindings)
                    config_factory = (isinstance(call.func, ast.Name)
                                      and call.func.id in configuration_factories)
                    builtin_range = (isinstance(call.func, ast.Name) and call.func.id == "range"
                                     and "range" not in initial_names | definition_names)
                    if not (imported in {"triton.Config", "triton.cdiv", "triton.next_power_of_2"}
                            or config_factory or builtin_range):
                        raise ValueError(f"Unapproved definition-time factory in decorator on {name!r}: "
                                         f"{ast.unparse(call.func)}")
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        expressions = [*args.defaults, *args.kw_defaults, node.returns]
        expressions.extend(arg.annotation for arg in [
            *args.posonlyargs, *args.args, *args.kwonlyargs,
            *([args.vararg] if args.vararg else []), *([args.kwarg] if args.kwarg else []),
        ])
        if not all(_passive_expression(expr) for expr in expressions):
            raise ValueError(f"Editable definition {name!r} has executable defaults or annotations")
    else:
        # A new class body executes at import time too. Plain data/method helper
        # classes remain usable; metaclasses, executable bodies and test classes
        # do not become an unguarded extension of the test environment.
        if node.keywords or any(not isinstance(base, ast.Name) or base.id != "object"
                                or "object" in initial_names | definition_names
                                for base in node.bases):
            raise ValueError(f"Editable helper class {name!r} has executable bases/metaclass")
        local_names = {child.name for child in node.body
                       if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        for child in node.body:
            targets = (child.targets if isinstance(child, ast.Assign) else
                       [child.target] if isinstance(child, ast.AnnAssign) else [])
            local_names.update(target.id for target in targets if isinstance(target, ast.Name))
        local_bindings = {key: value for key, value in bindings.items() if key not in local_names}
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                _validate_editable_definition(child, local_bindings, initial_names=initial_names,
                                              definition_names=definition_names | local_names,
                                              configuration_factories=configuration_factories - local_names,
                                              new_helper=True, method=True)
            elif isinstance(child, ast.Pass):
                continue
            elif isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant):
                continue
            elif (isinstance(child, ast.Assign) and all(isinstance(t, ast.Name) for t in child.targets)
                  and _passive_expression(child.value)):
                continue
            elif (isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)
                  and _passive_expression(child.value) and _passive_expression(child.annotation)):
                continue
            else:
                raise ValueError(f"Editable helper class {name!r} has an executable class body")


def _v2_symbol_digest(path: Path, edit: EditScope, initial_names: frozenset[str],
                      entrypoint_symbols: frozenset[str] = frozenset()) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return "invalid-python:" + _sha256(path)
    kept = []
    bindings = _import_bindings(tree)
    definition_names = {node.name for node in tree.body
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    for name in definition_names:
        if (name in edit.symbols or (edit.allow_new_helpers and name not in initial_names)):
            if sum(node.name == name for node in definitions) != 1:
                raise RuntimeError(f"Protected test/harness policy rejected {path.name}: "
                                   f"Multiple definitions of editable binding {name!r}")
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    # These definitions remain in the digest: an initial global alias, class,
    # editable target or newly introduced helper is not a protected factory.
    protected_functions = (set(functions) & initial_names) - set(edit.symbols)
    configuration_factories = set(protected_functions)
    for name in (set(functions) & initial_names & set(edit.symbols)) - entrypoint_symbols:
        if _config_factory_body_allowed(functions[name], bindings, protected_functions,
                                        set(initial_names) | definition_names):
            configuration_factories.add(name)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            new_helper = node.name not in initial_names
            if node.name in edit.symbols or (edit.allow_new_helpers and new_helper):
                try:
                    _validate_editable_definition(node, bindings, initial_names=initial_names,
                                                  definition_names=definition_names,
                                                  configuration_factories=configuration_factories,
                                                  new_helper=new_helper)
                except ValueError as exc:
                    raise RuntimeError(f"Protected test/harness policy rejected {path.name}: {exc}") from exc
                continue
        kept.append(node)
    tree.body = kept
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode("utf-8")).hexdigest()


def _v2_digests(root: Path, spec: TaskSpec, paths: Iterable[str],
                 initial_symbols: dict[str, frozenset[str]]) -> dict[str, str]:
    scopes = {edit.path: edit for edit in spec.candidate.editable if edit.scope == "symbols"}
    selected = set(paths)
    # Keep the existing policy for newly created files matching protected
    # patterns. Use the original TaskSpec, never an agent-modified config.
    selected.update(p.relative_to(root).as_posix() for p in _v2_files(root)
                    if _is_protected_path(p.relative_to(root)))
    result = {}
    for relative in sorted(selected):
        try:
            path = resolve_task_path(root, relative)
        except ValueError as exc:
            raise RuntimeError(f"Protected task path escaped workspace: {relative}") from exc
        if not path.is_file():
            continue
        if relative in scopes:
            if path.suffix != ".py":
                raise ValueError(f"Symbol-scoped protection currently requires Python: {relative}")
            entrypoints = frozenset(entry.symbol for entry in spec.candidate.entrypoints
                                    if entry.file == relative and entry.symbol)
            result[relative] = _v2_symbol_digest(path, scopes[relative],
                                                initial_symbols.get(relative, frozenset()), entrypoints)
        else:
            result[relative] = _sha256(path)
    return result


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


def _protected_digests(root: Path, extra_paths: Iterable[str] = ()) -> dict[str, str]:
    editable_entrypoints = _editable_entrypoint_targets(root)
    digests = {}
    paths = set(_iter_protected_files(root))
    paths.update(
        root / relative for relative in extra_paths if (root / relative).is_file()
    )
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved in editable_entrypoints:
            digest = _sha256_python_harness(path, editable_entrypoints[resolved])
        else:
            digest = _sha256(path)
        digests[str(path.relative_to(root))] = digest
    return digests


def describe_workspace_harness(
    root: Path, *, snapshot: WorkspaceSnapshot | None = None
) -> dict[str, object]:
    """Return trusted, non-secret facts about the active harness guard.

    Task validators run inside the materialized task workspace and cannot inspect
    the framework source tree that applies the guard.  Expose the effective path
    boundary and digest semantics so validators need not infer a whole-file lock
    from a digest. Sessions supply their original snapshot: candidate edits must
    not redefine this description through a changed config or new helper names.
    """

    root = Path(root)
    if snapshot is not None:
        if snapshot.root.resolve() != root.resolve():
            raise ValueError("Harness snapshot belongs to a different workspace")
        if snapshot.task_spec is not None:
            return _describe_v2_snapshot(snapshot)
    config = _task_config(root)
    if config.get("schema_version") == 2:
        return _describe_v2_snapshot(snapshot_workspace_harness(root))
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


def _describe_v2_snapshot(snapshot: WorkspaceSnapshot) -> dict[str, object]:
    """Describe existing digest selection; do not recompute or change enforcement."""
    assert snapshot.task_spec is not None
    scopes = {edit.path: edit for edit in snapshot.task_spec.candidate.editable
              if edit.scope == "symbols"}
    policies = {}
    for path, digest in sorted(snapshot.digests.items()):
        edit = scopes.get(path)
        policies[path] = {
            "digest": digest,
            "digest_mode": ("sha256_python_ast_excluding_editable_symbols" if edit
                            else "sha256_bytes"),
            "editable_symbols": list(edit.symbols) if edit else [],
            "allow_new_helpers": edit.allow_new_helpers if edit else False,
            "initial_top_level_names": sorted(snapshot.initial_symbols.get(path, ())),
            "definition_policy": "compiler_decorators_no_test_hooks_v1" if edit else "byte_protected",
            "allowed_compiler_decorators": sorted(_COMPILER_DECORATORS) if edit else [],
        }
    return {
        "enforced_during_optimization": True,
        "protected_paths": sorted(snapshot.digests),
        "editable_entrypoint_targets": {
            path: list(edit.symbols) for path, edit in scopes.items()
        },
        "protected_path_policies": policies,
    }


def snapshot_workspace_harness(
    root: Path, *, task_root: Path | None = None, task_spec: TaskSpec | None = None
) -> WorkspaceSnapshot:
    """Capture harness digests and, when supplied, immutable task-package inputs.

    Callers running optimization must pass the original task directory. Keep the
    snapshot outside the agent workspace and verify it before final evaluation.
    """

    root = Path(root)
    config = _task_config(root)
    if task_spec is not None or config.get("schema_version") == 2:
        spec = task_spec or TaskSpec.from_mapping(config, task_id="workspace")
        protected = _v2_protected_paths(root, spec)
        initial_symbols = {}
        for edit in spec.candidate.editable:
            if edit.scope == "symbols":
                path = resolve_task_path(root, edit.path, must_exist=True)
                initial_symbols[edit.path] = _top_level_names(path)
        if task_root is not None:
            # Missing shipped inputs cannot disappear from the snapshot merely
            # because a preparation step deleted them.
            protected.update(_v2_protected_paths(Path(task_root), spec))
        missing = sorted(relative for relative in protected if not (root / relative).is_file())
        if missing:
            raise RuntimeError(f"Task inputs missing before agent execution: {missing}")
        digests = _v2_digests(root, spec, protected, initial_symbols)
        return WorkspaceSnapshot(root, digests, spec, initial_symbols)
    task_inputs = (
        _task_input_paths(root, Path(task_root)) if task_root is not None else set()
    )
    missing = sorted(
        relative for relative in task_inputs if not (root / relative).is_file()
    )
    if missing:
        raise RuntimeError(f"Task inputs missing before agent execution: {missing}")
    digests = _protected_digests(root, task_inputs)
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
        # Recheck the original paths even when their names do not match a
        # harness pattern (e.g. session_cases.json or a reference module).
        if snapshot.task_spec is not None:
            return _v2_digests(snapshot.root, snapshot.task_spec, snapshot.digests, snapshot.initial_symbols)
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
