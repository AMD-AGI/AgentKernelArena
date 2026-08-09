# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Utilities for evaluator: command execution and file I/O.
"""
import ast
import hashlib
import json
import os
import shutil
import stat
import subprocess
import logging
import yaml
import shlex
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List
from .testcases import TestCaseResult
from .runtime_env import PYTHON_ENV_VAR, build_subprocess_env


FORMAL_SOURCE_ANTI_TAMPER_SCHEMA = "aka.formal-source-anti-tamper/v1"
FORMAL_SOURCE_ANTI_TAMPER_POLICY = "ast_static_side_effect_guard_v1"
FORMAL_SOURCE_MANIFEST_SCHEMA = "aka.formal-source-manifest/v1"
_FORMAL_SOURCE_MAX_BYTES = 16 * 1024 * 1024
_FORMAL_ALLOWED_IMPORT_ROOTS = ("math", "torch", "triton")
_FORMAL_FORBIDDEN_NAMESPACES = (
    "builtins",
    "ctypes",
    "importlib",
    "inspect",
    "marshal",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "socket",
    "subprocess",
    "sys",
)
_FORMAL_FORBIDDEN_DYNAMIC_CALLS = (
    "__import__",
    "compile",
    "delattr",
    "eval",
    "exec",
    "globals",
    "locals",
    "setattr",
    "vars",
)
_FORMAL_PROTECTED_STATE_MUTATING_CALLS = (
    "cuda.empty_cache",
    "cuda.manual_seed",
    "cuda.manual_seed_all",
    "cuda.set_device",
    "cuda.set_per_process_memory_fraction",
    "cuda.set_rng_state",
    "cuda.set_rng_state_all",
    "load",
    "manual_seed",
    "save",
    "seed",
    "set_default_device",
    "set_default_dtype",
    "set_deterministic_debug_mode",
    "set_num_interop_threads",
    "set_num_threads",
    "set_rng_state",
    "use_deterministic_algorithms",
)
FORMAL_SOURCE_ANTI_TAMPER_RULES = {
    "allowed_import_roots": list(_FORMAL_ALLOWED_IMPORT_ROOTS),
    "forbidden_dynamic_calls": list(_FORMAL_FORBIDDEN_DYNAMIC_CALLS),
    "forbidden_namespace_roots": list(_FORMAL_FORBIDDEN_NAMESPACES),
    "forbid_dunder_name_or_attribute_access": True,
    "forbid_dynamic_dunder_getattr": True,
    "forbid_dynamic_protected_namespace_getattr": True,
    "forbid_forbidden_namespace_bridges": True,
    "forbid_protected_attribute_assignment_or_deletion": [
        "builtins",
        "sys",
        "torch",
        "triton",
    ],
    "protected_state_mutating_calls": list(
        _FORMAL_PROTECTED_STATE_MUTATING_CALLS
    ),
    "max_source_bytes": _FORMAL_SOURCE_MAX_BYTES,
    "source_files_must_be_workspace_regular_single_link_files": True,
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


FORMAL_SOURCE_ANTI_TAMPER_RULES_SHA256 = hashlib.sha256(
    _canonical_json_bytes(FORMAL_SOURCE_ANTI_TAMPER_RULES)
).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Return the digest used for formal anti-tamper evidence bindings."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _anti_tamper_violation(
    rule: str,
    detail: str,
    node: Optional[ast.AST] = None,
) -> Dict[str, Any]:
    return {
        "rule": rule,
        "line": int(getattr(node, "lineno", 0) or 0),
        "column": int(getattr(node, "col_offset", 0) or 0),
        "detail": detail,
    }


def _attribute_root(node: ast.AST) -> Optional[str]:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _attribute_chain(node: ast.AST) -> Optional[List[str]]:
    components: List[str] = []
    while isinstance(node, ast.Attribute):
        components.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    components.append(node.id)
    return list(reversed(components))


def _is_dunder_name(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


class _FormalSourceAntiTamperVisitor(ast.NodeVisitor):
    """Conservative static side-effect guard for formal candidate Python source.

    This is a deterministic defense-in-depth filter, not a Python sandbox. The
    evaluator must still execute untrusted candidates across an OS/container
    isolation boundary.
    """

    def __init__(self) -> None:
        self.violations: List[Dict[str, Any]] = []
        self.protected_aliases = {"builtins", "sys", "torch", "triton", "tl"}
        self.torch_aliases = {"torch"}
        self.protected_mutator_aliases = set()
        self.forbidden_aliases = set(_FORMAL_FORBIDDEN_NAMESPACES)

    def _add(self, rule: str, detail: str, node: ast.AST) -> None:
        self.violations.append(_anti_tamper_violation(rule, detail, node))

    def _expression_is_protected(self, expression: ast.AST) -> bool:
        root = _attribute_root(expression)
        return root in self.protected_aliases

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            bound_name = alias.asname or root
            if root not in _FORMAL_ALLOWED_IMPORT_ROOTS:
                self._add(
                    "non_allowlisted_import",
                    f"import root {root!r} is not allowed in formal source",
                    node,
                )
            if root in _FORMAL_FORBIDDEN_NAMESPACES:
                self.forbidden_aliases.add(bound_name)
                self._add(
                    "forbidden_namespace_import",
                    f"import of namespace {root!r} is forbidden",
                    node,
                )
            if root in {"torch", "triton", "builtins", "sys"}:
                self.protected_aliases.add(bound_name)
            if root == "torch":
                self.torch_aliases.add(bound_name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".", 1)[0]
        if node.level or root not in _FORMAL_ALLOWED_IMPORT_ROOTS:
            self._add(
                "non_allowlisted_import",
                f"import root {root or '<relative>'!r} is not allowed in formal source",
                node,
            )
        if root in _FORMAL_FORBIDDEN_NAMESPACES:
            self._add(
                "forbidden_namespace_import",
                f"import from namespace {root!r} is forbidden",
                node,
            )
            for alias in node.names:
                self.forbidden_aliases.add(alias.asname or alias.name)
        if root in {"torch", "triton", "builtins", "sys"}:
            for alias in node.names:
                self.protected_aliases.add(alias.asname or alias.name)
                if root == "torch":
                    bound_name = alias.asname or alias.name
                    self.torch_aliases.add(bound_name)
                    if alias.name in _FORMAL_PROTECTED_STATE_MUTATING_CALLS:
                        self.protected_mutator_aliases.add(bound_name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._expression_is_protected(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.protected_aliases.add(target.id)
                    if _attribute_root(node.value) in self.torch_aliases:
                        self.torch_aliases.add(target.id)
        if self._expression_is_protected_mutator(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.protected_mutator_aliases.add(target.id)
        self._check_mutation_targets(node.targets, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and self._expression_is_protected(node.value):
            if isinstance(node.target, ast.Name):
                self.protected_aliases.add(node.target.id)
                if _attribute_root(node.value) in self.torch_aliases:
                    self.torch_aliases.add(node.target.id)
        if (
            node.value is not None
            and self._expression_is_protected_mutator(node.value)
            and isinstance(node.target, ast.Name)
        ):
            self.protected_mutator_aliases.add(node.target.id)
        self._check_mutation_targets([node.target], node)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_mutation_targets([node.target], node)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        self._check_mutation_targets(node.targets, node)
        self.generic_visit(node)

    def _check_mutation_targets(
        self,
        targets: List[ast.expr],
        node: ast.AST,
    ) -> None:
        for target in targets:
            for candidate in ast.walk(target):
                if not isinstance(candidate, ast.Attribute):
                    continue
                root = _attribute_root(candidate)
                if root in self.protected_aliases:
                    self._add(
                        "protected_attribute_mutation",
                        f"assignment or deletion through protected namespace {root!r} is forbidden",
                        node,
                    )
                    break

    def _expression_is_protected_mutator(self, expression: ast.AST) -> bool:
        chain = _attribute_chain(expression)
        if not chain or chain[0] not in self.torch_aliases:
            return False
        path = ".".join(chain[1:])
        return path in _FORMAL_PROTECTED_STATE_MUTATING_CALLS

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            if node.id in self.forbidden_aliases:
                self._add(
                    "forbidden_namespace_access",
                    f"access to namespace {node.id!r} is forbidden",
                    node,
                )
            if node.id in _FORMAL_FORBIDDEN_DYNAMIC_CALLS:
                self._add(
                    "forbidden_dynamic_namespace_primitive",
                    f"dynamic namespace primitive {node.id!r} is forbidden",
                    node,
                )
            if _is_dunder_name(node.id):
                self._add(
                    "dunder_namespace_access",
                    f"dunder name access {node.id!r} is forbidden",
                    node,
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        root = _attribute_root(node)
        if root in self.forbidden_aliases:
            self._add(
                "forbidden_namespace_access",
                f"attribute access through namespace {root!r} is forbidden",
                node,
            )
        chain = _attribute_chain(node)
        if (
            chain
            and chain[0] in self.protected_aliases
            and any(
                component in _FORMAL_FORBIDDEN_NAMESPACES
                for component in chain[1:]
            )
        ):
            self._add(
                "forbidden_namespace_bridge",
                "protected namespace attribute chain enters a forbidden namespace",
                node,
            )
        if _is_dunder_name(node.attr):
            self._add(
                "dunder_namespace_access",
                f"dunder attribute access {node.attr!r} is forbidden",
                node,
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in _FORMAL_FORBIDDEN_DYNAMIC_CALLS:
                self._add(
                    "forbidden_dynamic_namespace_call",
                    f"call to {node.func.id!r} is forbidden",
                    node,
                )
            if node.func.id in self.protected_mutator_aliases:
                self._add(
                    "protected_state_mutating_call",
                    f"call through protected mutator alias {node.func.id!r} is forbidden",
                    node,
                )
            if node.func.id == "getattr" and len(node.args) >= 2:
                if self._expression_is_protected(node.args[0]):
                    self._add(
                        "dynamic_protected_namespace_access",
                        "dynamic getattr through a protected namespace is forbidden",
                        node,
                    )
                attribute = node.args[1]
                if (
                    isinstance(attribute, ast.Constant)
                    and isinstance(attribute.value, str)
                    and _is_dunder_name(attribute.value)
                ):
                    self._add(
                        "dynamic_dunder_access",
                        f"dynamic dunder access {attribute.value!r} is forbidden",
                        node,
                    )
        elif isinstance(node.func, ast.Attribute):
            chain = _attribute_chain(node.func)
            if chain and chain[0] in self.torch_aliases:
                path = ".".join(chain[1:])
                if path in _FORMAL_PROTECTED_STATE_MUTATING_CALLS:
                    self._add(
                        "protected_state_mutating_call",
                        f"protected global-state mutator {path!r} is forbidden",
                        node,
                    )
        self.generic_visit(node)


def _safe_formal_source_bytes(
    workspace: Path,
    configured_path: str,
) -> Tuple[Optional[str], Optional[bytes], List[Dict[str, Any]]]:
    violations: List[Dict[str, Any]] = []
    try:
        root = workspace.resolve(strict=True)
    except OSError as error:
        return None, None, [
            _anti_tamper_violation(
                "unsafe_source_path", f"workspace cannot be resolved: {error}"
            )
        ]
    lexical = Path(configured_path)
    candidate = lexical if lexical.is_absolute() else root / lexical
    try:
        candidate = Path(os.path.abspath(os.fspath(candidate)))
        lexical_relative = candidate.relative_to(root)
        resolved = candidate.resolve(strict=False)
        relative = resolved.relative_to(root).as_posix()
    except (OSError, ValueError) as error:
        return None, None, [
            _anti_tamper_violation(
                "unsafe_source_path",
                f"source path is outside or cannot be resolved within workspace: {error}",
            )
        ]
    current = root
    try:
        for component in lexical_relative.parts:
            current = current / component
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise OSError(f"symlink component is forbidden: {current}")
        metadata_before = candidate.lstat()
        if not stat.S_ISREG(metadata_before.st_mode) or metadata_before.st_nlink != 1:
            raise OSError("source must be a regular single-link file")
        if metadata_before.st_size > _FORMAL_SOURCE_MAX_BYTES:
            raise OSError(
                f"source exceeds {_FORMAL_SOURCE_MAX_BYTES} byte formal limit"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate, flags)
        try:
            descriptor_metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(descriptor_metadata.st_mode)
                or descriptor_metadata.st_nlink != 1
                or descriptor_metadata.st_dev != metadata_before.st_dev
                or descriptor_metadata.st_ino != metadata_before.st_ino
            ):
                raise OSError("source descriptor identity is unsafe")
            chunks: List[bytes] = []
            remaining = _FORMAL_SOURCE_MAX_BYTES + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            if len(raw) > _FORMAL_SOURCE_MAX_BYTES:
                raise OSError("source grew beyond the formal byte limit while reading")
            descriptor_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        metadata_after = candidate.lstat()
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_nlink")
        if any(
            getattr(metadata_before, field) != getattr(metadata_after, field)
            or getattr(metadata_before, field) != getattr(descriptor_after, field)
            for field in stable_fields
        ):
            raise OSError("source changed while anti-tamper evidence was captured")
        if len(raw) != metadata_after.st_size:
            raise OSError("source byte count differs from descriptor metadata")
    except OSError as error:
        violations.append(
            _anti_tamper_violation("unsafe_source_file", f"{relative}: {error}")
        )
        return relative, None, violations
    return relative, raw, violations


def inspect_formal_source_anti_tamper(
    workspace: Path,
    task_config: Dict[str, Any],
    *,
    expected_source_manifest_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Build deterministic static anti-tamper evidence for formal evaluation.

    The report intentionally contains no source text. It binds the exact source
    bytes and records AST-policy violations. Passing this check does not make
    Python source safe to execute in the evaluator process.
    """
    configured_paths = _string_list(task_config.get("source_file_path"))
    file_reports: List[Dict[str, Any]] = []
    report_violations: List[Dict[str, Any]] = []
    seen_paths = set()
    if not configured_paths:
        report_violations.append(
            _anti_tamper_violation(
                "missing_source_file_path",
                "formal evaluation requires at least one configured source file",
            )
        )
    for configured_path in configured_paths:
        relative, raw, violations = _safe_formal_source_bytes(
            Path(workspace), configured_path
        )
        display_path = relative or str(configured_path)
        if display_path in seen_paths:
            violations.append(
                _anti_tamper_violation(
                    "duplicate_source_path",
                    f"source path {display_path!r} is configured more than once",
                )
            )
        seen_paths.add(display_path)
        language = "python" if Path(display_path).suffix == ".py" else "other"
        digest = hashlib.sha256(raw).hexdigest() if raw is not None else None
        size_bytes = len(raw) if raw is not None else None
        if raw is not None and language == "python":
            try:
                source_text = raw.decode("utf-8")
                tree = ast.parse(source_text, filename=display_path)
            except (UnicodeError, SyntaxError) as error:
                violations.append(
                    _anti_tamper_violation(
                        "invalid_python_source", f"cannot parse source: {error}"
                    )
                )
            else:
                visitor = _FormalSourceAntiTamperVisitor()
                visitor.visit(tree)
                violations.extend(visitor.violations)
        violations.sort(
            key=lambda item: (
                item["line"], item["column"], item["rule"], item["detail"]
            )
        )
        file_reports.append(
            {
                "path": display_path,
                "sha256": digest,
                "size_bytes": size_bytes,
                "language": language,
                "status": (
                    "FAIL"
                    if violations
                    else "PASS"
                    if language == "python"
                    else "NOT_APPLICABLE"
                ),
                "violations": violations,
            }
        )
    file_reports.sort(key=lambda item: item["path"])
    manifest_material = {
        "schema": FORMAL_SOURCE_MANIFEST_SCHEMA,
        "files": [
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
            }
            for item in file_reports
        ],
    }
    source_manifest_sha256 = canonical_json_sha256(manifest_material)
    if (
        expected_source_manifest_sha256 is not None
        and source_manifest_sha256 != expected_source_manifest_sha256
    ):
        report_violations.append(
            _anti_tamper_violation(
                "source_manifest_mismatch",
                "candidate source changed after the formal anti-tamper anchor",
            )
        )
    report_violations.sort(
        key=lambda item: (
            item["line"], item["column"], item["rule"], item["detail"]
        )
    )
    failed = bool(report_violations) or any(
        item["status"] == "FAIL" for item in file_reports
    )
    return {
        "schema": FORMAL_SOURCE_ANTI_TAMPER_SCHEMA,
        "policy": FORMAL_SOURCE_ANTI_TAMPER_POLICY,
        "rules_sha256": FORMAL_SOURCE_ANTI_TAMPER_RULES_SHA256,
        "verdict": "FAIL" if failed else "PASS",
        "source_manifest_sha256": source_manifest_sha256,
        "expected_source_manifest_sha256": expected_source_manifest_sha256,
        "files": file_reports,
        "violations": report_violations,
    }


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str)]
    return []


def _is_docstring_statement(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _is_not_implemented_exception(expression: Optional[ast.expr]) -> bool:
    if isinstance(expression, ast.Call):
        expression = expression.func
    if isinstance(expression, ast.Name):
        return expression.id == "NotImplementedError"
    return (
        isinstance(expression, ast.Attribute)
        and expression.attr == "NotImplementedError"
        and isinstance(expression.value, ast.Name)
        and expression.value.id == "builtins"
    )


def _is_unimplemented_target_stub(function: ast.AST) -> bool:
    """Match only a no-op body followed by a direct NotImplementedError raise."""
    body = list(getattr(function, "body", []))
    meaningful_statements: List[ast.stmt] = []
    for index, statement in enumerate(body):
        if index == 0 and _is_docstring_statement(statement):
            continue
        if isinstance(statement, ast.Pass):
            continue
        meaningful_statements.append(statement)

    return (
        len(meaningful_statements) == 1
        and isinstance(meaningful_statements[0], ast.Raise)
        and _is_not_implemented_exception(meaningful_statements[0].exc)
    )


def inspect_target_definitions(
    workspace: Path,
    task_config: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """Return missing and unimplemented declared top-level Python targets.

    This intentionally does not walk into function bodies.  An implemented
    target may use a conditional ``NotImplementedError`` for an unsupported
    shape without being classified as an unimplemented submission.  The task
    contract requires a Python ``def`` for each target; assignment aliases are
    not treated as target definitions.
    """
    target_names = set(_string_list(task_config.get("target_kernel_functions")))
    if not target_names:
        return [], []

    found_names = set()
    stub_names = set()
    for configured_path in _string_list(task_config.get("source_file_path")):
        source_path = Path(configured_path)
        if not source_path.is_absolute():
            source_path = Path(workspace) / source_path
        if not source_path.is_file() or source_path.suffix != ".py":
            continue
        try:
            module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        except (OSError, UnicodeError, SyntaxError):
            # Compilation reports missing, unreadable, and invalid source files;
            # this guard is deliberately limited to recognized starter stubs.
            continue

        # The last top-level definition is the one bound by the module at run
        # time.  Nested methods/functions are intentionally excluded.
        definitions = {}
        for statement in module.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions[statement.name] = statement
        for target_name in target_names:
            function = definitions.get(target_name)
            if function is None:
                continue
            found_names.add(target_name)
            if _is_unimplemented_target_stub(function):
                stub_names.add(target_name)

    return sorted(target_names - found_names), sorted(stub_names)


def find_unimplemented_target_stubs(
    workspace: Path,
    task_config: Dict[str, Any],
) -> List[str]:
    """Compatibility helper returning only declared starter stubs."""
    _, stub_names = inspect_target_definitions(workspace, task_config)
    return stub_names


def _replace_leading_token(command: str, token: str, replacement: str) -> str:
    leading_len = len(command) - len(command.lstrip())
    leading = command[:leading_len]
    stripped = command[leading_len:]
    if stripped == token or stripped.startswith(f"{token} "):
        return f"{leading}{replacement}{stripped[len(token):]}"
    return command


def normalize_python_command(command: str, python_path: str) -> str:
    """Route bare Python tooling commands through the selected interpreter."""
    normalized = command
    normalized = _replace_leading_token(normalized, "python3", python_path)
    normalized = _replace_leading_token(normalized, "python", python_path)
    normalized = _replace_leading_token(normalized, "pytest", f"{python_path} -m pytest")
    return normalized


def run_command(
    command: str,
    workspace: Path,
    timeout: float = 300,
    logger: Optional[logging.Logger] = None,
    docker_container: Optional[str] = None,
    extra_env: Optional[dict] = None,
) -> Tuple[bool, str, str]:
    """
    Run a shell command in the workspace directory.

    When ``docker_container`` is provided the command is executed inside the
    named Docker container via ``docker exec``.  The workspace path is
    assumed to be identical on host and inside the container (bind-mounted).

    Args:
        command: Shell command to execute
        workspace: Working directory
        timeout: Command timeout in seconds
        logger: Optional logger for output
        docker_container: If set, run the command inside this Docker container
        extra_env: Optional env vars applied to THIS subprocess only (merged over
            the inherited env). Used e.g. to scope AITER_REBUILD=1 to a single
            build step without leaking it into the parent process / later tasks.

    Returns:
        Tuple of (success: bool, stdout: str, stderr: str)
    """
    log = logger or logging.getLogger(__name__)

    try:
        env = build_subprocess_env()
        if extra_env:
            env.update({str(k): str(v) for k, v in extra_env.items()})
        if docker_container:
            # When running inside a Docker container we can't rewrite "python3" to
            # the host interpreter path — skip normalize_python_command and wrap
            # the original command in `docker exec` instead.
            escaped = command.replace("'", "'\\''")
            abs_workspace = Path(workspace).resolve()
            command_to_run = (
                f"docker exec -w {abs_workspace} {docker_container} "
                f"bash -c '{escaped}'"
            )
            log.info(f"Running in Docker [{docker_container}]: {command_to_run[:200]}")
        else:
            python_path = env.get(PYTHON_ENV_VAR)
            quoted_python = shlex.quote(python_path) if python_path else None
            command_to_run = normalize_python_command(command, quoted_python) if quoted_python else command
            log.info(f"Running command: {command_to_run}")
            if command_to_run != command:
                log.info(f"Original command: {command}")

        log.info(f"Working directory: {workspace}")

        result = subprocess.run(
            command_to_run,
            shell=True,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        if result.returncode == 0:
            log.info(f"Command succeeded")
            if result.stdout:
                log.debug(f"STDOUT: {result.stdout[:500]}")  # Log first 500 chars
            return True, result.stdout, result.stderr
        else:
            log.warning(f"Command failed with exit code {result.returncode}")
            if result.stderr:
                log.warning(f"STDERR: {result.stderr[:500]}")
            return False, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        log.error(f"Command timed out after {timeout} seconds")
        return False, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        log.error(f"Command execution failed: {e}")
        return False, "", str(e)


def checkout_aiter(
    commit: str,
    docker_container: str,
    aiter_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Pinned-commit checkout of the aiter repo.

    aiter_path resolution order:
      1. explicit ``aiter_path`` argument
      2. ``AKA_AITER_PATH`` env var
    No baked-in default — if neither is provided, returns False with a clear
    error message. (Reviewer note: previously hardcoded to /sgl-workspace/aiter.)
    """
    if aiter_path is None:
        aiter_path = os.environ.get("AKA_AITER_PATH")
    if not aiter_path:
        log = logger or logging.getLogger(__name__)
        log.error(
            "aiter path is not configured: pass aiter_path explicitly or set "
            "the AKA_AITER_PATH env var to the absolute path of the aiter repo."
        )
        return False

    log = logger or logging.getLogger(__name__)

    # Detect if we're already inside the container (no docker CLI available)
    inside_container = not shutil.which("docker")

    if not inside_container:
        # Verify container is running
        check = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", docker_container],
            capture_output=True, text=True,
        )
        if check.returncode != 0 or "true" not in check.stdout.lower():
            log.error(f"Docker container '{docker_container}' is not running")
            return False

    # Checkout the requested commit.
    # Always reset + clean to avoid stale files conflicting with new commit
    # (e.g. rope.py file coexisting with rope/ directory after branch switch).
    # Also clear __pycache__ to avoid stale bytecode.
    checkout_cmd = (
        f"cd {aiter_path} && git reset --hard && git clean -fd"
        f" && git checkout --quiet {commit}"
        f" && find . -name __pycache__ -type d -exec rm -rf {{}} + 2>/dev/null; true"
    )
    if inside_container:
        result = subprocess.run(
            ["bash", "-c", checkout_cmd],
            capture_output=True, text=True, timeout=60,
        )
    else:
        result = subprocess.run(
            ["docker", "exec", docker_container, "bash", "-c", checkout_cmd],
            capture_output=True, text=True, timeout=60,
        )
    if result.returncode != 0:
        log.warning(f"git checkout {commit[:12]} failed, trying hard reset")
        reset_cmd = f"cd {aiter_path} && git reset --hard && git clean -fd && git checkout {commit}"
        if inside_container:
            result = subprocess.run(
                ["bash", "-c", reset_cmd],
                capture_output=True, text=True, timeout=60,
            )
        else:
            result = subprocess.run(
                ["docker", "exec", docker_container, "bash", "-c", reset_cmd],
                capture_output=True, text=True, timeout=60,
            )
        if result.returncode != 0:
            log.error(f"Failed to checkout aiter {commit[:12]}: {result.stderr[:300]}")
            return False

    log.info(f"aiter checked out to {commit[:12]} in {docker_container}")
    return True
