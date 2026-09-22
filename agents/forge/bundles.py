"""Candidate bundles preserve declared paths and keep evaluators independent."""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess

from src.task_spec import EditScope, TaskSpec, resolve_task_path

_TRANSIENT = {".git", "__pycache__", ".pytest_cache", "forge_experiments", ".forge_rewrite"}


def allow_candidate_paths(engine: Path, spec: TaskSpec, *, prefix: str = "") -> None:
    """Expose declared new helpers to Git even inside ignored rewrite scratch."""
    lines = ["", "# Arena Forge candidate declarations"]
    directories = set()
    for scope in spec.candidate.editable:
        path = Path(prefix) / scope.path
        ancestors = list(reversed(path.parents))
        if scope.scope == "tree":
            ancestors.append(path)
        for parent in ancestors:
            if parent == Path(".") or parent in directories:
                continue
            directories.add(parent)
            # Reopen this parent but leave its other, undeclared children hidden.
            lines.extend([f"!/{parent.as_posix()}/", f"/{parent.as_posix()}/*"])
        lines.append(f"!/{path.as_posix()}/**" if scope.scope == "tree" else f"!/{path.as_posix()}")
    lines.extend(["__pycache__/", "*.pyc", ".pytest_cache/"])
    with (engine / ".gitignore").open("a") as stream:
        stream.write("\n".join(lines) + "\n")


def copy_workspace(source: Path, destination: Path) -> None:
    """Copy, without hardlinks, and refuse external symlinks before copying."""
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if any(part in _TRANSIENT for part in relative.parts):
            continue
        if path.is_symlink():
            resolve_task_path(source, relative.as_posix(), must_exist=True)
    shutil.copytree(source, destination, symlinks=True,
                    ignore=shutil.ignore_patterns(*_TRANSIENT))
    # Absolute internal links must point into the copy, not back to its source.
    # Preserve links rather than recursively following cyclic directory links.
    for path in destination.rglob("*"):
        if path.is_symlink() and Path(os.readlink(path)).is_absolute():
            original = source / path.relative_to(destination)
            target = destination / original.resolve(strict=True).relative_to(source)
            path.unlink()
            path.symlink_to(os.path.relpath(target, path.parent))


def protected_paths(spec: TaskSpec) -> set[str]:
    config = spec.to_mapping()
    result = {"config.yaml", "README.md", *config.get("instructions", []), *spec.baseline.source_files}
    if config["evaluation"].get("workloads"):
        result.add(config["evaluation"]["workloads"])
    # Task commands identify their scripts independently of directory naming.
    for action in spec.actions:
        for command in action.commands:
            for argument in command:
                if not argument.startswith("-") and Path(argument).suffix in (".py", ".sh"):
                    result.add(argument)
    return result


def candidate_files(spec: TaskSpec, root: Path, *, required: bool = True) -> dict[str, Path]:
    files = {}
    protected = protected_paths(spec)
    for scope in spec.candidate.editable:
        resolve_task_path(root, scope.path)
        path = root / scope.path
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root and root in parent.parents):
            raise ValueError(f"Candidate symlink is not a deliverable: {scope.path}")
        if scope.scope == "tree":
            if path.exists() and not path.is_dir():
                raise ValueError(f"Candidate tree is not a directory: {scope.path}")
            entries = sorted(path.rglob("*")) if path.exists() else []
        else:
            entries = [path]
            if required and not path.is_file():
                raise ValueError(f"Missing candidate file: {scope.path}")
        for entry in entries:
            relative = entry.relative_to(root).as_posix()
            if any(part in _TRANSIENT for part in Path(relative).parts):
                continue
            if relative in protected and scope.scope != "symbols":
                continue
            resolve_task_path(root, relative, must_exist=entry.exists())
            if entry.is_symlink():
                # A deliverable is a regular file; links could change binding later.
                raise ValueError(f"Candidate symlink is not a deliverable: {relative}")
            if entry.is_file():
                files[relative] = entry
    if required and not files:
        raise ValueError("Candidate bundle contains no implementation files")
    return files


def _check_symbols(original: Path, candidate: Path, scope: EditScope) -> None:
    """Keep colocated non-implementation statements intact, including tests.

    Python imports can change as part of implementing the declared symbols;
    new top-level functions/classes are allowed only with allow_new_helpers.
    This complements the common evaluator's harness guard; it is not a sandbox.
    """
    if candidate.suffix != ".py":
        raise ValueError("Forge symbol-scoped installation currently requires Python")
    before, after = ast.parse(original.read_text()), ast.parse(candidate.read_text())
    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    old_names = {node.name for node in before.body if isinstance(node, definitions)}

    def protected(tree, *, edited):
        nodes = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, definitions):
                if node.name in scope.symbols:
                    continue
                if edited and scope.allow_new_helpers and node.name not in old_names:
                    if node.name.startswith("test_"):
                        raise ValueError("New test functions are not implementation helpers")
                    continue
            nodes.append(ast.dump(node, include_attributes=False))
        return nodes

    if protected(before, edited=False) != protected(after, edited=True):
        raise ValueError(f"Candidate changes protected statements in {scope.path}")


def install_candidate(spec: TaskSpec, source: Path, destination: Path,
                      *, reference: Path | None = None) -> list[dict]:
    """Validate the complete bundle before changing destination; retain layout.

    ``reference`` holds the unedited declaration a symbol scope is compared
    against. It defaults to the destination, which is only the same file when
    the caller hands over a freshly materialized copy of the task package.
    """
    files = candidate_files(spec, source)
    origin = destination if reference is None else reference
    for scope in spec.candidate.editable:
        if scope.scope == "symbols":
            _check_symbols(resolve_task_path(origin, scope.path, must_exist=True),
                           files[scope.path], scope)
    for relative in files:
        resolve_task_path(destination, relative)
    # Propagate declared-tree deletions too, so a removed helper cannot survive
    # from the original template and turn a broken candidate into a valid one.
    stale = candidate_files(spec, destination, required=False)
    for relative, path in stale.items():
        if relative not in files:
            path.unlink()
    records = []
    for relative, path in files.items():
        target = resolve_task_path(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != target:
            shutil.copy2(path, target)
        records.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return records


def committed_candidate(spec: TaskSpec, engine: Path, commit: str, destination: Path,
                        *, prefix: str = "") -> Path:
    """Read the complete selected bundle from Git, not a later working edit."""
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit):
        raise ValueError("Forge result must identify the full selected commit")
    destination.mkdir()
    paths = [str(Path(prefix) / scope.path) for scope in spec.candidate.editable]
    tree = subprocess.run(["git", "ls-tree", "-r", "-z", "--full-tree", commit, "--", *paths],
                          cwd=engine, check=True, capture_output=True).stdout
    protected = protected_paths(spec)
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        relative = Path(os.fsdecode(raw_path)).relative_to(prefix).as_posix() if prefix else os.fsdecode(raw_path)
        scopes = [scope for scope in spec.candidate.editable if scope.contains(relative)]
        if not scopes or (relative in protected and all(scope.scope != "symbols" for scope in scopes)):
            continue
        if kind != "blob" or mode not in ("100644", "100755"):
            raise ValueError(f"Committed candidate is not a regular file: {relative}")
        target = resolve_task_path(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = subprocess.run(["git", "cat-file", "blob", oid], cwd=engine,
                              check=True, capture_output=True).stdout
        target.write_bytes(data)
        target.chmod(0o755 if mode == "100755" else 0o644)
    candidate_files(spec, destination)
    return destination
