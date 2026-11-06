# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path


# Files the framework, the QA auditor or quality_loop's own roles write into a
# run workspace. They are never part of a schema-v3 task package: the package
# names config.yaml, solution/ and _aka_eval/ are reserved for the framework.
GENERATED_NAMES = {
    "config.yaml", "task_result.yaml", "task_result.json",
    "qa_report.yaml", ".qa_complete", "qa_review.draft.json",
    "quality_loop_review.yaml", "quality_loop_repair_review.yaml",
    "_aka_benchmark.py",
}
GENERATED_SUFFIXES = {
    ".o", ".obj", ".so", ".a", ".dll", ".dylib", ".exe", ".hsaco", ".pyc", ".pyo",
}
# Transient trees: never snapshotted, never part of a task package.
TRANSIENT_DIRS = {
    ".git",
    ".pytest_cache",
    ".quality_loop_no_gh",
    ".quality_loop_original_sources",
    ".quality_loop_history",
    ".rocprofv3",
    "__pycache__",
    "build",
}
# Framework-written workspace trees. Snapshots still observe them, so a role
# that edits the candidate or the evaluator is detected, but they are never
# proposed as task-package changes.
WORKSPACE_DIRS = {"_aka_eval", "solution"}
GENERATED_DIRS = TRANSIENT_DIRS | WORKSPACE_DIRS


def _digest(path: Path) -> str:
    if path.is_symlink():
        return "link:" + str(path.readlink())
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def snapshot_tree(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in TRANSIENT_DIRS for part in rel.parts):
            continue
        if path.is_file() or path.is_symlink():
            result[rel.as_posix()] = _digest(path)
    return result


@dataclass(frozen=True)
class TreeChanges:
    added: tuple[str, ...]
    modified: tuple[str, ...]
    deleted: tuple[str, ...]

    @property
    def paths(self) -> tuple[str, ...]:
        return self.added + self.modified + self.deleted

    @property
    def empty(self) -> bool:
        return not self.paths


def diff_trees(before: dict[str, str], after: dict[str, str]) -> TreeChanges:
    return TreeChanges(
        added=tuple(sorted(set(after) - set(before))),
        modified=tuple(sorted(k for k in set(before) & set(after) if before[k] != after[k])),
        deleted=tuple(sorted(set(before) - set(after))),
    )


def is_generated_path(relative: str, *, root: Path | None = None) -> bool:
    """Exclude runtime outputs and framework-written workspace files.

    Arbitrary .pt/.bin files can be task inputs. Inspect ELF magic without
    executing files.
    """
    path = Path(relative)
    if path.name in GENERATED_NAMES or path.suffix in GENERATED_SUFFIXES:
        return True
    if any(part in GENERATED_DIRS for part in path.parts):
        return True
    if root is not None:
        source = root / path
        if source.is_file() and not source.is_symlink():
            with source.open("rb") as stream:
                if stream.read(4) == b"\x7fELF":
                    return True
    return False


def apply_changes(source: Path, destination: Path, changes: TreeChanges) -> None:
    """Apply an already-validated, root-relative change set."""
    source = source.resolve()
    destination = destination.resolve()
    for relative in changes.added + changes.modified:
        src = (source / relative).resolve()
        dst = (destination / relative).resolve()
        if not src.is_relative_to(source) or not dst.is_relative_to(destination):
            raise ValueError(f"unsafe quality_loop path: {relative}")
        if not src.is_file() and not src.is_symlink():
            raise ValueError(f"changed path is not a file: {relative}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_symlink():
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.readlink())
        else:
            shutil.copy2(src, dst)
    for relative in changes.deleted:
        dst = (destination / relative).resolve()
        if not dst.is_relative_to(destination):
            raise ValueError(f"unsafe quality_loop deletion: {relative}")
        if dst.is_file() or dst.is_symlink():
            dst.unlink()
