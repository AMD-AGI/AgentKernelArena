# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.perf_helper_materialization import (
    MARK_END,
    MARK_STARTS,
    ROCMBENCH_HELPER_STUB,
    VLLM_HELPER_STUB_BLOCK,
    replace_marked_region,
)


GENERATED_NAMES = {
    "validation_report.yaml",
    "task_result.yaml",
    "baseline_perf.yaml",
    "optimized_perf.yaml",
    "quality_loop_review.yaml",
    "performance_report.json",
    "compile_report.json",
}
GENERATED_DIRS = {
    ".git",
    ".pytest_cache",
    ".quality_loop_no_gh",
    ".quality_loop_original_sources",
    ".rocprofv3",
    "__pycache__",
    "build",
}


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
        if any(part in GENERATED_DIRS for part in rel.parts):
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


def is_generated_path(relative: str, *, repo_subdir: str | None = None) -> bool:
    path = Path(relative)
    if path.name in GENERATED_NAMES:
        return True
    if any(part in GENERATED_DIRS for part in path.parts):
        return True
    if repo_subdir and path.parts and path.parts[0] == repo_subdir:
        return True
    return False


def is_case_path(relative: str) -> bool:
    path = Path(relative)
    parts = set(path.parts[:-1])
    name = path.name
    return bool(
        parts & {"script", "scripts", "test", "tests"}
        or name.startswith("test_")
        or name.endswith(("_test.py", "_harness.py"))
    )


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


def restore_committed_perf_stubs(task_root: Path) -> None:
    """Undo runtime helper materialization before a task diff is committed."""
    for helper in task_root.rglob("performance_utils_pytest.py"):
        if helper.is_file():
            helper.write_text(ROCMBENCH_HELPER_STUB, encoding="utf-8")
    for runner in task_root.rglob("task_runner.py"):
        if not runner.is_file():
            continue
        current = runner.read_text(encoding="utf-8")
        if not (any(marker in current for marker in MARK_STARTS) or MARK_END in current):
            continue
        replaced = replace_marked_region(current, VLLM_HELPER_STUB_BLOCK)
        if replaced is None:
            raise RuntimeError(f"invalid AKA-GENERATED markers after audit: {runner}")
        runner.write_text(replaced, encoding="utf-8")
