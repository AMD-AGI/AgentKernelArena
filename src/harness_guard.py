# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Workspace integrity guard for task harness files.

Agents should optimize kernels, not the measurement harness.  This module
records a digest snapshot of task-owned harness files before an agent runs and
verifies that those files are unchanged before scoring.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_HARNESS_DIRS = {
    "script",
    "scripts",
    "test",
    "tests",
}
_HARNESS_FILE_NAMES = {
    "config.yaml",
    "config.yml",
    "conftest.py",
    "performance_utils_pytest.py",
}
_HARNESS_FILE_SUFFIXES = (
    "_test.py",
    "_test.cpp",
    "_test.cu",
    "_test.hip",
    "_harness.py",
)


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
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if ".git" in rel.parts or "__pycache__" in rel.parts:
            continue
        if _is_protected_path(rel):
            yield path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_workspace_harness(root: Path) -> WorkspaceSnapshot:
    """Capture digests for task-owned harness and test files."""

    root = Path(root)
    digests = {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(_iter_protected_files(root))
    }
    return WorkspaceSnapshot(root=root, digests=digests)


def verify_workspace_harness(snapshot: WorkspaceSnapshot) -> None:
    """Raise if any protected harness file was modified, deleted, or added."""

    current = {
        str(path.relative_to(snapshot.root)): _sha256(path)
        for path in sorted(_iter_protected_files(snapshot.root))
    }
    before = snapshot.digests
    modified = sorted(
        rel for rel, digest in before.items()
        if rel in current and current[rel] != digest
    )
    deleted = sorted(rel for rel in before if rel not in current)
    added = sorted(rel for rel in current if rel not in before)
    if not (modified or deleted or added):
        return
    details = []
    if modified:
        details.append(f"modified={modified}")
    if deleted:
        details.append(f"deleted={deleted}")
    if added:
        details.append(f"added={added}")
    raise RuntimeError(
        "Protected test/harness files changed during agent execution; "
        "kernel score is rejected to prevent harness hacking: "
        + "; ".join(details)
    )
