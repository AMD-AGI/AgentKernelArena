#!/usr/bin/env python3
"""Install declared head-kernel fixtures from an explicitly selected local source.

This setup utility uses only the standard library. It does not download,
deserialize tensors, generate reference outputs, or run GPU workloads.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Iterator
import uuid


DEFAULT_SUITE_ROOT = Path(__file__).resolve().parents[2] / "tasks/head_kernels"
HASH_KEYS = {
    "reference_io.pt": "reference_io_sha256",
    "timing_geometry.pt": "timing_geometry_sha256",
}
CHUNK_BYTES = 1024 * 1024


class ArtifactError(ValueError):
    """A fixture, manifest, or destination does not satisfy the contract."""


@dataclass(frozen=True)
class Artifact:
    task: str
    path: str
    size_bytes: int
    sha256: str
    mirror_path: str | None = None
    operation_id: str | None = None


@dataclass(frozen=True)
class Manifest:
    artifacts: tuple[Artifact, ...]
    tasks_without_persistent_fixtures: tuple[str, ...]

    @property
    def tasks(self) -> set[str]:
        return {a.task for a in self.artifacts} | set(
            self.tasks_without_persistent_fixtures
        )


def _task_name(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", value):
        raise ArtifactError(f"invalid task name: {value!r}")
    return value


def _relative_parts(value: object) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArtifactError(f"expected a relative POSIX path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or "/".join(path.parts) != value or any(
        part in (".", "..") for part in path.parts
    ):
        raise ArtifactError(f"unsafe relative path: {value!r}")
    return path.parts


def _task_path(value: object) -> str:
    parts = _relative_parts(value)
    for part in parts:
        _task_name(part)
    return "/".join(parts)


def load_manifest(path: Path) -> Manifest:
    """Validate the entire declaration before any fixture is copied."""
    with _directory(path.parent) as parent_fd:
        data = _read_json(parent_fd, path.name)
    version = data.get("schema_version") if isinstance(data, dict) else None
    if type(version) is not int or version not in (1, 2):
        raise ArtifactError("artifact manifest must declare schema_version: 1 or 2")
    rows = data.get("artifacts")
    exempt = data.get("tasks_without_persistent_fixtures", [])
    if not isinstance(rows, list) or not isinstance(exempt, list):
        raise ArtifactError("manifest artifact and task declarations must be lists")
    artifacts = []
    seen = set()
    task_operations = {}
    mirror_declarations = {}

    def task_identity(row: dict) -> tuple[str, str]:
        task = _task_name(row.get("task")) if version == 1 else _task_path(row.get("task"))
        operation = task if version == 1 else _task_name(row.get("operation_id"))
        if task_operations.setdefault(task, operation) != operation:
            raise ArtifactError(f"conflicting operation IDs for task: {task}")
        return task, operation

    for row in rows:
        if not isinstance(row, dict):
            raise ArtifactError("each artifact declaration must be an object")
        task, operation = task_identity(row)
        parts = _relative_parts(row.get("path"))
        if parts[:-1] != (*_relative_parts(task), "ut") or parts[-1] not in HASH_KEYS:
            raise ArtifactError(f"artifact must name {task}/ut/<declared fixture>")
        mirror_path = row["path"] if version == 1 else row.get("mirror_path")
        if _relative_parts(mirror_path) != (operation, "ut", parts[-1]):
            raise ArtifactError(f"mirror_path must name {operation}/ut/{parts[-1]}")
        if row["path"] in seen:
            raise ArtifactError(f"duplicate artifact path: {row['path']}")
        seen.add(row["path"])
        size = row.get("size_bytes")
        digest = row.get("sha256")
        if type(size) is not int or size <= 0:
            raise ArtifactError(f"invalid declared size for {row['path']}")
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ArtifactError(f"missing or invalid SHA-256 for {row['path']}")
        declaration = (size, digest)
        if mirror_declarations.setdefault(mirror_path, declaration) != declaration:
            raise ArtifactError(f"conflicting declarations for mirror_path: {mirror_path}")
        artifacts.append(Artifact(task, row["path"], size, digest, mirror_path, operation))
    without = []
    for row in exempt:
        if not isinstance(row, dict) or not row.get("reason"):
            raise ArtifactError("fixture-free tasks must declare a reason")
        task, _ = task_identity(row)
        if task in without or task in {a.task for a in artifacts}:
            raise ArtifactError(f"conflicting fixture-free task declaration: {task}")
        without.append(task)
    if not artifacts and not without:
        raise ArtifactError("artifact manifest declares no tasks")
    return Manifest(tuple(artifacts), tuple(without))


def select_artifacts(manifest: Manifest, tasks: list[str] | None) -> list[Artifact]:
    requested = set(tasks or manifest.tasks)
    unknown = requested - manifest.tasks
    if unknown:
        raise ArtifactError(f"unknown task(s): {', '.join(sorted(unknown))}")
    return [a for a in manifest.artifacts if a.task in requested]


@contextmanager
def _directory(path: Path) -> Iterator[int]:
    """Open every directory component without following symbolic links."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ArtifactError("fixture provisioning requires POSIX O_NOFOLLOW support")
    absolute = Path(os.path.abspath(path))
    fd = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in absolute.parts[1:]:
            next_fd = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = next_fd
        yield fd
    finally:
        os.close(fd)


@contextmanager
def _parent(root_fd: int, relative: str) -> Iterator[tuple[int, str]]:
    """Keep operations anchored to directory descriptors, including publication."""
    parts = _relative_parts(relative)
    fd = os.dup(root_fd)
    try:
        for component in parts[:-1]:
            next_fd = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = next_fd
        yield fd, parts[-1]
    finally:
        os.close(fd)


@contextmanager
def _regular_file(parent_fd: int, name: str) -> Iterator[int]:
    # NONBLOCK prevents a FIFO from hanging before fstat can reject it.
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ArtifactError(f"fixture or metadata must be a regular file: {name}")
        yield fd
    finally:
        os.close(fd)


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _verify_bytes(fd: int, artifact: Artifact, output_fd: int | None = None) -> None:
    before = os.fstat(fd)
    if before.st_size != artifact.size_bytes:
        raise ArtifactError(
            f"size mismatch for {artifact.path}: expected {artifact.size_bytes}, "
            f"found {before.st_size}"
        )
    digest = hashlib.sha256()
    copied = 0
    while chunk := os.read(fd, CHUNK_BYTES):
        copied += len(chunk)
        if copied > artifact.size_bytes:
            raise ArtifactError(f"fixture grew while reading: {artifact.path}")
        digest.update(chunk)
        if output_fd is not None:
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(output_fd, remaining)
                if written <= 0:
                    raise OSError("fixture write made no progress")
                remaining = remaining[written:]
    if copied != artifact.size_bytes or _identity(before) != _identity(os.fstat(fd)):
        raise ArtifactError(f"fixture changed while reading: {artifact.path}")
    if digest.hexdigest() != artifact.sha256:
        raise ArtifactError(f"SHA-256 mismatch for {artifact.path}")


def _read_json(parent_fd: int, name: str) -> object:
    with _regular_file(parent_fd, name) as fd:
        # Metadata is small JSON; this limit also rejects an accidentally supplied tensor.
        limit = 16 * CHUNK_BYTES
        if os.fstat(fd).st_size > limit:
            raise ArtifactError(f"unexpectedly large JSON metadata: {name}")
        with os.fdopen(os.dup(fd), "rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ArtifactError(f"unexpectedly large JSON metadata: {name}")
        return json.loads(raw)


def _check_metadata(parent_fd: int, artifact: Artifact) -> None:
    metadata = _read_json(parent_fd, "meta.json")
    key = HASH_KEYS[PurePosixPath(artifact.path).name]
    if not isinstance(metadata, dict) or metadata.get(key) != artifact.sha256:
        raise ArtifactError(f"manifest SHA-256 disagrees with {artifact.task}/ut/meta.json:{key}")


def _verify_existing(parent_fd: int, name: str, artifact: Artifact) -> bool:
    try:
        with _regular_file(parent_fd, name) as fd:
            _verify_bytes(fd, artifact)
        return True
    except FileNotFoundError:
        return False


def _publish_noreplace(parent_fd: int, temporary: str, name: str) -> None:
    """Publish the private copied inode atomically, without replacing a target.

    Only the newly written temporary output is linked. The shared source inode
    is never linked, and removing the private alias leaves a single-link copy.
    """
    os.link(
        temporary, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
        follow_symlinks=False,
    )
    os.unlink(temporary, dir_fd=parent_fd)


def _install(source_fd: int, parent_fd: int, name: str, artifact: Artifact) -> None:
    temporary = f".{name}.{uuid.uuid4().hex}.partial"
    fd = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600, dir_fd=parent_fd,
    )
    try:
        _verify_bytes(source_fd, artifact, fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
        _publish_noreplace(parent_fd, temporary, name)
        os.fsync(parent_fd)
    finally:
        os.close(fd)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass  # Successful publication removed the private temporary alias.


def prepare_artifacts(
    artifacts: list[Artifact],
    suite_root: Path,
    *,
    mirror: Path | None = None,
    cache: Path | None = None,
    verify_only: bool = False,
) -> list[tuple[str, str]]:
    """Verify or install fixtures; return (relative path, action) pairs.

    Version-2 mirror_path preserves the original flat operation namespace;
    version 1 reads <task>/ut/<filename>. Cache paths are SHA-256 hex digests.
    Existing task and ut directories are required. No existing file is modified.
    Atomicity is per file: completed fixtures remain installed if a later one fails.
    """
    if mirror is not None and cache is not None:
        raise ArtifactError("choose exactly one source: mirror or cache")
    if verify_only and (mirror is not None or cache is not None):
        raise ArtifactError("verification does not accept an artifact source")
    if not verify_only and mirror is None and cache is None:
        raise ArtifactError("installation requires an explicit mirror or cache")
    results = []
    with _directory(suite_root) as root_fd:
        for artifact in artifacts:
            with _parent(root_fd, artifact.path) as (parent_fd, name):
                _check_metadata(parent_fd, artifact)
                if _verify_existing(parent_fd, name, artifact):
                    results.append((artifact.path, "verified existing"))
                    continue
                if verify_only:
                    raise ArtifactError(f"missing fixture: {artifact.path}; provision it before a run")
                source = mirror if mirror is not None else cache
                relative = (artifact.mirror_path or artifact.path) if mirror is not None else artifact.sha256
                with _directory(source) as source_root_fd:
                    with _parent(source_root_fd, relative) as (source_parent_fd, source_name):
                        with _regular_file(source_parent_fd, source_name) as source_fd:
                            _install(source_fd, parent_fd, name, artifact)
                results.append((artifact.path, "installed"))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE_ROOT)
    parser.add_argument("--manifest", type=Path, help="default: <suite-root>/artifacts.json")
    parser.add_argument("--task", action="append", help="task path relative to suite root; repeat to select a subset")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--mirror", type=Path, help="local directory containing the manifest's mirror_path files")
    operation.add_argument("--cache", type=Path, help="local directory containing files named by SHA-256")
    operation.add_argument("--verify", action="store_true", help="verify installed fixtures without writing")
    operation.add_argument("--list", action="store_true", help="list declarations without reading tensors")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest or args.suite_root / "artifacts.json")
        artifacts = select_artifacts(manifest, args.task)
        if args.list:
            for artifact in artifacts:
                print(f"{artifact.size_bytes}\t{artifact.sha256}\t{artifact.path}")
        else:
            results = prepare_artifacts(
                artifacts, args.suite_root, mirror=args.mirror, cache=args.cache,
                verify_only=args.verify,
            )
            for path, action in results:
                print(f"{action}: {path}")
        print(f"{len(artifacts)} fixtures; {sum(a.size_bytes for a in artifacts)} bytes declared")
        selected = set(args.task or manifest.tasks)
        for task in manifest.tasks_without_persistent_fixtures:
            if task in selected:
                print(f"no persistent fixture required: {task}")
    except (ArtifactError, OSError, json.JSONDecodeError) as exc:
        print(f"artifact preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
