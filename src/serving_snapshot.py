"""Bounded submission reads anchored to open directories, without following links."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path, PurePosixPath
import stat


def _parts(path: str | Path) -> tuple[str, ...]:
    value = PurePosixPath(path)
    if value.is_absolute() or not value.parts or '..' in value.parts:
        raise ValueError('Submission paths must be relative and contained')
    return value.parts


@contextmanager
def directory_at(parent: int, path: str | Path):
    """Keep each ancestor open while descending; never resolve then reopen."""
    fd = os.dup(parent)
    try:
        for part in _parts(path):
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    except OSError as exc:
        raise ValueError('Submission directory is unavailable or contains a symlink') from exc
    finally:
        os.close(fd)


@contextmanager
def workspace_at(root: Path, relative: Path):
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        with directory_at(fd, relative) as workspace:
            yield workspace
    finally:
        os.close(fd)


def _read_file(parent: int, name: str, limit: int) -> bytes:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=parent)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            raise ValueError('Submission must contain bounded regular files without links')
        chunks, size = [], 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, limit + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                raise ValueError('Submission exceeds the declared source budget')
        after = os.fstat(fd)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if identity(before) != identity(after) or size != before.st_size:
            raise ValueError('Submission changed while being snapshotted; retry from stable files')
        return b''.join(chunks)
    finally:
        os.close(fd)


def read_file_at(workspace: int, path: str | Path, *, max_bytes: int) -> bytes:
    parts = _parts(path)
    try:
        if len(parts) == 1:
            return _read_file(workspace, parts[0], max_bytes)
        with directory_at(workspace, '/'.join(parts[:-1])) as parent:
            return _read_file(parent, parts[-1], max_bytes)
    except OSError as exc:
        raise ValueError('Submission file is unavailable or contains a symlink') from exc


def read_tree_at(workspace: int, path: str | Path, *, max_bytes: int,
                 max_files: int = 256) -> dict[str, bytes]:
    files, size, entries = {}, 0, 0

    def visit(parent, prefix, depth):
        nonlocal size, entries
        if depth > 32:
            raise ValueError('Submission directory nesting exceeds the source budget')
        with os.scandir(parent) as children:
            for entry in children:
                entries += 1
                if entries > 4096:
                    raise ValueError('Submission directory entries exceed the source budget')
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    raise ValueError('Submission must not contain symlinks')
                if entry.name == '__pycache__':
                    continue
                name = prefix + entry.name
                if stat.S_ISDIR(info.st_mode):
                    with directory_at(parent, entry.name) as child:
                        visit(child, name + '/', depth + 1)
                else:
                    if len(files) >= max_files:
                        raise ValueError('Submission exceeds the declared file budget')
                    data = _read_file(parent, entry.name, max_bytes - size)
                    files[name] = data
                    size += len(data)

    with directory_at(workspace, path) as root:
        visit(root, '', 0)
    if not files:
        raise ValueError('Kernel bundle is empty')
    return files
