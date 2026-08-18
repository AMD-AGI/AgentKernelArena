"""Identity and containment helpers for evaluator-selected native code objects."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

from ..contracts import ToolContext


def resolve_workspace_file(context: ToolContext, key: str) -> Path:
    """Resolve a required regular file without allowing workspace escape."""

    raw = context.options.get(key)
    if raw is None or not str(raw).strip():
        raise ValueError(f"missing required tool option: {key}")
    workspace = Path(context.workspace).resolve(strict=True)
    configured = Path(str(raw))
    candidate = (
        configured.resolve(strict=True)
        if configured.is_absolute()
        else (workspace / configured).resolve(strict=True)
    )
    try:
        candidate.relative_to(workspace)
    except ValueError as error:
        raise ValueError(f"{key} must resolve below the selected task workspace") from error
    if not candidate.is_file():
        raise ValueError(f"{key} is not a regular file: {candidate}")
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fnv1a64_file(path: Path) -> str:
    """Return rocJITsu's stable full-code-object identity."""

    value = 14695981039346656037
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            for byte in chunk:
                value ^= byte
                value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{value:016x}"


def command_references_file(
    command: Sequence[str], path: Path, workspace: Path
) -> bool:
    """Require the native launcher argv to name the selected code object."""

    expected = path.resolve(strict=True)
    root = workspace.resolve(strict=True)
    for raw in command:
        values = [str(raw)]
        if "=" in str(raw):
            values.append(str(raw).split("=", 1)[1])
        for value in values:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = root / candidate
            try:
                if candidate.resolve(strict=False) == expected:
                    return True
            except OSError:
                continue
    return False


__all__ = [
    "command_references_file",
    "fnv1a64_file",
    "resolve_workspace_file",
    "sha256_file",
]
