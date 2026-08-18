"""Immutable submission evidence used by external evaluation tools.

The normal Arena workflow edits a task workspace in place.  Numerical tools such
as Triton FpSan need both the as-shipped submission and the optimized candidate,
while resume support must be able to prove that a cached tool report belongs to
the current sources.  This module captures the declared submission files outside
the mutable task workspace and produces deterministic content fingerprints.

Only task-declared source paths are captured.  Repository/image tasks whose
candidate spans additional files must declare ``evaluation_profile.submission_paths``;
silently guessing a whole repository boundary would make the evidence both huge
and unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable


_MANIFEST_NAME = "manifest.json"


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    raise ValueError(f"expected a string or list of strings, got {type(value).__name__}")


def declared_submission_paths(task_config: dict[str, Any]) -> tuple[str, ...]:
    """Return normalized task-declared candidate paths.

    ``evaluation_profile.submission_paths`` is authoritative when present.  The
    legacy source/target fields remain a backwards-compatible fallback.
    """

    profile = task_config.get("evaluation_profile") or {}
    if not isinstance(profile, dict):
        raise ValueError("evaluation_profile must be a mapping")
    explicit = profile.get("submission_paths")
    values: list[str] = []
    if explicit is not None:
        values.extend(_string_list(explicit))
    else:
        values.extend(_string_list(task_config.get("source_file_path")))
        values.extend(_string_list(task_config.get("target_file_path")))

    normalized: set[str] = set()
    for raw in values:
        path = Path(raw)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"submission path must be workspace-relative: {raw!r}")
        normalized.add(path.as_posix())
    return tuple(sorted(normalized))


def _repo_subdir(task_config: dict[str, Any]) -> str | None:
    configured = task_config.get("repo_subdir")
    if configured:
        return Path(str(configured)).name
    for key in ("image_repo_path", "repo_url"):
        value = task_config.get(key)
        if not value:
            continue
        name = str(value).rstrip("/")
        if name.endswith(".git"):
            name = name[:-4]
        return Path(name).name
    return None


def _candidate_locations(
    workspace: Path, relative: str, task_config: dict[str, Any]
) -> Iterable[Path]:
    repo_subdir = _repo_subdir(task_config)
    if repo_subdir:
        yield workspace / repo_subdir / relative
    yield workspace / relative


def resolve_submission_path(
    workspace: Path, relative: str, task_config: dict[str, Any]
) -> Path:
    """Resolve a declared path without allowing it to escape ``workspace``.

    Missing files resolve to the first canonical candidate so their absence can
    be fingerprinted (important for tasks where the agent creates ``kernel.py``).
    """

    _lexical, resolved = _submission_location(workspace, relative, task_config)
    return resolved


def _submission_location(
    workspace: Path, relative: str, task_config: dict[str, Any]
) -> tuple[Path, Path]:
    """Return both the declared lookup path and its contained resolved target."""

    workspace = workspace.resolve()
    candidates = list(_candidate_locations(workspace, relative, task_config))
    lexical = next((path for path in candidates if path.exists()), candidates[0])
    resolved = lexical.resolve(strict=False)
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"submission path escapes workspace: {relative!r}")
    return lexical, resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SubmissionEvidence:
    storage_dir: Path
    workspace: Path
    manifest: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return str(self.manifest["fingerprint"])

    @property
    def files_dir(self) -> Path:
        return self.storage_dir / "files"

    def verify(self) -> None:
        """Raise when the external evidence was changed after capture."""

        manifest_path = self.storage_dir / _MANIFEST_NAME
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if loaded != self.manifest:
            raise RuntimeError("submission evidence manifest changed after capture")
        for entry in loaded["entries"]:
            if not entry["exists"]:
                continue
            stored = self.files_dir / entry["workspace_relative_path"]
            if not stored.is_file() or _sha256_file(stored) != entry["sha256"]:
                raise RuntimeError(
                    "submission evidence file changed after capture: "
                    + entry["workspace_relative_path"]
                )

    def candidate_fingerprint(self) -> str:
        """Fingerprint the current optimized candidate over the same path set."""

        # ``workspace`` is canonicalized when evidence is captured and recorded
        # in the manifest. Do not resolve it again here: the candidate could
        # replace the workspace path itself with a symlink after capture and
        # thereby redefine the containment root.
        workspace = self.workspace
        if not workspace.is_absolute():
            raise ValueError("submission evidence workspace must be absolute")
        entries: list[dict[str, Any]] = []
        for original in self.manifest["entries"]:
            relative_value = str(original["workspace_relative_path"])
            relative = Path(relative_value)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError(
                    "submission evidence path must be workspace-relative: "
                    f"{relative_value!r}"
                )
            # Resolve the current candidate again instead of trusting the path
            # captured in the manifest. The candidate may have replaced a file
            # or one of its parent directories with a symlink after capture.
            # Continue with the resolved target so retargeting the originally
            # declared symlink cannot redirect the subsequent file read.
            lexical = workspace / relative
            path = lexical.resolve(strict=False)
            if not path.is_relative_to(workspace):
                raise ValueError(
                    "candidate submission path escapes workspace: "
                    f"{relative_value!r}"
                )
            exists = path.is_file()
            entries.append(
                {
                    "declared_path": original["declared_path"],
                    "workspace_relative_path": original["workspace_relative_path"],
                    "resolved_workspace_relative_path": path.relative_to(
                        workspace
                    ).as_posix(),
                    "symlink_target": (
                        os.readlink(lexical) if lexical.is_symlink() else None
                    ),
                    "exists": exists,
                    "sha256": _sha256_file(path) if exists else None,
                    "size": path.stat().st_size if exists else None,
                }
            )
        return _stable_hash(entries)


def capture_submission_evidence(
    workspace: Path,
    task_config: dict[str, Any],
    storage_dir: Path,
) -> SubmissionEvidence:
    """Capture declared submission files and an immutable manifest."""

    workspace = workspace.resolve()
    storage_dir = storage_dir.resolve()
    if storage_dir.exists():
        raise FileExistsError(f"submission evidence already exists: {storage_dir}")
    files_dir = storage_dir / "files"
    files_dir.mkdir(parents=True)

    entries: list[dict[str, Any]] = []
    for declared in declared_submission_paths(task_config):
        lexical, source = _submission_location(workspace, declared, task_config)
        relative = lexical.relative_to(workspace).as_posix()
        exists = source.is_file()
        entry = {
            "declared_path": declared,
            "workspace_relative_path": relative,
            "resolved_workspace_relative_path": source.relative_to(
                workspace
            ).as_posix(),
            "symlink_target": os.readlink(lexical) if lexical.is_symlink() else None,
            "exists": exists,
            "sha256": _sha256_file(source) if exists else None,
            "size": source.stat().st_size if exists else None,
        }
        entries.append(entry)
        if exists:
            destination = files_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    manifest_body = {
        "schema_version": 2,
        "workspace": str(workspace),
        "entries": entries,
    }
    manifest = {**manifest_body, "fingerprint": _stable_hash(manifest_body)}
    (storage_dir / _MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence = SubmissionEvidence(storage_dir, workspace, manifest)
    evidence.verify()
    return evidence


def load_submission_evidence(storage_dir: Path) -> SubmissionEvidence:
    storage_dir = storage_dir.resolve()
    manifest = json.loads((storage_dir / _MANIFEST_NAME).read_text(encoding="utf-8"))
    # Capture stores an already-canonical absolute workspace path. Retain that
    # lexical boundary instead of following a symlink that may have replaced the
    # workspace between capture and resume.
    workspace = Path(str(manifest["workspace"]))
    if not workspace.is_absolute():
        raise ValueError("submission evidence workspace must be absolute")
    evidence = SubmissionEvidence(storage_dir, workspace, manifest)
    evidence.verify()
    return evidence
