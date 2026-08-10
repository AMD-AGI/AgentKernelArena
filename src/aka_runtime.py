# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Content-addressed AgentKernelArena and backend runtime evidence.

Formal campaigns must execute bytes that were captured from the committed Git
tree, not an ambient mutable checkout.  This module deliberately avoids Git's
index, worktree filters, and repository-local process hooks when establishing
those bytes.  The resulting manifest is also the deterministic input to the
host-side sealed SquashFS materialization step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


EXECUTION_MANIFEST_SCHEMA = "aka.execution-snapshot-manifest/v1"
BACKEND_CLOSURE_SCHEMA = "aka.backend-runtime-closure/v1"
IMMUTABLE_MOUNT_RECEIPT_SCHEMA = "aka.execution-snapshot-mount-receipt/v2"
IMMUTABLE_MOUNT_POLICY = "sealed_memfd_squashfs_docker_bindable_read_only_v2"
ENGINE_SERVICE_SCHEMA = "aka.immutable-runtime-mount-service-ready/v2"
ENGINE_SERVICE_POLICY = "single_docker_bindable_snapshot_signal_lifetime_v2"
ENGINE_EVIDENCE_SCHEMA = "aka.immutable-runtime-mount-engine/v2"
HOST_ACCESS_POLICY_SCHEMA = "aka.immutable-runtime-host-access-policy/v1"
HOST_ACCESS_POLICY_ID = "private_ancestor_docker_daemon_fuse_v1"
GIT_EVIDENCE_POLICY = "head_tree_direct_bytes_no_filters_v1"
IMAGE_INPUT_SCHEMA = "aka.apex-runtime-image-input/v1"
IMAGE_INPUT_POLICY = "deterministic_squashfs_inputs_v1"
_MEMFD_SEALS = (
    "F_SEAL_WRITE",
    "F_SEAL_SHRINK",
    "F_SEAL_GROW",
    "F_SEAL_SEAL",
)
_HOST_MEMFD_SEALS = (
    "F_SEAL_GROW",
    "F_SEAL_SEAL",
    "F_SEAL_SHRINK",
    "F_SEAL_WRITE",
)

_GIT = Path("/usr/bin/git")
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TRACKED_MODES = {"100644", "100755"}
_REQUESTED_MOUNT_OPTIONS = [
    "ro",
    "nodev",
    "nosuid",
    "default_permissions",
    "allow_other",
    "subtype=squashfuse",
]
_DANGEROUS_CONFIG_KEYS = {
    "core.attributesfile",
    "core.fsmonitor",
    "core.hookspath",
    "core.sparsecheckout",
    "core.sparsecheckoutcone",
    "core.untrackedcache",
    "extensions.worktreeconfig",
}


class AkaRuntimeError(RuntimeError):
    """Raised when formal runtime bytes cannot be established exactly."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _controlled_git_environment() -> dict[str, str]:
    """Remove all caller-controlled Git process settings.

    The controlled variables below are assigned by this module after every
    inherited ``GIT_*`` key has been removed.
    """

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _git(root: Path, arguments: list[str], *, text: bool = False) -> bytes | str:
    if not _GIT.is_file() or _GIT.is_symlink():
        raise AkaRuntimeError("formal provenance requires canonical /usr/bin/git")
    argv = [
        str(_GIT),
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        *arguments,
    ]
    try:
        completed = subprocess.run(
            argv,
            cwd=str(root),
            env=_controlled_git_environment(),
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise AkaRuntimeError(f"Git provenance command failed: {arguments[0]}") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace")[-1000:]
        raise AkaRuntimeError(
            f"Git provenance command failed: {arguments[0]}: {detail.strip()}"
        )
    if text:
        try:
            return completed.stdout.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError as error:
            raise AkaRuntimeError("Git provenance output is not UTF-8") from error
    return completed.stdout


def _validated_root(root: Path) -> Path:
    try:
        candidate = root.absolute()
        metadata = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise AkaRuntimeError(f"cannot inspect AKA repository root: {root}") from error
    if (
        resolved != candidate
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise AkaRuntimeError("AKA repository root must be a canonical directory")
    top = Path(str(_git(candidate, ["rev-parse", "--show-toplevel"], text=True)))
    if top != candidate:
        raise AkaRuntimeError("AKA repository root differs from the Git worktree root")
    return candidate


def _local_config_keys(root: Path) -> set[str]:
    raw = _git(root, ["config", "--local", "--name-only", "--list", "-z"])
    assert isinstance(raw, bytes)
    try:
        return {
            value.decode("utf-8").lower()
            for value in raw.split(b"\0")
            if value
        }
    except UnicodeDecodeError as error:
        raise AkaRuntimeError("repository-local Git config is not UTF-8") from error


def _reject_git_shortcuts(root: Path) -> None:
    keys = _local_config_keys(root)
    dangerous = sorted(
        key
        for key in keys
        if key in _DANGEROUS_CONFIG_KEYS
        or key.startswith("filter.")
        or key.startswith("include.")
        or key.startswith("includeif.")
    )
    filemode = _git(root, ["config", "--local", "--get", "core.filemode"], text=True)
    if filemode and str(filemode).lower() != "true":
        dangerous.append("core.filemode")
    if dangerous:
        raise AkaRuntimeError(
            "formal checkout rejects repository-local Git bypass config: "
            + ", ".join(sorted(set(dangerous)))
        )

    raw_flags = _git(root, ["ls-files", "-v", "-z"])
    assert isinstance(raw_flags, bytes)
    flagged: list[str] = []
    for record in raw_flags.split(b"\0"):
        if not record:
            continue
        if len(record) < 3 or record[1:2] != b" ":
            raise AkaRuntimeError("cannot parse Git index flags")
        marker = chr(record[0])
        if marker == "S" or marker.islower():
            flagged.append(record[2:].decode("utf-8", "replace"))
    if flagged:
        raise AkaRuntimeError(
            "formal checkout rejects assume-unchanged/skip-worktree entries: "
            + ", ".join(flagged[:10])
        )


def _tree_entries(root: Path, commit: str) -> list[dict[str, str]]:
    raw = _git(root, ["ls-tree", "-rz", "--full-tree", commit])
    assert isinstance(raw, bytes)
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            header, encoded_path = record.split(b"\t", 1)
            mode, kind, object_id = header.decode("ascii").split(" ")
            relative = encoded_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise AkaRuntimeError("cannot parse committed Git tree") from error
        logical = PurePosixPath(relative)
        if (
            kind != "blob"
            or mode not in _TRACKED_MODES
            or not _SHA1.fullmatch(object_id)
            or logical.is_absolute()
            or ".." in logical.parts
            or relative in seen
        ):
            raise AkaRuntimeError(f"unsupported committed Git tree entry: {relative}")
        seen.add(relative)
        entries.append({"path": relative, "mode": mode, "git_blob": object_id})
    if not entries:
        raise AkaRuntimeError("committed Git tree contains no regular files")
    return entries


def _read_tracked_file(root: Path, entry: dict[str, str]) -> bytes:
    path = root / entry["path"]
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise AkaRuntimeError(f"cannot open tracked file safely: {entry['path']}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AkaRuntimeError(f"tracked path is not a regular file: {entry['path']}")
        observed_mode = "100755" if metadata.st_mode & 0o111 else "100644"
        if observed_mode != entry["mode"]:
            raise AkaRuntimeError(f"tracked mode differs from commit: {entry['path']}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        value = b"".join(chunks)
    finally:
        os.close(descriptor)
    git_blob = hashlib.sha1(b"blob " + str(len(value)).encode() + b"\0" + value).hexdigest()
    if git_blob != entry["git_blob"]:
        raise AkaRuntimeError(f"tracked bytes differ from commit: {entry['path']}")
    return value


def capture_execution_manifest(root: Path) -> dict[str, Any]:
    """Capture every committed AKA file as deterministic snapshot input."""

    root = _validated_root(root)
    _reject_git_shortcuts(root)
    commit = str(_git(root, ["rev-parse", "--verify", "HEAD^{commit}"], text=True))
    tree = str(_git(root, ["rev-parse", "--verify", "HEAD^{tree}"], text=True))
    if not _SHA1.fullmatch(commit) or not _SHA1.fullmatch(tree):
        raise AkaRuntimeError("AKA checkout has an invalid commit or tree identity")
    files: list[dict[str, Any]] = []
    for entry in _tree_entries(root, commit):
        value = _read_tracked_file(root, entry)
        files.append(
            {
                **entry,
                "size": len(value),
                "sha256": _sha256_bytes(value),
            }
        )
    if str(_git(root, ["rev-parse", "--verify", "HEAD^{commit}"], text=True)) != commit:
        raise AkaRuntimeError("AKA HEAD changed while the runtime was captured")
    material: dict[str, Any] = {
        "schema": EXECUTION_MANIFEST_SCHEMA,
        "policy_id": GIT_EVIDENCE_POLICY,
        "source": {
            "commit": commit,
            "tree": tree,
            "file_count": len(files),
            "git_environment": "all_inherited_GIT_variables_removed_v1",
            "index_shortcuts": "rejected",
            "worktree_filters": "not_invoked",
        },
        "files": files,
    }
    return {**material, "manifest_sha256": _digest(material)}


def verify_execution_manifest(
    root: Path, manifest: dict[str, Any], expected_sha256: str | None = None
) -> dict[str, Any]:
    """Revalidate a manifest against the exact current checkout bytes."""

    if not isinstance(manifest, dict):
        raise AkaRuntimeError("AKA runtime manifest must be an object")
    material = dict(manifest)
    observed_digest = material.pop("manifest_sha256", None)
    if (
        manifest.get("schema") != EXECUTION_MANIFEST_SCHEMA
        or not isinstance(observed_digest, str)
        or not _SHA256.fullmatch(observed_digest)
        or observed_digest != _digest(material)
        or (expected_sha256 is not None and observed_digest != expected_sha256)
    ):
        raise AkaRuntimeError("AKA runtime manifest digest or schema is invalid")
    current = capture_execution_manifest(root)
    if current != manifest:
        raise AkaRuntimeError("AKA runtime bytes differ from the captured manifest")
    return current


def verify_materialized_snapshot(
    root: Path, manifest: dict[str, Any], expected_sha256: str | None = None
) -> dict[str, Any]:
    """Verify a Git-free materialized tree against its source manifest."""

    if not isinstance(manifest, dict):
        raise AkaRuntimeError("AKA runtime manifest must be an object")
    material = dict(manifest)
    observed_digest = material.pop("manifest_sha256", None)
    files = manifest.get("files")
    if (
        manifest.get("schema") != EXECUTION_MANIFEST_SCHEMA
        or not isinstance(observed_digest, str)
        or observed_digest != _digest(material)
        or (expected_sha256 is not None and observed_digest != expected_sha256)
        or not isinstance(files, list)
    ):
        raise AkaRuntimeError("AKA runtime manifest digest or schema is invalid")
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise AkaRuntimeError("AKA snapshot root is unavailable") from error
    if not root.is_dir() or root.is_symlink():
        raise AkaRuntimeError("AKA snapshot root is not a regular directory")
    expected_paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise AkaRuntimeError("AKA snapshot manifest contains an invalid file")
        relative = item.get("path")
        logical = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            logical is None
            or logical.is_absolute()
            or ".." in logical.parts
            or relative in expected_paths
        ):
            raise AkaRuntimeError("AKA snapshot manifest contains an unsafe path")
        expected_paths.add(relative)
        path = root / relative
        try:
            metadata = path.lstat()
        except OSError as error:
            raise AkaRuntimeError(f"AKA snapshot file is absent: {relative}") from error
        mode = "100755" if metadata.st_mode & 0o111 else "100644"
        if (
            not path.is_file()
            or path.is_symlink()
            or mode != item.get("mode")
            or metadata.st_size != item.get("size")
            or _sha256_file(path) != item.get("sha256")
        ):
            raise AkaRuntimeError(f"AKA snapshot file differs: {relative}")
    observed_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if observed_paths != expected_paths:
        raise AkaRuntimeError("AKA snapshot contains missing or additional files")
    expected_directories = {
        parent.as_posix()
        for relative in expected_paths
        for parent in PurePosixPath(relative).parents
        if parent != PurePosixPath(".")
    }
    observed_directories = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    if observed_directories != expected_directories:
        raise AkaRuntimeError("AKA snapshot contains missing or additional directories")
    return manifest


def materialization_inputs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the sorted relative files/modes consumed by the image builder."""

    material = dict(manifest)
    observed = material.pop("manifest_sha256", None)
    files = manifest.get("files")
    if observed != _digest(material) or not isinstance(files, list):
        raise AkaRuntimeError("cannot use an invalid AKA runtime manifest")
    return [
        {"path": item["path"], "mode": item["mode"], "sha256": item["sha256"]}
        for item in files
    ]


def execution_image_inputs(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Describe the exact normalized SquashFS inputs for one AKA snapshot."""

    verified = verify_materialized_snapshot(
        root, manifest, manifest.get("manifest_sha256")
    )
    directories = {
        parent.as_posix()
        for item in verified["files"]
        for parent in PurePosixPath(item["path"]).parents
        if parent != PurePosixPath(".")
    }
    entries: list[dict[str, Any]] = [
        {"path": ".", "type": "directory", "mode": 0o555}
    ]
    entries.extend(
        {"path": relative, "type": "directory", "mode": 0o555}
        for relative in directories
    )
    entries.extend(
        {
            "path": item["path"],
            "type": "file",
            "mode": 0o555 if item["mode"] == "100755" else 0o444,
            "size": item["size"],
            "sha256": item["sha256"],
        }
        for item in verified["files"]
    )
    entries = [
        entries[0],
        *sorted(entries[1:], key=lambda item: os.fsencode(item["path"])),
    ]
    material = {
        "schema": IMAGE_INPUT_SCHEMA,
        "policy_id": IMAGE_INPUT_POLICY,
        "runtime_manifest_sha256": verified["manifest_sha256"],
        "entries": entries,
        "entries_sha256": _digest(entries),
        "normalization": {
            "uid": 0,
            "gid": 0,
            "mtime_epoch": 0,
            "xattrs": "none",
            "ordering": "utf8_posix_path_ascending",
            "format": "squashfs",
            "compression": "caller_pinned_and_receipted",
        },
    }
    return {**material, "sha256": _digest(material)}


def materialize_execution_snapshot(
    root: Path, manifest: dict[str, Any], destination: Path
) -> Path:
    """Copy verified inputs into a new content-addressed image staging tree."""

    verified = verify_execution_manifest(root, manifest)
    digest = verified["manifest_sha256"]
    expected_destination = destination.parent / digest
    if destination.absolute() != expected_destination.absolute() or destination.exists():
        raise AkaRuntimeError("snapshot destination must be a new directory named by its digest")
    destination.mkdir(parents=True, mode=0o700)
    for item in verified["files"]:
        source = root / item["path"]
        target = destination / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o700 if item["mode"] == "100755" else 0o600,
        )
        try:
            with source.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    os.write(descriptor, chunk)
        finally:
            os.close(descriptor)
        target.chmod(0o555 if item["mode"] == "100755" else 0o444)
    if any(
        _sha256_file(destination / item["path"]) != item["sha256"]
        for item in verified["files"]
    ):
        raise AkaRuntimeError("materialized AKA snapshot differs from its source manifest")
    directories = sorted(
        (path for path in destination.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        directory.chmod(0o555)
    destination.chmod(0o555)
    execution_image_inputs(destination, verified)
    return destination


def _regular_tree(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise AkaRuntimeError(f"backend component is not a regular tree: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if path.is_symlink():
            raise AkaRuntimeError(f"backend component contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise AkaRuntimeError(f"backend component contains a special file: {path}")
        files.append(
            {
                "path": relative,
                "mode": stat.S_IMODE(metadata.st_mode),
                "size": metadata.st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise AkaRuntimeError(f"backend component is empty: {root}")
    return files


def _nearest_package_root(path: Path) -> Path | None:
    for parent in (path.parent, *path.parents):
        if (parent / "package.json").is_file():
            return parent
    return None


def _package_dependencies(package_root: Path) -> tuple[set[str], set[str]]:
    try:
        payload = json.loads((package_root / "package.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AkaRuntimeError(f"cannot parse backend package: {package_root}") from error
    if not isinstance(payload, dict):
        raise AkaRuntimeError(f"backend package metadata is not an object: {package_root}")
    required = payload.get("dependencies") or {}
    optional = payload.get("optionalDependencies") or {}
    if not isinstance(required, dict) or not isinstance(optional, dict):
        raise AkaRuntimeError(f"backend dependency metadata is malformed: {package_root}")
    return set(required), set(optional)


def _resolve_node_dependency(package_root: Path, name: str) -> Path | None:
    for ancestor in (package_root, *package_root.parents):
        candidate = ancestor / "node_modules" / PurePosixPath(name)
        if (candidate / "package.json").is_file():
            return candidate.resolve(strict=True)
    return None


def _node_package_closure(root: Path) -> list[Path]:
    pending = [root]
    selected: dict[Path, None] = {}
    while pending:
        package = pending.pop()
        package = package.resolve(strict=True)
        if package in selected:
            continue
        selected[package] = None
        required, optional = _package_dependencies(package)
        for name in sorted(required | optional):
            dependency = _resolve_node_dependency(package, name)
            if dependency is None:
                if name in required:
                    raise AkaRuntimeError(f"required backend dependency is absent: {name}")
                continue
            pending.append(dependency)
    return sorted(selected)


def _symlink_chain(path: Path) -> list[dict[str, str]]:
    chain: list[dict[str, str]] = []
    current = path.absolute()
    seen: set[Path] = set()
    while current.is_symlink():
        if current in seen:
            raise AkaRuntimeError("backend launcher symlink chain contains a loop")
        seen.add(current)
        target = os.readlink(current)
        chain.append({"path": str(current), "target": target})
        current = (current.parent / target).absolute() if not os.path.isabs(target) else Path(target)
    return chain


def capture_backend_closure(backend: str, executable: str | None = None) -> dict[str, Any]:
    """Capture a backend launcher, interpreter, and complete package closure."""

    if backend not in {"codex", "claude", "cursor"}:
        raise AkaRuntimeError(f"unsupported backend runtime: {backend}")
    requested = executable or shutil.which(backend)
    if not requested:
        raise AkaRuntimeError(f"backend executable is unavailable: {backend}")
    launcher = Path(requested).absolute()
    chain = _symlink_chain(launcher)
    try:
        resolved = launcher.resolve(strict=True)
        metadata = resolved.lstat()
    except OSError as error:
        raise AkaRuntimeError(f"cannot resolve backend launcher: {launcher}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise AkaRuntimeError("backend launcher does not resolve to a regular file")

    try:
        first_line = resolved.open("rb").readline(4096).decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise AkaRuntimeError("cannot inspect backend launcher shebang") from error
    interpreter: Path | None = None
    if first_line.startswith("#!"):
        words = first_line[2:].strip().split()
        command = words[1] if words and words[0] == "/usr/bin/env" and len(words) > 1 else words[0]
        selected = shutil.which(command) if not os.path.isabs(command) else command
        if selected:
            interpreter = Path(selected).resolve(strict=True)
    components: list[dict[str, Any]] = []
    package_root = _nearest_package_root(resolved)
    if package_root is not None:
        for package in _node_package_closure(package_root):
            files = _regular_tree(package)
            components.append(
                {
                    "kind": "node_package",
                    "root": str(package),
                    "files": files,
                    "files_sha256": _digest(files),
                }
            )
    launcher_record = {
        "requested_path": str(launcher),
        "symlink_chain": chain,
        "resolved_path": str(resolved),
        "mode": stat.S_IMODE(metadata.st_mode),
        "size": metadata.st_size,
        "sha256": _sha256_file(resolved),
    }
    interpreter_record = None
    if interpreter is not None:
        interpreter_metadata = interpreter.lstat()
        if not stat.S_ISREG(interpreter_metadata.st_mode):
            raise AkaRuntimeError("backend interpreter is not a regular file")
        interpreter_record = {
            "resolved_path": str(interpreter),
            "mode": stat.S_IMODE(interpreter_metadata.st_mode),
            "size": interpreter_metadata.st_size,
            "sha256": _sha256_file(interpreter),
        }
    material = {
        "schema": BACKEND_CLOSURE_SCHEMA,
        "backend": backend,
        "launcher": launcher_record,
        "interpreter": interpreter_record,
        "components": components,
    }
    return {**material, "closure_sha256": _digest(material)}


def verify_backend_closure(
    closure: dict[str, Any], expected_sha256: str | None = None
) -> dict[str, Any]:
    if not isinstance(closure, dict):
        raise AkaRuntimeError("backend closure must be an object")
    material = dict(closure)
    observed = material.pop("closure_sha256", None)
    if (
        closure.get("schema") != BACKEND_CLOSURE_SCHEMA
        or not isinstance(observed, str)
        or not _SHA256.fullmatch(observed)
        or observed != _digest(material)
        or (expected_sha256 is not None and observed != expected_sha256)
    ):
        raise AkaRuntimeError("backend closure digest or schema is invalid")
    current = capture_backend_closure(
        str(closure.get("backend")), str(closure.get("launcher", {}).get("requested_path"))
    )
    if current != closure:
        raise AkaRuntimeError("backend runtime closure changed after capture")
    return current


def _decode_mountinfo_path(value: str) -> Path:
    decoded = value
    for escaped, plain in (
        (r"\040", " "),
        (r"\011", "\t"),
        (r"\012", "\n"),
        (r"\134", "\\"),
    ):
        decoded = decoded.replace(escaped, plain)
    return Path(decoded)


def _mountinfo_entries() -> list[dict[str, Any]]:
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise AkaRuntimeError("cannot inspect current AKA runtime mounts") from error
    entries: list[dict[str, Any]] = []
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
            entry = {
                "mount_id": int(fields[0]),
                "parent_id": int(fields[1]),
                "major_minor": fields[2],
                "root": str(_decode_mountinfo_path(fields[3])),
                "mount_point": str(_decode_mountinfo_path(fields[4])),
                "mount_options": fields[5].split(","),
                "filesystem_type": fields[separator + 1],
                "source": fields[separator + 2],
                "super_options": fields[separator + 3].split(","),
            }
        except (IndexError, ValueError) as error:
            raise AkaRuntimeError("current mountinfo contains a malformed entry") from error
        if separator < 6 or entry["mount_id"] <= 0 or entry["parent_id"] <= 0:
            raise AkaRuntimeError("current mountinfo contains an invalid entry")
        entries.append(entry)
    if not entries:
        raise AkaRuntimeError("current mountinfo is empty")
    return entries


def _current_snapshot_mount(root: Path) -> dict[str, Any]:
    entries = _mountinfo_entries()
    exact = [entry for entry in entries if Path(entry["mount_point"]) == root]
    if len(exact) != 1:
        raise AkaRuntimeError("AKA snapshot root is not one exact mount point")
    entry = exact[0]
    nested = sorted(
        candidate["mount_point"]
        for candidate in entries
        if Path(candidate["mount_point"]) != root
        and Path(candidate["mount_point"]).is_relative_to(root)
    )
    return {
        "path": str(root),
        **entry,
        "read_only": "ro" in entry["mount_options"],
        "nested_mounts": nested,
    }


def _host_access_policy_valid(policy: Any) -> bool:
    if not isinstance(policy, dict) or set(policy) != {
        "schema",
        "policy_id",
        "requested_mount_options",
        "private_ancestor",
        "fuse_config",
        "mount_owner",
        "worker",
        "docker_daemon",
        "sha256",
    }:
        return False
    material = dict(policy)
    observed = material.pop("sha256", None)
    ancestor = policy.get("private_ancestor")
    config = policy.get("fuse_config")
    owner = policy.get("mount_owner")
    worker = policy.get("worker")
    daemon = policy.get("docker_daemon")
    return bool(
        policy.get("schema") == HOST_ACCESS_POLICY_SCHEMA
        and policy.get("policy_id") == HOST_ACCESS_POLICY_ID
        and policy.get("requested_mount_options") == _REQUESTED_MOUNT_OPTIONS
        and isinstance(observed, str)
        and observed == _digest(material)
        and isinstance(ancestor, dict)
        and set(ancestor) == {"path", "device", "inode", "uid", "gid", "mode"}
        and Path(str(ancestor.get("path") or "")).is_absolute()
        and Path(str(ancestor["path"]))
        == Path(os.path.abspath(str(ancestor["path"])))
        and all(type(ancestor.get(key)) is int for key in ("device", "inode", "uid", "gid", "mode"))
        and ancestor["device"] >= 0
        and ancestor["inode"] > 0
        and ancestor["mode"] == 0o700
        and isinstance(config, dict)
        and set(config) == {
            "path", "device", "inode", "uid", "gid", "mode", "nlink",
            "size_bytes", "sha256", "user_allow_other",
        }
        and config.get("path") == "/etc/fuse.conf"
        and config.get("uid") == 0
        and config.get("gid") == 0
        and type(config.get("mode")) is int
        and config["mode"] & 0o022 == 0
        and config.get("nlink") == 1
        and type(config.get("device")) is int
        and config["device"] >= 0
        and type(config.get("inode")) is int
        and config["inode"] > 0
        and type(config.get("size_bytes")) is int
        and config["size_bytes"] > 0
        and _SHA256.fullmatch(str(config.get("sha256") or "")) is not None
        and config.get("user_allow_other") is True
        and isinstance(owner, dict)
        and set(owner) == {"uid", "gid"}
        and owner == worker
        and owner == {"uid": ancestor["uid"], "gid": ancestor["gid"]}
        and daemon == {
            "uid": 0,
            "trusted_boundary": True,
            "access_via": "fuse_allow_other_with_private_ancestor_v1",
        }
    )


def load_runtime_service_evidence(
    path: str | Path,
    *,
    file_sha256: str,
    content_sha256: str,
    manifest_sha256: str,
    image_sha256: str,
) -> dict[str, Any]:
    """Load and validate the host mount engine evidence persisted by the runner."""

    source = Path(path)
    parent_fd = descriptor = -1
    try:
        if not source.is_absolute() or source != Path(os.path.abspath(source)):
            raise AkaRuntimeError("runtime service evidence path must be canonical")
        parent_metadata = source.parent.lstat()
        metadata = source.lstat()
        if (
            source.parent.resolve(strict=True) != source.parent
            or stat.S_ISLNK(parent_metadata.st_mode)
            or not stat.S_ISDIR(parent_metadata.st_mode)
            or source.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or metadata.st_size <= 0
            or metadata.st_size > 16 * 1024 * 1024
        ):
            raise AkaRuntimeError("immutable runtime service evidence file is unsafe")
        parent_fd = os.open(
            source.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_parent = os.fstat(parent_fd)
        if (opened_parent.st_dev, opened_parent.st_ino) != (
            parent_metadata.st_dev,
            parent_metadata.st_ino,
        ):
            raise AkaRuntimeError("runtime service evidence parent changed")
        descriptor = os.open(
            source.name,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = before.st_size + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        stable = (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        ) == (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        )
        if (
            not stable
            or (before.st_dev, before.st_ino) != (metadata.st_dev, metadata.st_ino)
            or len(payload) != before.st_size
        ):
            raise AkaRuntimeError("immutable runtime service evidence changed")
        evidence = json.loads(payload.decode("utf-8"))
    except AkaRuntimeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AkaRuntimeError("cannot read immutable runtime engine evidence") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_fd >= 0:
            os.close(parent_fd)
    if (
        hashlib.sha256(payload).hexdigest() != file_sha256
        or not isinstance(evidence, dict)
    ):
        raise AkaRuntimeError("immutable runtime engine evidence file is unsafe")
    service_material = dict(evidence)
    service_digest = service_material.pop("sha256", None)
    receipt = evidence.get("mount_receipt")
    engine = evidence.get("engine_evidence")
    if not isinstance(receipt, dict) or not isinstance(engine, dict):
        raise AkaRuntimeError("immutable runtime engine evidence is incomplete")
    receipt_material = dict(receipt)
    receipt_digest = receipt_material.pop("sha256", None)
    engine_material = dict(engine)
    engine_digest = engine_material.pop("sha256", None)
    service = evidence.get("service")
    service_process = engine.get("process")
    policy = receipt.get("host_access_policy")
    mount = receipt.get("mount")
    ready_path = evidence.get("ready_path")
    image = engine.get("image")
    backing = receipt.get("backing")
    if (
        set(evidence) != {
            "schema", "policy_id", "ready_path", "service", "mount_receipt",
            "engine_evidence", "sha256",
        }
        or set(engine) != {
            "schema", "policy_id", "receipt_sha256",
            "runtime_image_input_sha256", "image", "tools",
            "requested_mount_options", "host_access_policy_sha256", "process",
            "mountpoint_source", "mount", "inventory_verification",
            "write_probe_errno", "sha256",
        }
        or not isinstance(service, dict)
        or set(service) != {
            "pid", "starttime", "owner", "accepted_signals", "engine_process",
        }
        or type(service.get("pid")) is not int
        or service["pid"] <= 1
        or type(service.get("starttime")) is not int
        or service["starttime"] <= 0
        or not _host_access_policy_valid(policy)
        or service.get("owner") != policy.get("mount_owner")
        or service.get("accepted_signals") != ["SIGINT", "SIGTERM"]
        or service.get("engine_process")
        != {
            "pid": service_process.get("pid")
            if isinstance(service_process, dict)
            else None,
            "starttime": service_process.get("starttime")
            if isinstance(service_process, dict)
            else None,
        }
        or not isinstance(service_process, dict)
        or set(service_process) != {"pid", "starttime", "foreground"}
        or type(service_process.get("pid")) is not int
        or service_process["pid"] <= 1
        or type(service_process.get("starttime")) is not int
        or service_process["starttime"] <= 0
        or service_process["pid"] == service["pid"]
        or service_process.get("foreground") is not True
        or not isinstance(ready_path, str)
        or not Path(ready_path).is_absolute()
        or Path(ready_path) != Path(os.path.abspath(ready_path))
        or Path(ready_path).parent
        != Path(policy["private_ancestor"]["path"])
        or evidence.get("schema") != ENGINE_SERVICE_SCHEMA
        or evidence.get("policy_id") != ENGINE_SERVICE_POLICY
        or service_digest != content_sha256
        or service_digest != _digest(service_material)
        or set(receipt) != {
            "schema", "policy_id", "root", "runtime_manifest_sha256",
            "runtime_image_input_sha256", "image_sha256", "backing",
            "requested_mount_options", "host_access_policy", "mount", "sha256",
        }
        or receipt.get("schema") != "aka.host-runtime-immutable-mount/v2"
        or receipt.get("policy_id") != IMMUTABLE_MOUNT_POLICY
        or receipt.get("runtime_manifest_sha256") != manifest_sha256
        or receipt.get("image_sha256") != image_sha256
        or receipt.get("requested_mount_options") != _REQUESTED_MOUNT_OPTIONS
        or receipt_digest != _digest(receipt_material)
        or not isinstance(mount, dict)
        or set(mount) != {
            "mount_id", "device", "root", "mount_point", "filesystem",
            "mount_options", "super_options", "read_only",
        }
        or type(mount.get("mount_id")) is not int
        or mount["mount_id"] <= 0
        or not isinstance(mount.get("mount_point"), str)
        or Path(mount["mount_point"]) != Path(receipt.get("root", ""))
        or not Path(mount["mount_point"]).is_absolute()
        or Path(mount["mount_point"])
        != Path(os.path.abspath(mount["mount_point"]))
        or mount.get("root") != "/"
        or mount.get("filesystem")
        not in {"squashfs", "fuse.squashfuse", "fuse.squashfuse_ll"}
        or mount.get("read_only") is not True
        or not {"ro", "nodev", "nosuid"}.issubset(
            set(mount.get("mount_options", []))
        )
        or "allow_other" not in mount.get("super_options", [])
        or f"user_id={policy['mount_owner']['uid']}"
        not in mount.get("super_options", [])
        or f"group_id={policy['mount_owner']['gid']}"
        not in mount.get("super_options", [])
        or backing != {
            "kind": "sealed_memfd", "seals": list(_HOST_MEMFD_SEALS),
        }
        or engine.get("schema") != ENGINE_EVIDENCE_SCHEMA
        or engine.get("policy_id") != IMMUTABLE_MOUNT_POLICY
        or engine.get("receipt_sha256") != receipt_digest
        or engine.get("runtime_image_input_sha256")
        != receipt.get("runtime_image_input_sha256")
        or engine.get("requested_mount_options") != _REQUESTED_MOUNT_OPTIONS
        or engine.get("host_access_policy_sha256") != policy.get("sha256")
        or engine.get("mount") != mount
        or not isinstance(image, dict)
        or set(image) != {"size_bytes", "sha256", "memfd_seals"}
        or type(image.get("size_bytes")) is not int
        or image["size_bytes"] <= 0
        or image.get("sha256") != receipt.get("image_sha256")
        or image.get("memfd_seals") != list(_HOST_MEMFD_SEALS)
        or engine_digest != _digest(engine_material)
    ):
        raise AkaRuntimeError("immutable runtime engine evidence is invalid")
    return evidence


def recover_runtime_service(
    path: str | Path,
    *,
    file_sha256: str,
    content_sha256: str,
    manifest_sha256: str,
    image_sha256: str,
    controller_pid: int,
    controller_starttime: int,
    mountpoint: str | Path,
    private_ancestor: str | Path,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Recover an abnormal mount service from its exact persisted evidence."""

    if __package__:
        from .immutable_runtime_mount import (
            ImmutableRuntimeMountError,
            recover_immutable_runtime_service,
        )
    else:
        from immutable_runtime_mount import (
            ImmutableRuntimeMountError,
            recover_immutable_runtime_service,
        )

    evidence = load_runtime_service_evidence(
        path,
        file_sha256=file_sha256,
        content_sha256=content_sha256,
        manifest_sha256=manifest_sha256,
        image_sha256=image_sha256,
    )
    try:
        return recover_immutable_runtime_service(
            evidence,
            controller_pid=controller_pid,
            controller_starttime=controller_starttime,
            mountpoint=mountpoint,
            private_ancestor=private_ancestor,
            timeout_seconds=timeout_seconds,
        )
    except (ImmutableRuntimeMountError, OSError, ValueError) as error:
        raise AkaRuntimeError(str(error)) from error


def validate_immutable_mount_receipt(
    receipt: dict[str, Any], expected_manifest_sha256: str, expected_root: Path
) -> dict[str, Any]:
    """Validate the host-produced receipt for a sealed SquashFS execution mount."""

    if not isinstance(receipt, dict) or set(receipt) != {
        "schema", "policy_id", "requested_mount_options",
        "runtime_service_evidence_sha256", "runtime_engine_evidence_sha256",
        "host_access_policy", "manifest_sha256", "image_sha256",
        "memfd_seals", "mount", "sha256",
    }:
        raise AkaRuntimeError("AKA immutable mount receipt must be an object")
    material = dict(receipt)
    observed = material.pop("sha256", None)
    mount = receipt.get("mount")
    seals = receipt.get("memfd_seals")
    host_policy = receipt.get("host_access_policy")
    try:
        root = expected_root.resolve(strict=True)
        observed_mount = _current_snapshot_mount(root)
    except OSError as error:
        raise AkaRuntimeError("AKA immutable mount root is unavailable") from error
    if (
        receipt.get("schema") != IMMUTABLE_MOUNT_RECEIPT_SCHEMA
        or receipt.get("policy_id") != IMMUTABLE_MOUNT_POLICY
        or receipt.get("requested_mount_options") != _REQUESTED_MOUNT_OPTIONS
        or not _SHA256.fullmatch(
            str(receipt.get("runtime_service_evidence_sha256") or "")
        )
        or not _SHA256.fullmatch(
            str(receipt.get("runtime_engine_evidence_sha256") or "")
        )
        or not _host_access_policy_valid(host_policy)
        or receipt.get("manifest_sha256") != expected_manifest_sha256
        or not _SHA256.fullmatch(str(receipt.get("image_sha256") or ""))
        or not isinstance(observed, str)
        or observed != _digest(material)
        or not isinstance(mount, dict)
        or not isinstance(mount.get("path"), str)
        or not Path(mount["path"]).is_absolute()
        or Path(mount["path"]) != root
        or mount.get("filesystem_type")
        not in {"squashfs", "fuse.squashfuse", "fuse.squashfuse_ll"}
        or mount.get("read_only") is not True
        or "allow_other" not in mount.get("super_options", [])
        or f"user_id={host_policy['mount_owner']['uid']}"
        not in mount.get("super_options", [])
        or f"group_id={host_policy['mount_owner']['gid']}"
        not in mount.get("super_options", [])
        or mount.get("root") != "/"
        or not isinstance(mount.get("mount_id"), int)
        or mount["mount_id"] <= 0
        or mount.get("nested_mounts") != []
        or mount != observed_mount
        or seals != list(_MEMFD_SEALS)
    ):
        raise AkaRuntimeError("AKA immutable mount receipt is invalid")
    return receipt


def create_immutable_mount_receipt(
    root: Path,
    manifest_sha256: str,
    image_sha256: str,
    runtime_service_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Attest the immutable AKA mount in the caller's current namespace."""

    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise AkaRuntimeError("AKA immutable mount root is unavailable") from error
    if (
        resolved != root.absolute()
        or not _SHA256.fullmatch(manifest_sha256)
        or not _SHA256.fullmatch(image_sha256)
    ):
        raise AkaRuntimeError("AKA immutable mount identity is invalid")
    engine_evidence = runtime_service_evidence["engine_evidence"]
    host_policy = runtime_service_evidence["mount_receipt"]["host_access_policy"]
    material = {
        "schema": IMMUTABLE_MOUNT_RECEIPT_SCHEMA,
        "policy_id": IMMUTABLE_MOUNT_POLICY,
        "requested_mount_options": list(_REQUESTED_MOUNT_OPTIONS),
        "runtime_service_evidence_sha256": runtime_service_evidence["sha256"],
        "runtime_engine_evidence_sha256": engine_evidence["sha256"],
        "host_access_policy": host_policy,
        "manifest_sha256": manifest_sha256,
        "image_sha256": image_sha256,
        "memfd_seals": list(_MEMFD_SEALS),
        "mount": _current_snapshot_mount(resolved),
    }
    receipt = {**material, "sha256": _digest(material)}
    return validate_immutable_mount_receipt(
        receipt, manifest_sha256, expected_root=resolved
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AkaRuntimeError(f"cannot read runtime manifest: {path}") from error
    if not isinstance(payload, dict):
        raise AkaRuntimeError("runtime manifest must be a JSON object")
    return payload


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    discover = commands.add_parser("discover")
    discover.add_argument("--root", required=True, type=Path)
    discover.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--root", required=True, type=Path)
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--sha256", required=True)
    materialize = commands.add_parser("materialize")
    materialize.add_argument("--root", required=True, type=Path)
    materialize.add_argument("--manifest", required=True, type=Path)
    materialize.add_argument("--destination", required=True, type=Path)
    image_input = commands.add_parser("image-input")
    image_input.add_argument("--root", required=True, type=Path)
    image_input.add_argument("--manifest", required=True, type=Path)
    image_input.add_argument("--sha256", required=True)
    image_input.add_argument("--output", required=True, type=Path)
    mount_receipt = commands.add_parser("mount-receipt")
    mount_receipt.add_argument("--root", required=True, type=Path)
    mount_receipt.add_argument("--manifest-sha256", required=True)
    mount_receipt.add_argument("--image-sha256", required=True)
    mount_receipt.add_argument("--service-evidence", required=True, type=Path)
    mount_receipt.add_argument("--service-file-sha256", required=True)
    mount_receipt.add_argument("--service-content-sha256", required=True)
    mount_receipt.add_argument("--output", required=True, type=Path)
    recover_service = commands.add_parser("recover-service")
    recover_service.add_argument("--service-evidence", required=True, type=Path)
    recover_service.add_argument("--service-file-sha256", required=True)
    recover_service.add_argument("--service-content-sha256", required=True)
    recover_service.add_argument("--manifest-sha256", required=True)
    recover_service.add_argument("--image-sha256", required=True)
    recover_service.add_argument("--controller-pid", required=True, type=int)
    recover_service.add_argument("--controller-starttime", required=True, type=int)
    recover_service.add_argument("--mountpoint", required=True, type=Path)
    recover_service.add_argument("--private-ancestor", required=True, type=Path)
    recover_service.add_argument("--timeout-seconds", type=float, default=10.0)
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if arguments.command == "discover":
            payload = capture_execution_manifest(arguments.root)
            _write_json(arguments.output, payload)
            print(payload["manifest_sha256"])
        elif arguments.command == "verify":
            verify_execution_manifest(
                arguments.root, _load_json(arguments.manifest), arguments.sha256
            )
            print(arguments.sha256)
        elif arguments.command == "materialize":
            manifest = _load_json(arguments.manifest)
            materialize_execution_snapshot(
                arguments.root, manifest, arguments.destination
            )
            print(manifest["manifest_sha256"])
        elif arguments.command == "image-input":
            manifest = _load_json(arguments.manifest)
            verify_materialized_snapshot(
                arguments.root, manifest, arguments.sha256
            )
            payload = execution_image_inputs(arguments.root, manifest)
            _write_json(arguments.output, payload)
            print(payload["sha256"])
        elif arguments.command == "mount-receipt":
            service = load_runtime_service_evidence(
                arguments.service_evidence,
                file_sha256=arguments.service_file_sha256,
                content_sha256=arguments.service_content_sha256,
                manifest_sha256=arguments.manifest_sha256,
                image_sha256=arguments.image_sha256,
            )
            payload = create_immutable_mount_receipt(
                arguments.root,
                arguments.manifest_sha256,
                arguments.image_sha256,
                service,
            )
            _write_json(arguments.output, payload)
            print(payload["sha256"])
        else:
            if not 0.0 < arguments.timeout_seconds <= 300.0:
                raise AkaRuntimeError("recovery timeout must be in (0, 300]")
            payload = recover_runtime_service(
                arguments.service_evidence,
                file_sha256=arguments.service_file_sha256,
                content_sha256=arguments.service_content_sha256,
                manifest_sha256=arguments.manifest_sha256,
                image_sha256=arguments.image_sha256,
                controller_pid=arguments.controller_pid,
                controller_starttime=arguments.controller_starttime,
                mountpoint=arguments.mountpoint,
                private_ancestor=arguments.private_ancestor,
                timeout_seconds=arguments.timeout_seconds,
            )
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    except AkaRuntimeError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
