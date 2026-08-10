# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Exact-byte planning and materialization for formal Apex runtimes.

The formal worker never executes the mutable checkout directly.  This module
validates tracked Apex bytes against one captured commit, inventories the
complete virtualenv and every statically declared editable or managed root,
and materializes those bytes into an attempt-local content-addressed snapshot.
Snapshot execution uses the container's pinned system interpreter through a
sealed ``python``/``python3`` wrapper.  Every primary invocation and child
alias exposed by this execution contract is forced through ``-I -S``; copied
``.pth`` and ``sitecustomize`` bytes remain evidence, never executable
configuration.
Execution additionally requires a receipt for an immutable, read-only image
mount.  Host mode bits alone are not an immutability claim.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import posixpath
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence
from urllib.parse import unquote, urlparse

if __package__:
    from .aka_runtime import (
        ENGINE_EVIDENCE_SCHEMA,
        ENGINE_SERVICE_SCHEMA,
        HOST_ACCESS_POLICY_SCHEMA,
        _host_access_policy_valid,
        load_runtime_service_evidence,
    )
else:
    from aka_runtime import (
        ENGINE_EVIDENCE_SCHEMA,
        ENGINE_SERVICE_SCHEMA,
        HOST_ACCESS_POLICY_SCHEMA,
        _host_access_policy_valid,
        load_runtime_service_evidence,
    )


RUNTIME_MANIFEST_SCHEMA = "aka.apex-runtime-manifest/v2"
RUNTIME_SNAPSHOT_SCHEMA = "aka.apex-runtime-snapshot/v2"
RUNTIME_POLICY_ID = "content_addressed_apex_runtime_snapshot_v2"
RUNTIME_BOOTSTRAP_NAME = "runtime_bootstrap.py"
RUNTIME_BOOTSTRAP_POLICY_ID = "sealed_python_alias_dispatch_v2"
RUNTIME_WRAPPER_NAME = "sealed-bin/python"
RUNTIME_WRAPPER_ALIASES = ("python", "python3")
RUNTIME_WRAPPER_POLICY_ID = "posix_wrapper_forced_isolated_no_site_v1"
RUNTIME_IMAGE_INPUT_SCHEMA = "aka.apex-runtime-image-input/v1"
RUNTIME_IMAGE_INPUT_POLICY_ID = "deterministic_squashfs_inputs_v1"
RUNTIME_IMMUTABLE_MOUNT_SCHEMA = "aka.apex-runtime-immutable-mount/v2"
RUNTIME_IMMUTABLE_MOUNT_POLICY_ID = (
    "sealed_memfd_squashfs_docker_bindable_read_only_v2"
)
_REQUIRED_MEMFD_SEALS = (
    "F_SEAL_GROW",
    "F_SEAL_SEAL",
    "F_SEAL_SHRINK",
    "F_SEAL_WRITE",
)
RUNTIME_BOOTSTRAP = b"""\
import json
import os
import runpy
import sys

root = os.path.dirname(os.path.abspath(__file__))
wrapper = sys.argv[1]
arguments = sys.argv[2:]
with open(os.path.join(root, "runtime_manifest.json"), "rb") as stream:
    manifest = json.load(stream)
execution = manifest.get("execution", {})
relative_paths = execution.get("pythonpath")
if not isinstance(relative_paths, list) or not relative_paths:
    raise SystemExit("invalid sealed runtime pythonpath")
pythonpath = []
for value in relative_paths:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise SystemExit("invalid sealed runtime pythonpath")
    candidate = os.path.abspath(os.path.join(root, value))
    if os.path.commonpath((root, candidate)) != root:
        raise SystemExit("sealed runtime pythonpath escapes snapshot")
    pythonpath.append(candidate)
sys.path[:0] = pythonpath
sys.executable = wrapper
sys._base_executable = wrapper

while arguments and arguments[0] in {"-B", "-E", "-I", "-P", "-S", "-s", "-u"}:
    arguments = arguments[1:]
if not arguments:
    raise SystemExit("interactive sealed Python is unavailable")
mode = arguments[0]
if mode == "--apex-entrypoint":
    if len(arguments) < 2:
        raise SystemExit("sealed Apex entrypoint is missing")
    entrypoint = arguments[1]
    sys.argv = [entrypoint, *arguments[2:]]
    runpy.run_path(entrypoint, run_name="__main__")
elif mode == "-c":
    if len(arguments) < 2:
        raise SystemExit("sealed Python -c source is missing")
    sys.argv = ["-c", *arguments[2:]]
    namespace = {"__name__": "__main__", "__package__": None}
    exec(compile(arguments[1], "<string>", "exec"), namespace, namespace)
elif mode == "-m":
    if len(arguments) < 2:
        raise SystemExit("sealed Python -m module is missing")
    sys.argv = [arguments[1], *arguments[2:]]
    runpy.run_module(arguments[1], run_name="__main__", alter_sys=True)
elif mode.startswith("-"):
    raise SystemExit("unsupported sealed Python option: " + mode)
else:
    sys.argv = [mode, *arguments[1:]]
    runpy.run_path(mode, run_name="__main__")
"""
RUNTIME_BOOTSTRAP_SHA256 = hashlib.sha256(RUNTIME_BOOTSTRAP).hexdigest()
RUNTIME_WRAPPER = b"""\
#!/bin/sh
wrapper_dir=${0%/*}
root=${wrapper_dir%/*}
case "$root" in
  /*) ;;
  *) echo "sealed Python requires an absolute argv[0]" >&2; exit 126 ;;
esac
exec "$root/venv/bin/python" -I -S -u "$root/runtime_bootstrap.py" "$0" "$@"
"""
RUNTIME_WRAPPER_SHA256 = hashlib.sha256(RUNTIME_WRAPPER).hexdigest()
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TREE_ENTRY = re.compile(
    rb"^(?P<mode>[0-9]{6}) (?P<kind>[a-z]+) (?P<oid>[0-9a-f]{40})\t(?P<path>.+)$"
)
_IGNORED_EXTERNAL_PARTS = frozenset({".git", ".hg", ".svn"})
_REJECTED_LOCAL_GIT_OPTIONS = (
    "core.attributesfile",
    "core.excludesfile",
    "core.fsmonitor",
    "core.ignorestat",
    "core.sparsecheckout",
    "core.sparsecheckoutcone",
)


class ApexRuntimeError(RuntimeError):
    """Raised when the formal runtime cannot be proven or snapshotted."""


@dataclass(frozen=True)
class RuntimePlan:
    """In-memory source bindings for one canonical runtime manifest."""

    manifest: dict[str, Any]
    roots: tuple[tuple[str, Path], ...]
    system_python: Path

    @property
    def sha256(self) -> str:
        value = self.manifest.get("sha256")
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ApexRuntimeError("runtime plan has no canonical digest")
        return value

    @property
    def external_roots(self) -> tuple[Path, ...]:
        return tuple(
            source
            for role, source in self.roots
            if role.startswith("external/")
        )


@dataclass(frozen=True)
class _GitBinding:
    """Canonical paths used by a fixed set of non-extensible Git builtins."""

    work_tree: Path
    git_dir: Path


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_regular(path: Path) -> tuple[bytes, os.stat_result]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ApexRuntimeError(f"runtime file is unavailable: {path}") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ApexRuntimeError(f"runtime file is not a unique regular file: {path}")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            observed = os.fstat(stream.fileno())
            if (observed.st_dev, observed.st_ino) != (
                metadata.st_dev,
                metadata.st_ino,
            ):
                raise ApexRuntimeError(f"runtime file changed while opening: {path}")
            content = stream.read()
    except OSError as error:
        raise ApexRuntimeError(f"cannot read runtime file: {path}") from error
    return content, metadata


def _canonical_directory(raw: str | Path, *, label: str) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise ApexRuntimeError(f"{label} must be absolute")
    lexical = Path(os.path.abspath(candidate))
    try:
        metadata = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise ApexRuntimeError(f"{label} is unavailable: {candidate}") from error
    if (
        resolved != lexical
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise ApexRuntimeError(f"{label} must be a canonical directory")
    return resolved


def _path_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ApexRuntimeError(f"unsafe runtime relative path: {value!r}")
    return path.as_posix()


def _git_environment() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "HOME": "/nonexistent",
        "XDG_CONFIG_HOME": "/nonexistent",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _git_directory(root: Path) -> Path:
    marker = root / ".git"
    try:
        metadata = marker.lstat()
    except OSError as error:
        raise ApexRuntimeError("APEX_ROOT has no Git metadata") from error
    if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
        return _canonical_directory(marker, label="Apex Git directory")
    content, _opened = _read_regular(marker)
    if len(content) > 4096:
        raise ApexRuntimeError("Apex Git indirection is oversized")
    try:
        line = content.decode("utf-8", "strict").strip()
    except UnicodeError as error:
        raise ApexRuntimeError("Apex Git indirection is not UTF-8") from error
    prefix = "gitdir: "
    if not line.startswith(prefix) or "\n" in line or "\r" in line:
        raise ApexRuntimeError("Apex Git indirection is malformed")
    candidate = Path(line[len(prefix) :])
    if not candidate.is_absolute():
        candidate = root / candidate
    return _canonical_directory(candidate, label="Apex Git directory")


def _git(
    binding: _GitBinding,
    arguments: Sequence[str],
    *,
    binary: bool = False,
) -> bytes | str:
    """Run only fixed raw object/index builtins without checkout/filter paths."""

    if not arguments or arguments[0] not in {
        "cat-file",
        "config",
        "for-each-ref",
        "ls-files",
        "ls-tree",
        "rev-parse",
    }:
        raise ApexRuntimeError("unapproved Git builtin requested")
    try:
        completed = subprocess.run(
            [
                "/usr/bin/git",
                "--no-pager",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.hooksPath=/dev/null",
                f"--git-dir={binding.git_dir}",
                f"--work-tree={binding.work_tree}",
                *arguments,
            ],
            env=_git_environment(),
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ApexRuntimeError(f"trusted Git command failed: {arguments[0]}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-1000:].decode(
            "utf-8", "replace"
        )
        raise ApexRuntimeError(f"trusted Git command failed: {detail.strip()}")
    if binary:
        return completed.stdout
    return completed.stdout.decode("utf-8", "strict").rstrip("\r\n")


def _reject_git_shortcuts(binding: _GitBinding) -> None:
    raw_config = _git(
        binding,
        ["config", "--local", "--null", "--list"],
        binary=True,
    )
    assert isinstance(raw_config, bytes)
    config: dict[str, list[str]] = {}
    for record in raw_config.split(b"\0"):
        if not record:
            continue
        raw_key, separator, raw_value = record.partition(b"\n")
        if not separator:
            raise ApexRuntimeError("Apex local Git config is malformed")
        key = raw_key.decode("utf-8", "strict").lower()
        value = raw_value.decode("utf-8", "strict")
        config.setdefault(key, []).append(value)
    for option in _REJECTED_LOCAL_GIT_OPTIONS:
        values = config.get(option, [])
        forbidden_path_setting = option in {
            "core.attributesfile",
            "core.excludesfile",
        }
        if (forbidden_path_setting and values) or any(
            value.strip().lower() not in {"", "false", "0", "no", "off"}
            for value in values
        ):
            raise ApexRuntimeError(f"formal Apex rejects local Git option {option}")
    if any(
        key == "include.path" or key.startswith("includeif.") for key in config
    ):
        raise ApexRuntimeError("formal Apex rejects local Git config includes")
    git_dir = binding.git_dir
    common_dir = git_dir
    common_marker = git_dir / "commondir"
    if common_marker.exists():
        content, _metadata = _read_regular(common_marker)
        try:
            raw_common = content.decode("utf-8", "strict").strip()
        except UnicodeError as error:
            raise ApexRuntimeError("Apex Git common-dir marker is invalid") from error
        candidate = Path(raw_common)
        if not candidate.is_absolute():
            candidate = git_dir / candidate
        common_dir = _canonical_directory(candidate, label="Apex common Git directory")
    rejected_metadata = (
        common_dir / "info/grafts",
        git_dir / "info/sparse-checkout",
        common_dir / "objects/info/alternates",
        common_dir / "shallow",
    )
    if any(path.exists() for path in rejected_metadata):
        raise ApexRuntimeError("formal Apex rejects non-self-contained Git metadata")
    exclude_path = common_dir / "info/exclude"
    if exclude_path.exists():
        content, _metadata = _read_regular(exclude_path)
        try:
            active_excludes = [
                line
                for line in content.decode("utf-8", "strict").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        except UnicodeError as error:
            raise ApexRuntimeError("Apex Git info/exclude is invalid") from error
        if active_excludes:
            raise ApexRuntimeError("formal Apex rejects Git info/exclude patterns")
    replacement_refs = str(
        _git(binding, ["for-each-ref", "--format=%(refname)", "refs/replace/"])
    )
    if replacement_refs:
        raise ApexRuntimeError("formal Apex rejects Git replacement refs")
    for flag in ("-v", "-f"):
        tagged = _git(binding, ["ls-files", flag, "-z"], binary=True)
        assert isinstance(tagged, bytes)
        for record in tagged.split(b"\0"):
            if not record:
                continue
            tag = chr(record[0])
            if tag == "S" or tag.islower():
                meaning = "skip-worktree/assume-unchanged" if flag == "-v" else "fsmonitor-valid"
                raise ApexRuntimeError(f"formal Apex rejects {meaning} index flags")


def _head_tree(binding: _GitBinding) -> tuple[str, list[dict[str, str]]]:
    top_level = Path(str(_git(binding, ["rev-parse", "--show-toplevel"]))).resolve(
        strict=True
    )
    if top_level != binding.work_tree:
        raise ApexRuntimeError("APEX_ROOT is not the Git top-level")
    commit = str(_git(binding, ["rev-parse", "--verify", "HEAD^{commit}"]))
    if not _SHA1.fullmatch(commit):
        raise ApexRuntimeError("Apex HEAD is not a canonical SHA-1 commit")
    raw = _git(
        binding,
        ["ls-tree", "-rz", "--full-tree", commit],
        binary=True,
    )
    assert isinstance(raw, bytes)
    entries: list[dict[str, str]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        match = _TREE_ENTRY.fullmatch(record)
        if match is None or match.group("kind") != b"blob":
            raise ApexRuntimeError("Apex HEAD contains an unsupported tree entry")
        try:
            relative = match.group("path").decode("utf-8", "strict")
        except UnicodeError as error:
            raise ApexRuntimeError("Apex HEAD path is not UTF-8") from error
        entries.append(
            {
                "path": _safe_relative(relative),
                "mode": match.group("mode").decode("ascii"),
                "oid": match.group("oid").decode("ascii"),
            }
        )
    if not entries or not any(entry["path"] == "main.py" for entry in entries):
        raise ApexRuntimeError("Apex HEAD does not contain main.py")
    return commit, entries


def _index_matches_head(binding: _GitBinding, head: list[dict[str, str]]) -> None:
    raw = _git(binding, ["ls-files", "--stage", "-z"], binary=True)
    assert isinstance(raw, bytes)
    index: dict[str, tuple[str, str]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        fields = metadata.split()
        if not separator or len(fields) != 3 or fields[2] != b"0":
            raise ApexRuntimeError("Apex index contains an unmerged or malformed entry")
        try:
            path = _safe_relative(raw_path.decode("utf-8", "strict"))
        except UnicodeError as error:
            raise ApexRuntimeError("Apex index path is not UTF-8") from error
        index[path] = (fields[0].decode("ascii"), fields[1].decode("ascii"))
    expected = {entry["path"]: (entry["mode"], entry["oid"]) for entry in head}
    if index != expected:
        raise ApexRuntimeError("Apex index differs from HEAD")


def _tracked_file_entry(
    binding: _GitBinding, entry: dict[str, str]
) -> dict[str, Any]:
    relative = entry["path"]
    path = binding.work_tree / relative
    expected = _git(
        binding,
        ["cat-file", "blob", entry["oid"]],
        binary=True,
    )
    assert isinstance(expected, bytes)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ApexRuntimeError(f"tracked Apex path is unavailable: {relative}") from error
    mode = entry["mode"]
    if mode == "120000":
        if not stat.S_ISLNK(metadata.st_mode):
            raise ApexRuntimeError(f"tracked Apex symlink type differs: {relative}")
        target = os.readlink(path)
        if os.fsencode(target) != expected:
            raise ApexRuntimeError(f"tracked Apex symlink differs from HEAD: {relative}")
        if PurePosixPath(target).is_absolute():
            raise ApexRuntimeError(f"tracked Apex symlink is absolute: {relative}")
        lexical = posixpath.normpath(
            posixpath.join(posixpath.dirname(relative), target)
        )
        if lexical == ".." or lexical.startswith("../"):
            raise ApexRuntimeError(f"tracked Apex symlink escapes the checkout: {relative}")
        return {
            "path": relative,
            "type": "symlink",
            "mode": mode,
            "target": target,
            "sha256": _sha256_bytes(expected),
        }
    if mode not in {"100644", "100755"}:
        raise ApexRuntimeError(f"unsupported tracked Apex mode {mode}: {relative}")
    content, opened = _read_regular(path)
    if content != expected:
        raise ApexRuntimeError(f"tracked Apex bytes differ from HEAD: {relative}")
    executable = bool(opened.st_mode & 0o111)
    if executable != (mode == "100755"):
        raise ApexRuntimeError(f"tracked Apex executable mode differs: {relative}")
    return {
        "path": relative,
        "type": "file",
        "mode": mode,
        "size": len(content),
        "sha256": _sha256_bytes(content),
    }


def _head_anchor_recheck(binding: _GitBinding, commit: str) -> None:
    observed = str(
        _git(binding, ["rev-parse", "--verify", "HEAD^{commit}"])
    )
    if observed != commit:
        raise ApexRuntimeError("Apex HEAD changed while planning the runtime")


def _apex_tree_manifest(
    root: Path,
) -> tuple[str, list[dict[str, Any]], str, _GitBinding]:
    binding = _GitBinding(work_tree=root, git_dir=_git_directory(root))
    _reject_git_shortcuts(binding)
    commit, head = _head_tree(binding)
    _index_matches_head(binding, head)
    untracked = _git(
        binding,
        ["ls-files", "--others", "--exclude-standard", "-z"],
        binary=True,
    )
    assert isinstance(untracked, bytes)
    if untracked:
        raise ApexRuntimeError("formal Apex checkout is not clean")
    files = [_tracked_file_entry(binding, entry) for entry in head]
    _head_anchor_recheck(binding, commit)
    return commit, files, _sha256_bytes(b""), binding


def _launcher_identity(apex_root: Path, raw_python: str | Path) -> dict[str, Any]:
    launcher = Path(raw_python).expanduser()
    if not launcher.is_absolute():
        raise ApexRuntimeError("APEX_PYTHON must be absolute")
    launcher = Path(os.path.abspath(launcher))
    expected_alias = apex_root / ".venv"
    try:
        launcher.relative_to(expected_alias)
        alias_metadata = expected_alias.lstat()
        venv_root = expected_alias.resolve(strict=True)
        resolved_python = launcher.resolve(strict=True)
    except (OSError, ValueError) as error:
        raise ApexRuntimeError(
            "APEX_PYTHON must be below APEX_ROOT/.venv"
        ) from error
    if not stat.S_ISDIR(venv_root.lstat().st_mode):
        raise ApexRuntimeError("resolved Apex virtualenv is not a directory")
    python_content, python_metadata = _read_regular(resolved_python)
    if not os.access(resolved_python, os.X_OK):
        raise ApexRuntimeError("resolved Apex Python is not executable")
    chain: list[dict[str, Any]] = []
    for path in (expected_alias, launcher.parent / "python", launcher):
        if any(item.get("path") == str(path) for item in chain):
            continue
        try:
            observed = path.lstat()
        except OSError as error:
            raise ApexRuntimeError(f"Python launcher component is missing: {path}") from error
        item: dict[str, Any] = {
            "path": str(path),
            "mode": stat.S_IMODE(observed.st_mode),
        }
        if stat.S_ISLNK(observed.st_mode):
            item.update({"type": "symlink", "target": os.readlink(path)})
        elif stat.S_ISDIR(observed.st_mode):
            item["type"] = "directory"
        elif stat.S_ISREG(observed.st_mode):
            item.update(
                {
                    "type": "file",
                    "sha256": _sha256_bytes(path.read_bytes()),
                }
            )
        else:
            raise ApexRuntimeError(f"unsafe Python launcher component: {path}")
        chain.append(item)
    return {
        "launcher_path": str(launcher),
        "venv_alias_path": str(expected_alias),
        "venv_alias_type": (
            "symlink" if stat.S_ISLNK(alias_metadata.st_mode) else "directory"
        ),
        "venv_alias_target": (
            os.readlink(expected_alias)
            if stat.S_ISLNK(alias_metadata.st_mode)
            else None
        ),
        "venv_root": str(venv_root),
        "launcher_chain": chain,
        "system_python": {
            "path": str(resolved_python),
            "binding": "formal_docker_image_plus_attempt_receipt_v1",
            "size": len(python_content),
            "sha256": _sha256_bytes(python_content),
            "mode": stat.S_IMODE(python_metadata.st_mode),
            "device": python_metadata.st_dev,
            "inode": python_metadata.st_ino,
        },
    }


def _site_packages(venv_root: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for library in (venv_root / "lib", venv_root / "lib64"):
        if not library.exists():
            continue
        for candidate in sorted(library.glob("python*/site-packages")):
            try:
                resolved = candidate.resolve(strict=True)
            except OSError as error:
                raise ApexRuntimeError(
                    f"virtualenv site-packages is unavailable: {candidate}"
                ) from error
            if not _path_below(resolved, venv_root):
                raise ApexRuntimeError("site-packages escapes the virtualenv")
            if resolved not in candidates:
                candidates.append(resolved)
    if not candidates:
        raise ApexRuntimeError("Apex virtualenv has no site-packages")
    return tuple(candidates)


def _system_site_packages(
    venv_root: Path, venv_sites: Iterable[Path]
) -> tuple[Path, ...]:
    config, _metadata = _read_regular(venv_root / "pyvenv.cfg")
    try:
        settings = {
            key.strip().lower(): value.strip().lower()
            for line in config.decode("utf-8", "strict").splitlines()
            if "=" in line
            for key, value in (line.split("=", 1),)
        }
    except UnicodeError as error:
        raise ApexRuntimeError("Apex pyvenv.cfg is not UTF-8") from error
    enabled = settings.get("include-system-site-packages", "false")
    if enabled not in {"true", "false"}:
        raise ApexRuntimeError("Apex system-site-packages setting is invalid")
    if enabled == "false":
        return ()
    versions = {
        site.parent.name
        for site in venv_sites
        if re.fullmatch(r"python[0-9]+\.[0-9]+", site.parent.name)
    }
    if len(versions) != 1:
        raise ApexRuntimeError("Apex virtualenv Python ABI is ambiguous")
    version = next(iter(versions))
    candidates = (
        Path(f"/usr/local/lib/{version}/dist-packages"),
        Path(f"/usr/local/lib/{version}/site-packages"),
        Path(f"/usr/lib/{version}/dist-packages"),
        Path(f"/usr/lib/{version}/site-packages"),
        Path("/usr/lib/python3/dist-packages"),
    )
    roots: list[Path] = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        resolved = _canonical_directory(candidate, label="system site-packages")
        if resolved not in roots:
            roots.append(resolved)
    if not roots:
        raise ApexRuntimeError("enabled system site-packages are unavailable")
    return tuple(roots)


def _file_url_path(value: Any) -> Path | None:
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        return None
    return Path(unquote(parsed.path))


def _path_literals(path: Path) -> set[Path]:
    content, _metadata = _read_regular(path)
    try:
        tree = ast.parse(content.decode("utf-8", "strict"), filename=str(path))
    except (UnicodeError, SyntaxError) as error:
        raise ApexRuntimeError(f"editable finder is not static UTF-8 Python: {path}") from error
    candidates: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            candidate = Path(node.value)
            if candidate.is_absolute() and candidate.exists():
                candidates.add(candidate)
    return candidates


def _metadata_external_candidates(site_roots: Iterable[Path]) -> set[Path]:
    candidates: set[Path] = set()
    for site_root in site_roots:
        selected: list[Path] = []
        for directory, names, filenames in os.walk(site_root, followlinks=False):
            names[:] = sorted(names)
            for filename in sorted(filenames):
                if (
                    filename.endswith((".pth", ".egg-link"))
                    or filename == "direct_url.json"
                    or filename == "RECORD"
                    or (
                        filename.startswith("__editable__")
                        and filename.endswith("_finder.py")
                    )
                ):
                    selected.append(Path(directory) / filename)
        for path in selected:
            if path.is_symlink():
                raise ApexRuntimeError(f"runtime path metadata is a symlink: {path}")
            if path.suffix in {".pth", ".egg-link"}:
                content, _metadata = _read_regular(path)
                try:
                    lines = content.decode("utf-8", "strict").splitlines()
                except UnicodeError as error:
                    raise ApexRuntimeError(f"non-UTF-8 path metadata: {path}") from error
                for line in lines:
                    value = line.strip()
                    if not value or value.startswith("#") or value.startswith("import "):
                        continue
                    candidate = Path(value)
                    if not candidate.is_absolute():
                        candidate = path.parent / candidate
                    if candidate.exists():
                        candidates.add(candidate)
            elif path.name == "direct_url.json":
                content, _metadata = _read_regular(path)
                try:
                    value = json.loads(content)
                except (UnicodeError, json.JSONDecodeError) as error:
                    raise ApexRuntimeError(f"invalid direct_url.json: {path}") from error
                candidate = _file_url_path(value.get("url") if isinstance(value, dict) else None)
                if candidate is not None and candidate.exists():
                    candidates.add(candidate)
            elif path.name == "RECORD":
                content, _metadata = _read_regular(path)
                try:
                    rows = csv.reader(content.decode("utf-8", "strict").splitlines())
                    record_paths = [row[0] for row in rows if row]
                except (UnicodeError, csv.Error) as error:
                    raise ApexRuntimeError(f"invalid wheel RECORD: {path}") from error
                for value in record_paths:
                    candidate = Path(value)
                    if not candidate.is_absolute():
                        candidate = site_root / candidate
                    if candidate.exists():
                        candidates.add(candidate)
            elif path.name.startswith("__editable__") and path.name.endswith("_finder.py"):
                candidates.update(_path_literals(path))
    return candidates


def _normalized_external_roots(
    candidates: Iterable[Path],
    *,
    apex_root: Path,
    venv_root: Path,
    apex_files: Iterable[dict[str, Any]],
) -> tuple[Path, ...]:
    tracked = {entry["path"] for entry in apex_files}

    def covered_by_apex_tree(path: Path) -> bool:
        if path == apex_root:
            return True
        if not _path_below(path, apex_root):
            return False
        prefix = path.relative_to(apex_root).as_posix().rstrip("/") + "/"
        return any(value == prefix[:-1] or value.startswith(prefix) for value in tracked)

    resolved: list[Path] = []
    for candidate in candidates:
        try:
            path = candidate.resolve(strict=True)
        except OSError as error:
            raise ApexRuntimeError(f"editable runtime root is unavailable: {candidate}") from error
        if path.is_file():
            path = path.parent
        if not path.is_dir():
            raise ApexRuntimeError(f"editable runtime root is not a directory: {path}")
        if covered_by_apex_tree(path) or _path_below(path, venv_root):
            continue
        if path in {Path("/"), Path("/tmp"), Path("/home"), Path("/usr")}:
            raise ApexRuntimeError(f"editable runtime root is too broad: {path}")
        if any(_path_below(path, existing) for existing in resolved):
            continue
        resolved = [existing for existing in resolved if not _path_below(existing, path)]
        resolved.append(path)
    return tuple(sorted(resolved, key=str))


def _managed_dependency_candidates(
    apex_root: Path, apex_files: Iterable[dict[str, Any]]
) -> set[Path]:
    relative = "scripts/dependencies.lock.json"
    tracked = {entry["path"]: entry for entry in apex_files}
    evidence = tracked.get(relative)
    if evidence is None:
        return set()
    content, _metadata = _read_regular(apex_root / relative)
    if _sha256_bytes(content) != evidence.get("sha256"):
        raise ApexRuntimeError("Apex dependency lock changed during planning")
    try:
        lock = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ApexRuntimeError("Apex dependency lock is invalid") from error
    dependencies = lock.get("dependencies") if isinstance(lock, dict) else None
    if not isinstance(dependencies, dict):
        raise ApexRuntimeError("Apex dependency lock has no dependency map")
    candidates: set[Path] = set()
    for name, raw in sorted(dependencies.items()):
        checkout = raw.get("managed_checkout") if isinstance(raw, dict) else None
        if (
            not isinstance(name, str)
            or not isinstance(checkout, str)
            or _safe_relative(checkout) != checkout
            or "/" in checkout
        ):
            raise ApexRuntimeError("Apex managed dependency locator is invalid")
        candidate = apex_root / ".cache/apex-dependencies" / checkout
        if candidate.exists():
            candidates.add(candidate)
    return candidates


def _system_directory_link_candidates(roots: Iterable[Path]) -> set[Path]:
    discovered: set[Path] = set()
    pending = list(roots)
    scanned: set[Path] = set()
    while pending:
        root = pending.pop()
        if root in scanned:
            continue
        scanned.add(root)
        for directory, names, filenames in os.walk(root, followlinks=False):
            for name in [*sorted(names), *sorted(filenames)]:
                path = Path(directory) / name
                if not path.is_symlink():
                    continue
                try:
                    target = path.resolve(strict=True)
                except OSError as error:
                    raise ApexRuntimeError(f"system runtime symlink is broken: {path}") from error
                if not target.is_dir() or _path_below(target, root):
                    continue
                if not any(
                    _path_below(target, prefix)
                    for prefix in (Path("/usr"), Path("/lib"), Path("/lib64"), Path("/var/lib"))
                ):
                    raise ApexRuntimeError(
                        f"system directory symlink has an unsafe target: {path}"
                    )
                if target in {Path("/usr"), Path("/usr/share"), Path("/var/lib")}:
                    raise ApexRuntimeError(
                        f"system directory symlink target is too broad: {path}"
                    )
                if target not in discovered:
                    discovered.add(target)
                    pending.append(target)
    return discovered


def _discover_runtime_roots(
    root: Path,
    venv_root: Path,
    sites: Iterable[Path],
    system_sites: Iterable[Path],
    apex_files: list[dict[str, Any]],
) -> tuple[Path, ...]:
    candidates = _metadata_external_candidates(sites)
    candidates.update(system_sites)
    candidates.update(_system_directory_link_candidates(system_sites))
    candidates.update(_managed_dependency_candidates(root, apex_files))
    return _normalized_external_roots(
        candidates,
        apex_root=root,
        venv_root=venv_root,
        apex_files=apex_files,
    )


def discover_external_roots(
    apex_root: str | Path, apex_python: str | Path
) -> tuple[Path, ...]:
    """Statically discover every path-bearing editable installation artifact."""

    root = _canonical_directory(apex_root, label="APEX_ROOT")
    launcher = _launcher_identity(root, apex_python)
    venv_root = Path(launcher["venv_root"])
    sites = _site_packages(venv_root)
    system_sites = _system_site_packages(venv_root, sites)
    commit, apex_files, _status, binding = _apex_tree_manifest(root)
    external = _discover_runtime_roots(
        root, venv_root, sites, system_sites, apex_files
    )
    _head_anchor_recheck(binding, commit)
    return external


def _tree_entry(
    path: Path,
    relative: str,
    *,
    allow_system_bindings: bool,
    destination: str,
    directory_bindings: tuple[tuple[Path, str], ...],
) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ApexRuntimeError(f"cannot inspect runtime tree path: {path}") from error
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISDIR(metadata.st_mode):
        return {"path": relative, "type": "directory", "mode": mode}
    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(path)
        lexical_target = posixpath.normpath(
            posixpath.join(posixpath.dirname(relative), target)
        )
        escapes_root = lexical_target == ".." or lexical_target.startswith("../")
        if PurePosixPath(target).is_absolute() or escapes_root:
            try:
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise ApexRuntimeError(f"runtime symlink is broken: {path}") from error
            system_target = any(
                _path_below(resolved, prefix)
                for prefix in (
                    Path("/usr"),
                    Path("/lib"),
                    Path("/lib64"),
                    Path("/var/lib"),
                )
            )
            if not allow_system_bindings or not system_target:
                raise ApexRuntimeError(f"absolute runtime symlink is forbidden: {path}")
            if resolved.is_dir():
                target_destination: PurePosixPath | None = None
                for source_root, output_root in directory_bindings:
                    if not _path_below(resolved, source_root):
                        continue
                    suffix = resolved.relative_to(source_root)
                    target_destination = PurePosixPath(output_root) / PurePosixPath(
                        suffix.as_posix()
                    )
                    break
                if target_destination is None:
                    raise ApexRuntimeError(
                        f"system directory binding is not in the runtime closure: {path}"
                    )
                output_parent = (
                    PurePosixPath(destination) / PurePosixPath(relative)
                ).parent
                materialized_target = posixpath.relpath(
                    target_destination.as_posix(), output_parent.as_posix()
                )
                opened = resolved.lstat()
                return {
                    "path": relative,
                    "type": "directory_link_binding",
                    "mode": mode,
                    "target": target,
                    "target_sha256": _sha256_bytes(os.fsencode(target)),
                    "resolved_path": str(resolved),
                    "resolved_device": opened.st_dev,
                    "resolved_inode": opened.st_ino,
                    "resolved_mode": stat.S_IMODE(opened.st_mode),
                    "materialized_target": materialized_target,
                    "materialized_target_sha256": _sha256_bytes(
                        os.fsencode(materialized_target)
                    ),
                }
            content, opened = _read_regular(resolved)
            materialized_mode = 0o555 if opened.st_mode & 0o111 else 0o444
            return {
                "path": relative,
                "type": "system_file_binding",
                "mode": mode,
                "target": target,
                "target_sha256": _sha256_bytes(os.fsencode(target)),
                "resolved_path": str(resolved),
                "resolved_device": opened.st_dev,
                "resolved_inode": opened.st_ino,
                "resolved_mode": stat.S_IMODE(opened.st_mode),
                "size": len(content),
                "sha256": _sha256_bytes(content),
                "materialized_mode": materialized_mode,
            }
        return {
            "path": relative,
            "type": "symlink",
            "mode": mode,
            "target": target,
            "sha256": _sha256_bytes(os.fsencode(target)),
        }
    content, opened = _read_regular(path)
    return {
        "path": relative,
        "type": "file",
        "mode": stat.S_IMODE(opened.st_mode),
        "size": len(content),
        "sha256": _sha256_bytes(content),
    }


def _complete_tree_manifest(
    root: Path,
    *,
    destination: str,
    directory_bindings: tuple[tuple[Path, str], ...],
    exclude_vcs: bool,
    allow_system_bindings: bool = False,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError as error:
            raise ApexRuntimeError(f"cannot enumerate runtime tree: {directory}") from error
        for path in children:
            relative = path.relative_to(root).as_posix()
            if exclude_vcs and any(part in _IGNORED_EXTERNAL_PARTS for part in Path(relative).parts):
                continue
            entry = _tree_entry(
                path,
                _safe_relative(relative),
                allow_system_bindings=allow_system_bindings,
                destination=destination,
                directory_bindings=directory_bindings,
            )
            if entry["type"] == "symlink":
                target = entry["target"]
                lexical = posixpath.normpath(
                    posixpath.join(posixpath.dirname(relative), target)
                )
                if lexical == ".." or lexical.startswith("../"):
                    raise ApexRuntimeError(f"runtime symlink escapes its root: {path}")
                entry["resolved_target_class"] = "internal"
            entries.append(entry)
            if entry["type"] == "directory":
                pending.append(path)
    entries.sort(key=lambda entry: entry["path"])
    if not entries:
        raise ApexRuntimeError(f"runtime tree is empty: {root}")
    available = {entry["path"] for entry in entries}
    available.update(
        entry["path"]
        for entry in entries
        if entry["type"] == "directory"
    )
    for entry in entries:
        if entry["type"] != "symlink":
            continue
        target = posixpath.normpath(
            posixpath.join(posixpath.dirname(entry["path"]), entry["target"])
        )
        if target not in available:
            raise ApexRuntimeError(
                f"runtime symlink target is not materialized: {entry['path']}"
            )
    return entries


def _root_identity(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ApexRuntimeError(f"runtime root is unavailable: {path}") from error
    return {
        "path": str(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": stat.S_IMODE(metadata.st_mode),
    }


def _declared_roots_match(
    discovered: tuple[Path, ...], declared_roots: Iterable[str | Path] | None
) -> None:
    if declared_roots is None:
        return
    declared_values = tuple(declared_roots)
    declared = tuple(
        sorted(
            (
                _canonical_directory(path, label="declared Apex external root")
                for path in declared_values
            ),
            key=str,
        )
    )
    if declared != discovered:
        raise ApexRuntimeError(
            "declared Apex external roots differ from installed editable metadata"
        )


def _manifest_root(
    *, role: str, source: Path, files: list[dict[str, Any]], destination: str
) -> dict[str, Any]:
    material = {
        "role": role,
        "source": _root_identity(source),
        "destination": destination,
        "files": files,
    }
    return {**material, "sha256": _canonical_digest(material)}


def _native_inventory(manifests: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for root_manifest in manifests:
        role = root_manifest["role"]
        for entry in root_manifest["files"]:
            path = entry["path"]
            name = PurePosixPath(path).name
            native_name = (
                name.endswith((".a", ".dylib", ".dll", ".pyd", ".so"))
                or ".so." in name
                or entry["type"] == "system_file_binding"
            )
            if not native_name or entry["type"] not in {
                "file",
                "system_file_binding",
            }:
                continue
            records.append(
                {
                    "role": role,
                    "path": path,
                    "type": entry["type"],
                    "size": entry["size"],
                    "sha256": entry["sha256"],
                    "materialized_mode": (
                        entry.get("materialized_mode")
                        if entry["type"] == "system_file_binding"
                        else entry["mode"]
                    ),
                }
            )
    records.sort(key=lambda value: (value["role"], value["path"]))
    return {
        "policy_id": "exact_native_file_inventory_v1",
        "count": len(records),
        "records": records,
        "sha256": _canonical_digest(records),
        "dynamic_loader_binding": "formal_container_image_plus_mount_receipt",
    }


def plan_runtime(
    apex_root: str | Path,
    apex_python: str | Path,
    *,
    declared_roots: Iterable[str | Path] | None = None,
) -> RuntimePlan:
    """Validate and inventory the complete source-side runtime closure."""

    root = _canonical_directory(apex_root, label="APEX_ROOT")
    launcher = _launcher_identity(root, apex_python)
    venv_root = Path(launcher["venv_root"])
    site_roots = _site_packages(venv_root)
    system_site_roots = _system_site_packages(venv_root, site_roots)
    commit, apex_files, status_digest, binding = _apex_tree_manifest(root)
    external = _discover_runtime_roots(
        root,
        venv_root,
        site_roots,
        system_site_roots,
        apex_files,
    )
    _declared_roots_match(external, declared_roots)
    roots: list[tuple[str, Path]] = [("apex", root), ("venv", venv_root)]
    roots.extend((f"external/{index:03d}", path) for index, path in enumerate(external))
    directory_bindings = (
        (venv_root, "venv"),
        *(
            (path, f"external/{index:03d}")
            for index, path in enumerate(external)
        ),
    )
    manifests = [
        _manifest_root(
            role="apex",
            source=root,
            files=apex_files,
            destination="repo",
        ),
        _manifest_root(
            role="venv",
            source=venv_root,
            files=_complete_tree_manifest(
                venv_root,
                destination="venv",
                directory_bindings=directory_bindings,
                exclude_vcs=False,
                allow_system_bindings=True,
            ),
            destination="venv",
        ),
    ]
    for index, path in enumerate(external):
        manifests.append(
            _manifest_root(
                role=f"external/{index:03d}",
                source=path,
                files=_complete_tree_manifest(
                    path,
                    destination=f"external/{index:03d}",
                    directory_bindings=directory_bindings,
                    exclude_vcs=True,
                    allow_system_bindings=_path_below(path, Path("/usr")),
                ),
                destination=f"external/{index:03d}",
            )
        )
    site_destinations = [
        str(Path("venv") / site.relative_to(venv_root)) for site in site_roots
    ]
    _head_anchor_recheck(binding, commit)
    material: dict[str, Any] = {
        "schema": RUNTIME_MANIFEST_SCHEMA,
        "policy_id": RUNTIME_POLICY_ID,
        "git": {
            "commit": commit,
            "dirty": False,
            "status_sha256": status_digest,
            "index_shortcuts_rejected": True,
            "git_environment_sanitized": True,
            "raw_object_reads_only": True,
            "object_anchor_rechecked": True,
        },
        "launcher": launcher,
        "roots": manifests,
        "native_closure": _native_inventory(manifests),
        "execution": {
            "interpreter": RUNTIME_WRAPPER_NAME,
            "underlying_interpreter": "venv/bin/python",
            "flags": ["-I", "-S", "-u"],
            "bootstrap": RUNTIME_BOOTSTRAP_NAME,
            "bootstrap_policy_id": RUNTIME_BOOTSTRAP_POLICY_ID,
            "bootstrap_sha256": RUNTIME_BOOTSTRAP_SHA256,
            "wrapper_policy_id": RUNTIME_WRAPPER_POLICY_ID,
            "wrapper_sha256": RUNTIME_WRAPPER_SHA256,
            "wrapper_aliases": [
                f"sealed-bin/{alias}" for alias in RUNTIME_WRAPPER_ALIASES
            ],
            "entrypoint": "repo/main.py",
            "pythonpath": [
                "repo/src",
                *site_destinations,
                *(f"external/{index:03d}" for index in range(len(external))),
            ],
            "site_hook_policy": {
                "primary_invocation": "forced_isolated_no_site",
                "python_alias_children": "forced_isolated_no_site",
                "sys_executable_rebound_to_wrapper": True,
                "pth_execution_via_contract": False,
                "sitecustomize_execution_via_contract": False,
                "raw_interpreter_is_not_an_execution_contract": True,
            },
            "no_live_interpreter_fallback": True,
        },
        "immutability": {
            "required_for_execution": True,
            "receipt_schema": RUNTIME_IMMUTABLE_MOUNT_SCHEMA,
            "receipt_policy_id": RUNTIME_IMMUTABLE_MOUNT_POLICY_ID,
            "image_input_schema": RUNTIME_IMAGE_INPUT_SCHEMA,
            "host_mode_bits_are_evidence_only": True,
            "runtime_service_evidence_schema": ENGINE_SERVICE_SCHEMA,
            "runtime_engine_evidence_schema": ENGINE_EVIDENCE_SCHEMA,
            "host_access_policy_schema": HOST_ACCESS_POLICY_SCHEMA,
            "requested_mount_options": [
                "ro", "nodev", "nosuid", "default_permissions",
                "allow_other", "subtype=squashfuse",
            ],
        },
        "excluded_external_directories": sorted(_IGNORED_EXTERNAL_PARTS),
    }
    manifest = {**material, "sha256": _canonical_digest(material)}
    return RuntimePlan(
        manifest=manifest,
        roots=tuple(roots),
        system_python=Path(launcher["system_python"]["path"]),
    )


def _manifest_material(manifest: dict[str, Any]) -> dict[str, Any]:
    if set(manifest).issuperset({"schema", "policy_id", "roots", "execution", "sha256"}):
        material = dict(manifest)
        observed = material.pop("sha256", None)
        if (
            manifest.get("schema") != RUNTIME_MANIFEST_SCHEMA
            or manifest.get("policy_id") != RUNTIME_POLICY_ID
            or not isinstance(observed, str)
            or not _SHA256.fullmatch(observed)
            or _canonical_digest(material) != observed
        ):
            raise ApexRuntimeError("runtime manifest digest is invalid")
        return material
    raise ApexRuntimeError("runtime manifest shape is invalid")


def _canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    _manifest_material(manifest)
    return (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _root_file_map(root_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if set(root_manifest) != {
        "role",
        "source",
        "destination",
        "files",
        "sha256",
    }:
        raise ApexRuntimeError("runtime root manifest shape is invalid")
    material = dict(root_manifest)
    observed = material.pop("sha256")
    if not isinstance(observed, str) or _canonical_digest(material) != observed:
        raise ApexRuntimeError("runtime root manifest digest is invalid")
    destination = root_manifest.get("destination")
    role = root_manifest.get("role")
    source = root_manifest.get("source")
    files = root_manifest.get("files")
    if (
        not isinstance(destination, str)
        or _safe_relative(destination) != destination
        or not isinstance(role, str)
        or not isinstance(source, dict)
        or set(source) != {"path", "device", "inode", "mode"}
        or not isinstance(files, list)
    ):
        raise ApexRuntimeError("runtime root manifest fields are invalid")
    mapped: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ApexRuntimeError("runtime file manifest is invalid")
        relative = _safe_relative(entry["path"])
        if relative in mapped:
            raise ApexRuntimeError("runtime file manifest has duplicate paths")
        kind = entry.get("type")
        required = {"path", "type", "mode"}
        if kind == "directory":
            if set(entry) != required:
                raise ApexRuntimeError("runtime directory entry is invalid")
        elif kind == "file":
            if set(entry) != required | {"size", "sha256"}:
                raise ApexRuntimeError("runtime file entry is invalid")
            if (
                type(entry["size"]) is not int
                or entry["size"] < 0
                or not isinstance(entry["sha256"], str)
                or not _SHA256.fullmatch(entry["sha256"])
            ):
                raise ApexRuntimeError("runtime file evidence is invalid")
        elif kind == "symlink":
            permitted = required | {
                "target",
                "sha256",
                "resolved_target_class",
            }
            if (
                not required | {"target", "sha256"} <= set(entry)
                or not set(entry) <= permitted
                or not isinstance(entry["target"], str)
                or not isinstance(entry["sha256"], str)
                or entry["sha256"] != _sha256_bytes(os.fsencode(entry["target"]))
            ):
                raise ApexRuntimeError("runtime symlink evidence is invalid")
            if PurePosixPath(entry["target"]).is_absolute():
                raise ApexRuntimeError("absolute runtime symlink is invalid")
        elif kind == "system_file_binding":
            required_binding = required | {
                "target",
                "target_sha256",
                "resolved_path",
                "resolved_device",
                "resolved_inode",
                "resolved_mode",
                "size",
                "sha256",
                "materialized_mode",
            }
            if (
                set(entry) != required_binding
                or not isinstance(entry["target"], str)
                or entry["target_sha256"]
                != _sha256_bytes(os.fsencode(entry["target"]))
                or not isinstance(entry["resolved_path"], str)
                or not Path(entry["resolved_path"]).is_absolute()
                or type(entry["resolved_device"]) is not int
                or type(entry["resolved_inode"]) is not int
                or type(entry["resolved_mode"]) is not int
                or type(entry["size"]) is not int
                or entry["size"] < 0
                or not isinstance(entry["sha256"], str)
                or not _SHA256.fullmatch(entry["sha256"])
                or entry["materialized_mode"] not in {0o444, 0o555}
            ):
                raise ApexRuntimeError("system runtime binding evidence is invalid")
        elif kind == "directory_link_binding":
            required_binding = required | {
                "target",
                "target_sha256",
                "resolved_path",
                "resolved_device",
                "resolved_inode",
                "resolved_mode",
                "materialized_target",
                "materialized_target_sha256",
            }
            if (
                set(entry) != required_binding
                or not isinstance(entry["target"], str)
                or entry["target_sha256"]
                != _sha256_bytes(os.fsencode(entry["target"]))
                or not isinstance(entry["resolved_path"], str)
                or not Path(entry["resolved_path"]).is_absolute()
                or type(entry["resolved_device"]) is not int
                or type(entry["resolved_inode"]) is not int
                or type(entry["resolved_mode"]) is not int
                or not isinstance(entry["materialized_target"], str)
                or PurePosixPath(entry["materialized_target"]).is_absolute()
                or entry["materialized_target_sha256"]
                != _sha256_bytes(os.fsencode(entry["materialized_target"]))
            ):
                raise ApexRuntimeError("system directory binding evidence is invalid")
        else:
            raise ApexRuntimeError("runtime file type is unsupported")
        if type(entry["mode"]) is not int and not (
            role == "apex"
            and isinstance(entry["mode"], str)
            and entry["mode"] in {"100644", "100755", "120000"}
        ):
            raise ApexRuntimeError("runtime source mode is invalid")
        mapped[relative] = entry
    if not mapped:
        raise ApexRuntimeError("runtime root manifest is empty")
    return mapped


def _pinned_root(source: Path, identity: dict[str, Any]) -> int:
    try:
        descriptor = os.open(
            source,
            os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise ApexRuntimeError(f"cannot pin runtime source root: {source}") from error
    expected = (
        identity.get("device"),
        identity.get("inode"),
        identity.get("mode"),
    )
    observed = (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IMODE(metadata.st_mode),
    )
    if expected != observed or not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise ApexRuntimeError(f"runtime source root identity changed: {source}")
    return descriptor


def _open_parent_at(root_fd: int, relative: str) -> tuple[int, str]:
    parts = PurePosixPath(_safe_relative(relative)).parts
    current = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            following = os.open(
                part,
                os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=current,
            )
            os.close(current)
            current = following
        return current, parts[-1]
    except OSError as error:
        os.close(current)
        raise ApexRuntimeError(f"runtime source path changed: {relative}") from error


def _read_manifest_entry_at(
    root_fd: int, relative: str, entry: dict[str, Any]
) -> bytes | str | None:
    parent_fd, name = _open_parent_at(root_fd, relative)
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        kind = entry["type"]
        if kind == "directory":
            if not stat.S_ISDIR(metadata.st_mode):
                raise ApexRuntimeError(f"runtime source directory changed: {relative}")
            if stat.S_IMODE(metadata.st_mode) != entry["mode"]:
                raise ApexRuntimeError(f"runtime source directory mode changed: {relative}")
            return None
        if kind == "symlink":
            if not stat.S_ISLNK(metadata.st_mode):
                raise ApexRuntimeError(f"runtime source symlink changed: {relative}")
            target = os.readlink(name, dir_fd=parent_fd)
            if (
                target != entry["target"]
                or (
                    isinstance(entry["mode"], int)
                    and stat.S_IMODE(metadata.st_mode) != entry["mode"]
                )
            ):
                raise ApexRuntimeError(f"runtime source symlink changed: {relative}")
            return target
        if kind == "directory_link_binding":
            if not stat.S_ISLNK(metadata.st_mode):
                raise ApexRuntimeError(
                    f"runtime directory binding changed type: {relative}"
                )
            target = os.readlink(name, dir_fd=parent_fd)
            resolved = Path(entry["resolved_path"])
            try:
                opened = resolved.lstat()
            except OSError as error:
                raise ApexRuntimeError(
                    f"runtime directory binding disappeared: {relative}"
                ) from error
            if (
                target != entry["target"]
                or stat.S_IMODE(metadata.st_mode) != entry["mode"]
                or not stat.S_ISDIR(opened.st_mode)
                or opened.st_dev != entry["resolved_device"]
                or opened.st_ino != entry["resolved_inode"]
                or stat.S_IMODE(opened.st_mode) != entry["resolved_mode"]
            ):
                raise ApexRuntimeError(
                    f"runtime directory binding changed: {relative}"
                )
            return entry["materialized_target"]
        if kind == "system_file_binding":
            if not stat.S_ISLNK(metadata.st_mode):
                raise ApexRuntimeError(
                    f"runtime system binding changed type: {relative}"
                )
            target = os.readlink(name, dir_fd=parent_fd)
            if (
                target != entry["target"]
                or stat.S_IMODE(metadata.st_mode) != entry["mode"]
            ):
                raise ApexRuntimeError(
                    f"runtime system binding changed: {relative}"
                )
            resolved = Path(entry["resolved_path"])
            content, opened = _read_regular(resolved)
            if (
                opened.st_dev != entry["resolved_device"]
                or opened.st_ino != entry["resolved_inode"]
                or stat.S_IMODE(opened.st_mode) != entry["resolved_mode"]
                or len(content) != entry["size"]
                or _sha256_bytes(content) != entry["sha256"]
            ):
                raise ApexRuntimeError(
                    f"runtime system binding target changed: {relative}"
                )
            return content
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ApexRuntimeError(f"runtime source file changed: {relative}")
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise ApexRuntimeError(f"runtime source file raced: {relative}")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                content = stream.read()
        finally:
            os.close(descriptor)
        if (
            len(content) != entry["size"]
            or _sha256_bytes(content) != entry["sha256"]
        ):
            raise ApexRuntimeError(f"runtime source bytes changed: {relative}")
        source_mode = stat.S_IMODE(opened.st_mode)
        expected_mode = entry["mode"]
        if isinstance(expected_mode, str):
            mode_matches = bool(source_mode & 0o111) == (expected_mode == "100755")
        else:
            mode_matches = source_mode == expected_mode
        if not mode_matches:
            raise ApexRuntimeError(f"runtime source mode changed: {relative}")
        return content
    except OSError as error:
        raise ApexRuntimeError(f"cannot read runtime source: {relative}") from error
    finally:
        os.close(parent_fd)


def _snapshot_file_mode(entry: dict[str, Any]) -> int:
    if entry["type"] == "system_file_binding":
        return entry["materialized_mode"]
    raw = entry["mode"]
    executable = raw == "100755" or (isinstance(raw, int) and bool(raw & 0o111))
    return 0o555 if executable else 0o444


def _ensure_output_parent(root: Path, relative: str) -> Path:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.is_symlink():
        raise ApexRuntimeError("runtime snapshot output contains a symlink")
    return destination


def _copy_manifest_root(
    source: Path,
    root_manifest: dict[str, Any],
    snapshot: Path,
) -> None:
    entries = _root_file_map(root_manifest)
    identity = root_manifest["source"]
    if identity.get("path") != str(source):
        raise ApexRuntimeError("runtime plan root binding differs from manifest")
    source_fd = _pinned_root(source, identity)
    destination_root = snapshot / root_manifest["destination"]
    destination_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        directories = sorted(
            (
                (relative, entry)
                for relative, entry in entries.items()
                if entry["type"] == "directory"
            ),
            key=lambda item: (len(PurePosixPath(item[0]).parts), item[0]),
        )
        for relative, entry in directories:
            _read_manifest_entry_at(source_fd, relative, entry)
            (destination_root / relative).mkdir(parents=True, exist_ok=True, mode=0o700)
        for relative, entry in sorted(entries.items()):
            if entry["type"] == "directory":
                continue
            value = _read_manifest_entry_at(source_fd, relative, entry)
            destination = _ensure_output_parent(destination_root, relative)
            if entry["type"] in {"symlink", "directory_link_binding"}:
                assert isinstance(value, str)
                os.symlink(value, destination)
            else:
                assert isinstance(value, bytes)
                descriptor = os.open(
                    destination,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_NOFOLLOW
                    | os.O_CLOEXEC,
                    0o600,
                )
                try:
                    with os.fdopen(descriptor, "wb", closefd=False) as stream:
                        stream.write(value)
                        stream.flush()
                        os.fsync(descriptor)
                    os.fchmod(descriptor, _snapshot_file_mode(entry))
                finally:
                    os.close(descriptor)
        for relative, entry in sorted(entries.items()):
            if entry["type"] not in {"symlink", "directory_link_binding"}:
                continue
            output = destination_root / relative
            if PurePosixPath(os.readlink(output)).is_absolute():
                raise ApexRuntimeError("materialized runtime symlink is absolute")
            if entry["type"] == "directory_link_binding":
                lexical = Path(os.path.abspath(output.parent / os.readlink(output)))
                if not _path_below(lexical, snapshot):
                    raise ApexRuntimeError(
                        f"materialized directory binding escapes snapshot: {relative}"
                    )
                continue
            try:
                resolved = output.resolve(strict=True)
            except OSError as error:
                raise ApexRuntimeError(
                    f"materialized runtime symlink is broken: {relative}"
                ) from error
            if not _path_below(resolved, destination_root):
                raise ApexRuntimeError(
                    f"materialized runtime symlink escapes its root: {relative}"
                )
        for directory in sorted(
            (path for path in destination_root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        destination_root.chmod(0o555)
        after = os.fstat(source_fd)
        if (
            after.st_dev,
            after.st_ino,
            stat.S_IMODE(after.st_mode),
        ) != (
            identity["device"],
            identity["inode"],
            identity["mode"],
        ):
            raise ApexRuntimeError("runtime source root changed during materialization")
    finally:
        os.close(source_fd)


def materialize_runtime(plan: RuntimePlan, snapshot_parent: str | Path) -> Path:
    """Create and seal ``<snapshot_parent>/<manifest sha256>`` atomically."""

    _manifest_material(plan.manifest)
    parent = Path(snapshot_parent).expanduser()
    if not parent.is_absolute():
        raise ApexRuntimeError("runtime snapshot parent must be absolute")
    parent = Path(os.path.abspath(parent))
    if parent.exists():
        parent = _canonical_directory(parent, label="runtime snapshot parent")
        final = parent / plan.sha256
        if final.is_dir():
            verify_runtime_snapshot(final, plan.sha256)
            return final
        if final.exists() or final.is_symlink():
            raise ApexRuntimeError("runtime snapshot digest path is unsafe")
    else:
        parent.mkdir(mode=0o700, parents=False, exist_ok=False)
        parent = _canonical_directory(parent, label="runtime snapshot parent")
    for _role, source in plan.roots:
        if _path_below(parent, source) or _path_below(source, parent):
            raise ApexRuntimeError("runtime snapshot overlaps a source root")
    final = parent / plan.sha256
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=parent))
    try:
        manifests = plan.manifest.get("roots")
        if not isinstance(manifests, list) or len(manifests) != len(plan.roots):
            raise ApexRuntimeError("runtime plan root bindings are incomplete")
        for (role, source), root_manifest in zip(plan.roots, manifests, strict=True):
            if not isinstance(root_manifest, dict) or root_manifest.get("role") != role:
                raise ApexRuntimeError("runtime plan root order is invalid")
            _copy_manifest_root(source, root_manifest, temporary)
        manifest_path = temporary / "runtime_manifest.json"
        content = _canonical_manifest_bytes(plan.manifest)
        descriptor = os.open(
            manifest_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(content)
                stream.flush()
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
        finally:
            os.close(descriptor)
        bootstrap_path = temporary / RUNTIME_BOOTSTRAP_NAME
        descriptor = os.open(
            bootstrap_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(RUNTIME_BOOTSTRAP)
                stream.flush()
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
        finally:
            os.close(descriptor)
        wrapper_directory = temporary / "sealed-bin"
        wrapper_directory.mkdir(mode=0o700)
        for alias in RUNTIME_WRAPPER_ALIASES:
            wrapper_path = wrapper_directory / alias
            descriptor = os.open(
                wrapper_path,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | os.O_CLOEXEC,
                0o600,
            )
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as stream:
                    stream.write(RUNTIME_WRAPPER)
                    stream.flush()
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o555)
            finally:
                os.close(descriptor)
        for directory in sorted(
            (path for path in temporary.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        temporary.chmod(0o555)
        try:
            os.rename(temporary, final)
        except OSError:
            if not final.is_dir():
                raise
            verify_runtime_snapshot(final, plan.sha256)
            for path in sorted(
                (item for item in temporary.rglob("*") if item.is_dir()),
                key=lambda item: len(item.parts),
                reverse=True,
            ):
                path.chmod(0o700)
            temporary.chmod(0o700)
            shutil.rmtree(temporary)
        verify_runtime_snapshot(final, plan.sha256)
        return final
    except BaseException:
        if temporary.exists():
            for path in sorted(
                (item for item in temporary.rglob("*") if item.is_dir()),
                key=lambda item: len(item.parts),
                reverse=True,
            ):
                path.chmod(0o700)
            temporary.chmod(0o700)
            shutil.rmtree(temporary)
        raise


def _expected_snapshot_paths(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {
        "runtime_manifest.json": {"type": "manifest", "mode": 0o444},
        RUNTIME_BOOTSTRAP_NAME: {
            "type": "bootstrap",
            "mode": 0o444,
            "size": len(RUNTIME_BOOTSTRAP),
            "sha256": RUNTIME_BOOTSTRAP_SHA256,
        },
        "sealed-bin": {"type": "directory", "mode": 0o555},
    }
    for alias in RUNTIME_WRAPPER_ALIASES:
        expected[f"sealed-bin/{alias}"] = {
            "type": "wrapper",
            "mode": 0o555,
            "size": len(RUNTIME_WRAPPER),
            "sha256": RUNTIME_WRAPPER_SHA256,
        }
    roots = manifest.get("roots")
    if not isinstance(roots, list):
        raise ApexRuntimeError("runtime manifest roots are invalid")
    for root_manifest in roots:
        if not isinstance(root_manifest, dict):
            raise ApexRuntimeError("runtime root manifest is invalid")
        entries = _root_file_map(root_manifest)
        destination = root_manifest["destination"]
        expected[destination] = {"type": "directory", "mode": 0o555}
        destination_parent = PurePosixPath(destination).parent
        while str(destination_parent) != ".":
            expected.setdefault(
                destination_parent.as_posix(),
                {"type": "directory", "mode": 0o555},
            )
            destination_parent = destination_parent.parent
        for relative, entry in entries.items():
            output = str(PurePosixPath(destination) / PurePosixPath(relative))
            expected[output] = (
                {
                    "path": entry["path"],
                    "type": "symlink",
                    "mode": entry["mode"],
                    "target": entry["materialized_target"],
                    "sha256": entry["materialized_target_sha256"],
                }
                if entry["type"] == "directory_link_binding"
                else entry
            )
            parent = PurePosixPath(output).parent
            while str(parent) not in {".", destination}:
                expected.setdefault(
                    parent.as_posix(), {"type": "directory", "mode": 0o555}
                )
                parent = parent.parent
    return expected


def verify_runtime_snapshot(
    root: str | Path, expected_manifest_sha256: str
) -> dict[str, Any]:
    """Recompute every snapshotted byte, link, mode, and the manifest digest."""

    if not isinstance(expected_manifest_sha256, str) or not _SHA256.fullmatch(
        expected_manifest_sha256
    ):
        raise ApexRuntimeError("expected runtime manifest digest is invalid")
    snapshot = _canonical_directory(root, label="runtime snapshot")
    if snapshot.name != expected_manifest_sha256:
        raise ApexRuntimeError("runtime snapshot directory is not digest-addressed")
    if stat.S_IMODE(snapshot.stat().st_mode) != 0o555:
        raise ApexRuntimeError("runtime snapshot root is not sealed")
    manifest_path = snapshot / "runtime_manifest.json"
    content, metadata = _read_regular(manifest_path)
    if stat.S_IMODE(metadata.st_mode) != 0o444 or len(content) > 64 * 1024 * 1024:
        raise ApexRuntimeError("runtime snapshot manifest is not sealed")
    try:
        manifest = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ApexRuntimeError("runtime snapshot manifest is invalid JSON") from error
    if not isinstance(manifest, dict):
        raise ApexRuntimeError("runtime snapshot manifest is not an object")
    if content != _canonical_manifest_bytes(manifest):
        raise ApexRuntimeError("runtime snapshot manifest is not canonical")
    if manifest.get("sha256") != expected_manifest_sha256:
        raise ApexRuntimeError("runtime snapshot manifest digest differs")
    expected = _expected_snapshot_paths(manifest)
    observed: set[str] = set()
    for path in snapshot.rglob("*"):
        relative = path.relative_to(snapshot).as_posix()
        if relative not in expected:
            raise ApexRuntimeError(f"runtime snapshot has an unreceipted path: {relative}")
        observed.add(relative)
        entry = expected[relative]
        metadata = path.lstat()
        kind = entry["type"]
        if kind == "directory":
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o555:
                raise ApexRuntimeError(f"runtime snapshot directory changed: {relative}")
        elif kind == "symlink":
            target = os.readlink(path) if stat.S_ISLNK(metadata.st_mode) else ""
            if (
                not stat.S_ISLNK(metadata.st_mode)
                or target != entry["target"]
                or PurePosixPath(target).is_absolute()
            ):
                raise ApexRuntimeError(f"runtime snapshot symlink changed: {relative}")
            try:
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise ApexRuntimeError(
                    f"runtime snapshot symlink is broken: {relative}"
                ) from error
            if not _path_below(resolved, snapshot):
                raise ApexRuntimeError(
                    f"runtime snapshot symlink escapes: {relative}"
                )
        else:
            data, opened = _read_regular(path)
            mode = (
                0o444
                if kind in {"manifest", "bootstrap"}
                else _snapshot_file_mode(entry)
            )
            digest = (
                _sha256_bytes(data)
                if kind == "manifest"
                else entry["sha256"]
            )
            if (
                stat.S_IMODE(opened.st_mode) != mode
                or (kind not in {"manifest", "bootstrap"} and len(data) != entry["size"])
                or (kind == "bootstrap" and len(data) != entry["size"])
                or (kind != "manifest" and _sha256_bytes(data) != digest)
            ):
                raise ApexRuntimeError(f"runtime snapshot file changed: {relative}")
    if observed != set(expected):
        missing = sorted(set(expected) - observed)[0]
        raise ApexRuntimeError(f"runtime snapshot path is missing: {missing}")
    return manifest


def runtime_image_inputs(
    root: str | Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Return canonical bytes/modes for a deterministic immutable image build."""

    snapshot = Path(root)
    expected_digest = manifest.get("sha256")
    verified = verify_runtime_snapshot(snapshot, expected_digest)
    if verified != manifest:
        raise ApexRuntimeError("runtime image input manifest differs")
    expected = _expected_snapshot_paths(manifest)
    entries: list[dict[str, Any]] = [
        {"path": ".", "type": "directory", "mode": 0o555}
    ]
    for relative, entry in sorted(expected.items()):
        path = snapshot / relative
        kind = entry["type"]
        if kind == "directory":
            value = {"path": relative, "type": "directory", "mode": 0o555}
        elif kind == "symlink":
            value = {
                "path": relative,
                "type": "symlink",
                "mode": 0o777,
                "target": entry["target"],
                "target_sha256": _sha256_bytes(os.fsencode(entry["target"])),
            }
        else:
            content, metadata = _read_regular(path)
            value = {
                "path": relative,
                "type": "file",
                "mode": stat.S_IMODE(metadata.st_mode),
                "size": len(content),
                "sha256": _sha256_bytes(content),
            }
        entries.append(value)
    material = {
        "schema": RUNTIME_IMAGE_INPUT_SCHEMA,
        "policy_id": RUNTIME_IMAGE_INPUT_POLICY_ID,
        "runtime_manifest_sha256": expected_digest,
        "entries": entries,
        "entries_sha256": _canonical_digest(entries),
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
    return {**material, "sha256": _canonical_digest(material)}


def _decode_mountinfo(value: str) -> str:
    replacements = {"\\040": " ", "\\011": "\t", "\\012": "\n", "\\134": "\\"}
    for escaped, decoded in replacements.items():
        value = value.replace(escaped, decoded)
    return value


def _observed_immutable_mount(snapshot: Path) -> dict[str, Any]:
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ApexRuntimeError("cannot inspect runtime mount identity") from error
    selected: dict[str, Any] | None = None
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator < 6 or len(fields) < separator + 4:
            continue
        mount_point = _decode_mountinfo(fields[4])
        if mount_point != str(snapshot):
            continue
        mount_options = sorted(set(fields[5].split(",")))
        super_options = sorted(set(fields[separator + 3].split(",")))
        selected = {
            "mount_id": int(fields[0]),
            "device": fields[2],
            "root": _decode_mountinfo(fields[3]),
            "mount_point": mount_point,
            "filesystem": fields[separator + 1],
            "mount_options": mount_options,
            "super_options": super_options,
            "read_only": "ro" in mount_options or "ro" in super_options,
        }
        break
    if selected is None:
        raise ApexRuntimeError("runtime snapshot is not an exact mount point")
    return selected


def validate_immutable_mount_receipt(
    snapshot: str | Path,
    manifest: dict[str, Any],
    receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    """Bind execution to a read-only SquashFS backed by a sealed memfd."""

    root = _canonical_directory(snapshot, label="immutable runtime mount")
    if not isinstance(receipt, dict):
        raise ApexRuntimeError("immutable runtime mount receipt is required")
    required = {
        "schema",
        "policy_id",
        "root",
        "runtime_manifest_sha256",
        "runtime_image_input_sha256",
        "image_sha256",
        "backing",
        "requested_mount_options",
        "runtime_service_evidence_sha256",
        "runtime_engine_evidence_sha256",
        "host_access_policy",
        "mount",
        "sha256",
    }
    if set(receipt) != required:
        raise ApexRuntimeError("immutable runtime mount receipt shape is invalid")
    material = dict(receipt)
    observed_digest = material.pop("sha256")
    backing = receipt.get("backing")
    mount = receipt.get("mount")
    host_policy = receipt.get("host_access_policy")
    image_inputs = runtime_image_inputs(root, manifest)
    if (
        receipt.get("schema") != RUNTIME_IMMUTABLE_MOUNT_SCHEMA
        or receipt.get("policy_id") != RUNTIME_IMMUTABLE_MOUNT_POLICY_ID
        or receipt.get("root") != str(root)
        or receipt.get("runtime_manifest_sha256") != manifest.get("sha256")
        or receipt.get("runtime_image_input_sha256") != image_inputs["sha256"]
        or receipt.get("requested_mount_options")
        != [
            "ro", "nodev", "nosuid", "default_permissions", "allow_other",
            "subtype=squashfuse",
        ]
        or not _SHA256.fullmatch(
            str(receipt.get("runtime_service_evidence_sha256") or "")
        )
        or not _SHA256.fullmatch(
            str(receipt.get("runtime_engine_evidence_sha256") or "")
        )
        or not _host_access_policy_valid(host_policy)
        or not isinstance(receipt.get("image_sha256"), str)
        or not _SHA256.fullmatch(receipt["image_sha256"])
        or not isinstance(observed_digest, str)
        or observed_digest != _canonical_digest(material)
        or backing
        != {"kind": "sealed_memfd", "seals": list(_REQUIRED_MEMFD_SEALS)}
        or not isinstance(mount, dict)
    ):
        raise ApexRuntimeError("immutable runtime mount receipt is invalid")
    observed_mount = _observed_immutable_mount(root)
    if (
        mount != observed_mount
        or mount.get("filesystem")
        not in {"squashfs", "fuse.squashfuse", "fuse.squashfuse_ll"}
        or mount.get("read_only") is not True
        or "allow_other" not in mount.get("super_options", [])
        or f"user_id={host_policy['mount_owner']['uid']}"
        not in mount.get("super_options", [])
        or f"group_id={host_policy['mount_owner']['gid']}"
        not in mount.get("super_options", [])
        or type(mount.get("mount_id")) is not int
        or mount["mount_id"] <= 0
    ):
        raise ApexRuntimeError("runtime mount is not immutable SquashFS")
    return receipt


def create_immutable_mount_receipt(
    snapshot: str | Path,
    manifest: dict[str, Any],
    image_sha256: str,
    runtime_service_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Attest the Apex SquashFS mount in the caller's current namespace."""

    root = _canonical_directory(snapshot, label="immutable runtime mount")
    verify_runtime_snapshot(root, manifest.get("sha256"))
    if not _SHA256.fullmatch(image_sha256):
        raise ApexRuntimeError("immutable runtime image digest is invalid")
    image_inputs = runtime_image_inputs(root, manifest)
    engine_evidence = runtime_service_evidence["engine_evidence"]
    host_policy = runtime_service_evidence["mount_receipt"]["host_access_policy"]
    material = {
        "schema": RUNTIME_IMMUTABLE_MOUNT_SCHEMA,
        "policy_id": RUNTIME_IMMUTABLE_MOUNT_POLICY_ID,
        "root": str(root),
        "runtime_manifest_sha256": manifest["sha256"],
        "runtime_image_input_sha256": image_inputs["sha256"],
        "image_sha256": image_sha256,
        "backing": {
            "kind": "sealed_memfd",
            "seals": list(_REQUIRED_MEMFD_SEALS),
        },
        "requested_mount_options": [
            "ro", "nodev", "nosuid", "default_permissions", "allow_other",
            "subtype=squashfuse",
        ],
        "runtime_service_evidence_sha256": runtime_service_evidence["sha256"],
        "runtime_engine_evidence_sha256": engine_evidence["sha256"],
        "host_access_policy": host_policy,
        "mount": _observed_immutable_mount(root),
    }
    receipt = {**material, "sha256": _canonical_digest(material)}
    return validate_immutable_mount_receipt(root, manifest, receipt)


def runtime_command(
    snapshot: str | Path,
    manifest: dict[str, Any],
    arguments: Sequence[str],
    *,
    immutable_mount_receipt: dict[str, Any] | None = None,
) -> list[str]:
    """Build the formal argv from a verified snapshot manifest."""

    root = Path(snapshot)
    expected = manifest.get("sha256")
    verify_runtime_snapshot(root, expected)
    validate_immutable_mount_receipt(root, manifest, immutable_mount_receipt)
    execution = manifest.get("execution")
    if not isinstance(execution, dict):
        raise ApexRuntimeError("runtime execution contract is invalid")
    interpreter = root / _safe_relative(execution.get("interpreter", ""))
    underlying = root / _safe_relative(
        execution.get("underlying_interpreter", "")
    )
    entrypoint = root / _safe_relative(execution.get("entrypoint", ""))
    bootstrap = root / _safe_relative(execution.get("bootstrap", ""))
    flags = execution.get("flags")
    pythonpath = execution.get("pythonpath")
    if (
        flags != ["-I", "-S", "-u"]
        or execution.get("bootstrap_policy_id") != RUNTIME_BOOTSTRAP_POLICY_ID
        or execution.get("bootstrap_sha256") != RUNTIME_BOOTSTRAP_SHA256
        or execution.get("wrapper_policy_id") != RUNTIME_WRAPPER_POLICY_ID
        or execution.get("wrapper_sha256") != RUNTIME_WRAPPER_SHA256
        or execution.get("wrapper_aliases")
        != [f"sealed-bin/{alias}" for alias in RUNTIME_WRAPPER_ALIASES]
        or execution.get("no_live_interpreter_fallback") is not True
        or not isinstance(pythonpath, list)
        or not pythonpath
        or not os.access(interpreter, os.X_OK)
        or not os.access(underlying, os.X_OK)
        or _sha256_bytes(_read_regular(interpreter)[0]) != RUNTIME_WRAPPER_SHA256
        or _sha256_bytes(_read_regular(bootstrap)[0]) != RUNTIME_BOOTSTRAP_SHA256
    ):
        raise ApexRuntimeError("runtime interpreter contract is invalid")
    paths = [str(root / _safe_relative(value)) for value in pythonpath]
    return [
        str(interpreter),
        "--apex-entrypoint",
        str(entrypoint),
        *arguments,
    ]


def runtime_environment(snapshot: str | Path, manifest: dict[str, Any]) -> dict[str, str]:
    """Return the exact Python isolation environment for a snapshot."""

    root = Path(snapshot)
    execution = manifest.get("execution")
    pythonpath = execution.get("pythonpath") if isinstance(execution, dict) else None
    if not isinstance(pythonpath, list) or not pythonpath:
        raise ApexRuntimeError("runtime PYTHONPATH contract is invalid")
    paths = [str(root / _safe_relative(value)) for value in pythonpath]
    return {
        "PATH": os.pathsep.join(
            (
                str(root / "sealed-bin"),
                "/opt/rocm/bin",
                "/usr/local/bin",
                "/usr/bin",
                "/bin",
            )
        ),
        "APEX_RUNTIME_PYTHON": str(root / RUNTIME_WRAPPER_NAME),
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(paths),
    }


def _write_output(path: str | None, value: Any) -> None:
    content = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    if path is None or path == "-":
        print(content, end="")
        return
    destination = Path(path)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(content)


def _main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "discover",
        "digest",
        "manifest",
        "materialize",
        "verify",
        "image-input",
        "mount-receipt",
    ):
        command = subparsers.add_parser(name)
        if name not in {"verify", "image-input", "mount-receipt"}:
            command.add_argument("--root", required=True)
            command.add_argument("--python", required=True)
            command.add_argument("--declared-root", action="append", default=[])
        if name == "materialize":
            command.add_argument("--snapshot-parent", required=True)
        if name in {"verify", "image-input", "mount-receipt"}:
            command.add_argument("--snapshot", required=True)
            command.add_argument("--sha256", required=True)
        if name == "mount-receipt":
            command.add_argument("--image-sha256", required=True)
            command.add_argument("--service-evidence", required=True)
            command.add_argument("--service-file-sha256", required=True)
            command.add_argument("--service-content-sha256", required=True)
        command.add_argument("--output")
    options = parser.parse_args(arguments)
    if options.command in {"verify", "image-input", "mount-receipt"}:
        manifest = verify_runtime_snapshot(options.snapshot, options.sha256)
        if options.command == "verify":
            value = manifest
        elif options.command == "image-input":
            value = runtime_image_inputs(options.snapshot, manifest)
        else:
            service = load_runtime_service_evidence(
                options.service_evidence,
                file_sha256=options.service_file_sha256,
                content_sha256=options.service_content_sha256,
                manifest_sha256=options.sha256,
                image_sha256=options.image_sha256,
            )
            value = create_immutable_mount_receipt(
                options.snapshot, manifest, options.image_sha256, service
            )
    else:
        if options.command == "discover":
            root = _canonical_directory(options.root, label="APEX_ROOT")
            launcher = _launcher_identity(root, options.python)
            venv_root = Path(launcher["venv_root"])
            external = discover_external_roots(options.root, options.python)
            value = {
                "schema": RUNTIME_SNAPSHOT_SCHEMA,
                "venv_root": str(venv_root),
                "external_roots": [str(path) for path in external],
            }
        else:
            plan = plan_runtime(
                options.root,
                options.python,
                declared_roots=options.declared_root,
            )
            if options.command == "manifest":
                value = plan.manifest
            elif options.command == "digest":
                value = {
                    "schema": RUNTIME_SNAPSHOT_SCHEMA,
                    "runtime_manifest_sha256": plan.sha256,
                }
            else:
                snapshot = materialize_runtime(plan, options.snapshot_parent)
                value = {
                    "schema": RUNTIME_SNAPSHOT_SCHEMA,
                    "root": str(snapshot),
                    "runtime_manifest_sha256": plan.sha256,
                    "repository": {
                        key: plan.manifest["git"][key]
                        for key in ("commit", "dirty", "status_sha256")
                    },
                }
    _write_output(options.output, value)
    if (
        options.command == "mount-receipt"
        and options.output is not None
        and options.output != "-"
    ):
        # ``docker_benchmark.sh`` captures this content digest while the full
        # receipt is written to its namespace-local evidence file.  Keep the
        # stdout contract aligned with ``aka_runtime.py mount-receipt``.
        print(value["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
