# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Host-owned immutable SquashFS mounts for preverified runtime snapshots."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


class ImmutableRuntimeMountError(RuntimeError):
    """Raised when an immutable runtime cannot be built, proven, or removed."""


IMAGE_INPUT_SCHEMA = "aka.apex-runtime-image-input/v1"
IMAGE_INPUT_POLICY_ID = "deterministic_squashfs_inputs_v1"
MOUNT_RECEIPT_SCHEMA = "aka.apex-runtime-immutable-mount/v1"
MOUNT_POLICY_ID = "sealed_memfd_squashfs_read_only_v1"
ENGINE_EVIDENCE_SCHEMA = "aka.immutable-runtime-mount-engine/v1"
_MKSQUASHFS = Path("/usr/bin/mksquashfs")
_MKSQUASHFS_SHA256 = "403080bcd98ea7be2cbb261a10e99a89571e3a3beed6ab6cc3b88e01a0b51053"
_MKSQUASHFS_SIZE = 260_792
_MKSQUASHFS_VERSION = "mksquashfs version 4.5 (2021/07/22)"
_SQUASHFUSE = Path("/usr/bin/squashfuse")
_SQUASHFUSE_SHA256 = "6b2efeca3df43609c93859daebd6426c9e53f17613e76a034bde63b18dd01fd0"
_SQUASHFUSE_SIZE = 14_488
_SQUASHFUSE_VERSION = "squashfuse 0.1.103 (c) 2012 Dave Vasilevsky"
_FUSERMOUNT = Path("/usr/bin/fusermount3")
_FUSERMOUNT_SHA256 = "fa2dc1bb00be297004cfa4fc82dab3a6d568042736f7eb5b6fd8de49804db2d1"
_FUSERMOUNT_SIZE = 35_200
_SEAL_NAMES = (
    "F_SEAL_GROW",
    "F_SEAL_SEAL",
    "F_SEAL_SHRINK",
    "F_SEAL_WRITE",
)
_SEAL_MASK = (
    fcntl.F_SEAL_GROW
    | fcntl.F_SEAL_SEAL
    | fcntl.F_SEAL_SHRINK
    | fcntl.F_SEAL_WRITE
)
_SHA256 = frozenset("0123456789abcdef")


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - _SHA256)
    )


def _descriptor_bytes(descriptor: int, size: int) -> bytes:
    result = bytearray()
    offset = 0
    while len(result) <= size:
        chunk = os.pread(descriptor, size + 1 - len(result), offset)
        if not chunk:
            break
        result.extend(chunk)
        offset += len(chunk)
    return bytes(result)


def _open_pinned_tool(
    path: Path, *, expected_sha256: str, expected_size: int, expected_mode: int
) -> tuple[int, dict[str, Any]]:
    descriptor = -1
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        )
        opened = os.fstat(descriptor)
        lexical = path.lstat()
        payload = _descriptor_bytes(descriptor, expected_size)
        digest = hashlib.sha256(payload).hexdigest()
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(lexical.st_mode)
            or (opened.st_dev, opened.st_ino) != (lexical.st_dev, lexical.st_ino)
            or opened.st_uid != 0
            or opened.st_gid != 0
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != expected_mode
            or opened.st_size != expected_size
            or len(payload) != expected_size
            or digest != expected_sha256
        ):
            raise ImmutableRuntimeMountError(f"pinned tool identity changed: {path}")
        identity = {
            "canonical_path": str(path),
            "device": opened.st_dev,
            "inode": opened.st_ino,
            "mode": stat.S_IMODE(opened.st_mode),
            "uid": opened.st_uid,
            "gid": opened.st_gid,
            "size_bytes": opened.st_size,
            "sha256": digest,
        }
        result, descriptor = descriptor, -1
        return result, identity
    except OSError as error:
        raise ImmutableRuntimeMountError(f"cannot open pinned tool: {path}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _sealed_memfd(payload: bytes, *, name: str, mode: int) -> int:
    descriptor = -1
    try:
        descriptor = os.memfd_create(
            name, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
        )
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise ImmutableRuntimeMountError(f"cannot write sealed {name}")
            remaining = remaining[written:]
        os.fchmod(descriptor, mode)
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, _SEAL_MASK)
        if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != _SEAL_MASK:
            raise ImmutableRuntimeMountError(f"sealed {name} lacks required seals")
        if _descriptor_bytes(descriptor, len(payload)) != payload:
            raise ImmutableRuntimeMountError(f"sealed {name} bytes changed")
        result, descriptor = descriptor, -1
        return result
    except (AttributeError, OSError) as error:
        raise ImmutableRuntimeMountError(f"cannot create sealed {name}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _sealed_tool(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> tuple[int, dict[str, Any]]:
    source_fd = sealed_fd = -1
    try:
        source_fd, source = _open_pinned_tool(
            path,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            expected_mode=0o755,
        )
        payload = _descriptor_bytes(source_fd, expected_size)
        sealed_fd = _sealed_memfd(payload, name=f"aka-{path.name}", mode=0o555)
        sealed = {
            "transport": "sealed_memfd_proc_self_fd",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "seals": list(_SEAL_NAMES),
        }
        result, sealed_fd = sealed_fd, -1
        return result, {"source": source, "sealed_exec": sealed}
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        if sealed_fd >= 0:
            os.close(sealed_fd)


def _tool_version(
    descriptor: int, arguments: list[str], expected_line: str
) -> str:
    try:
        completed = subprocess.run(
            [f"/proc/self/fd/{descriptor}", *arguments],
            pass_fds=(descriptor,),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ImmutableRuntimeMountError("cannot execute pinned tool") from error
    lines = (completed.stdout + completed.stderr).splitlines()
    first = lines[0].strip() if lines else ""
    if first != expected_line:
        raise ImmutableRuntimeMountError("pinned tool version changed")
    return first


def _safe_inventory_path(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(ord(character) < 32 for character in value)
    ):
        raise ImmutableRuntimeMountError("runtime inventory path is unsafe")
    path = PurePosixPath(value)
    if value == ".":
        return value
    if path.is_absolute() or value != path.as_posix() or ".." in path.parts:
        raise ImmutableRuntimeMountError("runtime inventory path is unsafe")
    return value


def _validated_entries(inventory: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entries = inventory.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ImmutableRuntimeMountError("runtime image inventory is empty")
    result: dict[str, dict[str, Any]] = {}
    for raw in entries:
        if not isinstance(raw, dict):
            raise ImmutableRuntimeMountError("runtime image entry is malformed")
        path = _safe_inventory_path(raw.get("path"))
        kind = raw.get("type")
        required = {"path", "type", "mode"}
        if kind == "file":
            required |= {"size", "sha256"}
        elif kind == "symlink":
            required |= {"target", "target_sha256"}
        elif kind != "directory":
            raise ImmutableRuntimeMountError("runtime image entry type is invalid")
        if set(raw) != required or path in result:
            raise ImmutableRuntimeMountError("runtime image entry shape is invalid")
        mode = raw.get("mode")
        if type(mode) is not int or mode < 0 or mode > 0o777:
            raise ImmutableRuntimeMountError("runtime image entry mode is invalid")
        if kind == "file" and (
            type(raw.get("size")) is not int
            or raw["size"] < 0
            or not _valid_sha256(raw.get("sha256"))
        ):
            raise ImmutableRuntimeMountError("runtime image file identity is invalid")
        if kind == "symlink":
            target = raw.get("target")
            if (
                not isinstance(target, str)
                or not target
                or PurePosixPath(target).is_absolute()
                or not _valid_sha256(raw.get("target_sha256"))
                or hashlib.sha256(os.fsencode(target)).hexdigest()
                != raw["target_sha256"]
            ):
                raise ImmutableRuntimeMountError("runtime image symlink is invalid")
        result[path] = dict(raw)
    ordered = [".", *sorted((path for path in result if path != "."), key=str.encode)]
    if list(result) != ordered or result.get(".") != {
        "path": ".",
        "type": "directory",
        "mode": 0o555,
    }:
        raise ImmutableRuntimeMountError("runtime image entries are not canonical")
    return result


def validate_image_inventory(inventory: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate the canonical image-input receipt produced by runtime planning."""

    required = {
        "schema",
        "policy_id",
        "runtime_manifest_sha256",
        "entries",
        "entries_sha256",
        "normalization",
        "sha256",
    }
    material = dict(inventory)
    digest = material.pop("sha256", None)
    entries = _validated_entries(inventory)
    if (
        set(inventory) != required
        or inventory.get("schema") != IMAGE_INPUT_SCHEMA
        or inventory.get("policy_id") != IMAGE_INPUT_POLICY_ID
        or not _valid_sha256(inventory.get("runtime_manifest_sha256"))
        or inventory.get("entries_sha256") != _canonical_digest(inventory["entries"])
        or not _valid_sha256(digest)
        or digest != _canonical_digest(material)
        or inventory.get("normalization")
        != {
            "uid": 0,
            "gid": 0,
            "mtime_epoch": 0,
            "xattrs": "none",
            "ordering": "utf8_posix_path_ascending",
            "format": "squashfs",
            "compression": "caller_pinned_and_receipted",
        }
    ):
        raise ImmutableRuntimeMountError("runtime image inventory digest is invalid")
    return entries


def _walk_tree(root: Path) -> dict[str, os.stat_result]:
    observed: dict[str, os.stat_result] = {".": root.lstat()}
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            children = sorted(os.scandir(directory), key=lambda entry: os.fsencode(entry.name))
        except OSError as error:
            raise ImmutableRuntimeMountError("cannot traverse runtime tree") from error
        for child in children:
            path = Path(child.path)
            relative = path.relative_to(root).as_posix()
            metadata = path.lstat()
            observed[relative] = metadata
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(path)
            elif not (stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)):
                raise ImmutableRuntimeMountError(
                    f"runtime tree contains unsupported entry: {relative}"
                )
    return observed


def _verify_tree(
    root: Path,
    entries: Mapping[str, Mapping[str, Any]],
    *,
    mounted: bool,
) -> dict[str, Any]:
    observed = _walk_tree(root)
    if set(observed) != set(entries):
        raise ImmutableRuntimeMountError("runtime tree differs from canonical inventory")
    for relative, entry in entries.items():
        path = root if relative == "." else root / relative
        metadata = observed[relative]
        kind = entry["type"]
        expected_type = {
            "directory": stat.S_ISDIR,
            "file": stat.S_ISREG,
            "symlink": stat.S_ISLNK,
        }[kind]
        if (
            not expected_type(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != entry["mode"]
        ):
            raise ImmutableRuntimeMountError(f"runtime entry mode changed: {relative}")
        if mounted and (metadata.st_uid != 0 or metadata.st_gid != 0 or int(metadata.st_mtime) != 0):
            raise ImmutableRuntimeMountError(f"runtime entry normalization changed: {relative}")
        if kind == "file":
            try:
                descriptor = os.open(
                    path,
                    os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    opened = os.fstat(descriptor)
                    payload = _descriptor_bytes(descriptor, entry["size"])
                finally:
                    os.close(descriptor)
            except OSError as error:
                raise ImmutableRuntimeMountError(
                    f"cannot verify runtime file: {relative}"
                ) from error
            if (
                (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
                or len(payload) != entry["size"]
                or hashlib.sha256(payload).hexdigest() != entry["sha256"]
            ):
                raise ImmutableRuntimeMountError(f"runtime file changed: {relative}")
        elif kind == "symlink":
            target = os.readlink(path)
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError) as error:
                raise ImmutableRuntimeMountError(
                    f"runtime symlink escapes: {relative}"
                ) from error
            if target != entry["target"]:
                raise ImmutableRuntimeMountError(f"runtime symlink changed: {relative}")
        try:
            if os.listxattr(path, follow_symlinks=False):
                raise ImmutableRuntimeMountError(f"runtime xattrs are present: {relative}")
        except OSError as error:
            raise ImmutableRuntimeMountError(
                f"cannot verify runtime xattrs: {relative}"
            ) from error
    return {"entries_verified": len(entries), "entries_sha256": _canonical_digest(list(entries.values()))}


def _canonical_empty_mountpoint(path: str | Path, staging: Path) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise ImmutableRuntimeMountError("runtime mountpoint must be absolute")
    lexical = Path(os.path.abspath(raw))
    try:
        metadata = lexical.lstat()
        root = lexical.resolve(strict=True)
    except OSError as error:
        raise ImmutableRuntimeMountError("runtime mountpoint is unavailable") from error
    if (
        root != lexical
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or any(root.iterdir())
        or root == staging
        or root.is_relative_to(staging)
        or staging.is_relative_to(root)
    ):
        raise ImmutableRuntimeMountError("runtime mountpoint is unsafe or nonempty")
    return root


def _build_image(
    staging: Path, entries: Mapping[str, Mapping[str, Any]]
) -> tuple[bytes, dict[str, Any]]:
    executable_fd = -1
    try:
        executable_fd, tool = _sealed_tool(
            _MKSQUASHFS,
            expected_sha256=_MKSQUASHFS_SHA256,
            expected_size=_MKSQUASHFS_SIZE,
        )
        tool["version"] = _tool_version(
            executable_fd, ["-version"], _MKSQUASHFS_VERSION
        )
        with tempfile.TemporaryDirectory(prefix="aka-runtime-image-") as temporary:
            work = Path(temporary)
            output = work / "runtime.squashfs"
            sort_file = work / "sort.txt"
            sortable = [path for path in entries if path != "."]
            sort_file.write_text(
                "".join(
                    f"{path} {len(sortable) - index}\n"
                    for index, path in enumerate(sortable)
                ),
                encoding="utf-8",
            )
            command = [
                f"/proc/self/fd/{executable_fd}",
                str(staging),
                str(output),
                "-noappend",
                "-all-root",
                "-no-xattrs",
                "-no-exports",
                "-no-progress",
                "-quiet",
                "-comp",
                "gzip",
                "-b",
                "131072",
                "-no-fragments",
                "-no-duplicates",
                "-processors",
                "1",
                "-mkfs-time",
                "0",
                "-all-time",
                "0",
                "-sort",
                str(sort_file),
            ]
            completed = subprocess.run(
                command,
                pass_fds=(executable_fd,),
                capture_output=True,
                check=False,
                timeout=120,
                env={
                    "LC_ALL": "C",
                    "PATH": "/usr/bin:/bin",
                    "TZ": "UTC",
                },
            )
            if completed.returncode != 0 or not output.is_file():
                detail = completed.stderr.decode("utf-8", "replace")[-2000:]
                raise ImmutableRuntimeMountError(
                    f"deterministic mksquashfs failed: {detail}"
                )
            image = output.read_bytes()
        if not image:
            raise ImmutableRuntimeMountError("mksquashfs emitted an empty image")
        return image, tool
    finally:
        if executable_fd >= 0:
            os.close(executable_fd)


def _decode_mount_path(value: str) -> str:
    for escaped, plain in ((r"\040", " "), (r"\011", "\t"), (r"\012", "\n"), (r"\134", "\\")):
        value = value.replace(escaped, plain)
    return value


def _mountinfo(root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ImmutableRuntimeMountError("cannot inspect runtime mount") from error
    exact: list[dict[str, Any]] = []
    nested: list[str] = []
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
            mount_point = Path(_decode_mount_path(fields[4]))
        except (ValueError, IndexError):
            continue
        if mount_point != root and not mount_point.is_relative_to(root):
            continue
        if mount_point != root:
            nested.append(str(mount_point))
            continue
        options = sorted(set(fields[5].split(",")))
        super_options = sorted(set(fields[separator + 3].split(",")))
        exact.append(
            {
                "mount_id": int(fields[0]),
                "device": fields[2],
                "root": _decode_mount_path(fields[3]),
                "mount_point": str(mount_point),
                "filesystem": fields[separator + 1],
                "mount_options": options,
                "super_options": super_options,
                "read_only": "ro" in options or "ro" in super_options,
            }
        )
    if len(exact) > 1:
        raise ImmutableRuntimeMountError("runtime mountpoint has stacked mounts")
    return (exact[0] if exact else None), sorted(nested)


def _wait_for_mount(
    root: Path, process: subprocess.Popen[bytes], timeout: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        mount, nested = _mountinfo(root)
        if mount is not None:
            if nested:
                raise ImmutableRuntimeMountError("runtime mount contains nested mounts")
            if (
                mount["filesystem"] not in {"squashfs", "fuse.squashfuse", "fuse.squashfuse_ll"}
                or mount["read_only"] is not True
                or "ro" not in mount["mount_options"]
                or not {"nodev", "nosuid"}.issubset(mount["mount_options"])
            ):
                raise ImmutableRuntimeMountError("runtime SquashFS mount is not read-only")
            return mount
        if process.poll() is not None:
            stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
            raise ImmutableRuntimeMountError(
                f"squashfuse exited before mounting: {stderr[-2000:]}"
            )
        time.sleep(0.05)
    raise ImmutableRuntimeMountError("timed out waiting for SquashFS mount")


def _read_only_probe(root: Path) -> int:
    probe = root / ".aka-write-probe"
    try:
        probe.write_bytes(b"forbidden")
    except OSError as error:
        return error.errno or 0
    probe.unlink(missing_ok=True)
    raise ImmutableRuntimeMountError("runtime mount accepted a write")


def _process_starttime(pid: int) -> int:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return int(fields[fields.rfind(")") + 2 :].split()[19])
    except (OSError, ValueError, IndexError) as error:
        raise ImmutableRuntimeMountError("cannot bind squashfuse process identity") from error


def _receipt(
    *, root: Path, inventory: Mapping[str, Any], image: bytes, mount: dict[str, Any]
) -> dict[str, Any]:
    material = {
        "schema": MOUNT_RECEIPT_SCHEMA,
        "policy_id": MOUNT_POLICY_ID,
        "root": str(root),
        "runtime_manifest_sha256": inventory["runtime_manifest_sha256"],
        "runtime_image_input_sha256": inventory["sha256"],
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "backing": {"kind": "sealed_memfd", "seals": list(_SEAL_NAMES)},
        "mount": mount,
    }
    return {**material, "sha256": _canonical_digest(material)}


@dataclass
class ImmutableRuntimeMount:
    """Live mount ownership; callers must close it after the attempt."""

    root: Path
    process: subprocess.Popen[bytes]
    image_fd: int
    receipt: dict[str, Any]
    evidence: dict[str, Any]
    mountpoint_identity: dict[str, int]
    cleanup_evidence: dict[str, Any] | None = None

    def __enter__(self) -> "ImmutableRuntimeMount":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self.image_fd < 0

    def close(self, *, timeout_seconds: float = 10.0) -> dict[str, Any]:
        if self.cleanup_evidence is not None:
            return self.cleanup_evidence
        image_fd, self.image_fd = self.image_fd, -1
        unmount: subprocess.CompletedProcess[bytes] | None = None
        forced = False
        try:
            mount, _nested = _mountinfo(self.root)
            if mount is not None:
                _fd, tool = _open_pinned_tool(
                    _FUSERMOUNT,
                    expected_sha256=_FUSERMOUNT_SHA256,
                    expected_size=_FUSERMOUNT_SIZE,
                    expected_mode=0o4755,
                )
                os.close(_fd)
                unmount = subprocess.run(
                    [str(_FUSERMOUNT), "-u", str(self.root)],
                    capture_output=True,
                    check=False,
                    timeout=timeout_seconds,
                    env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
                )
                if unmount.returncode != 0:
                    forced = True
                    if self.process.poll() is None:
                        os.killpg(self.process.pid, signal.SIGTERM)
                        try:
                            self.process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            os.killpg(self.process.pid, signal.SIGKILL)
                            self.process.wait(timeout=2)
                    if _mountinfo(self.root)[0] is not None:
                        unmount = subprocess.run(
                            [str(_FUSERMOUNT), "-uz", str(self.root)],
                            capture_output=True,
                            check=False,
                            timeout=timeout_seconds,
                            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
                        )
                    if (
                        _mountinfo(self.root)[0] is not None
                        and unmount.returncode != 0
                    ):
                        raise ImmutableRuntimeMountError(
                            "fusermount could not unmount runtime"
                        )
            try:
                self.process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                forced = True
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=2)
            remaining, nested = _mountinfo(self.root)
            if remaining is not None or nested:
                raise ImmutableRuntimeMountError("runtime mount survived exact cleanup")
            if any(self.root.iterdir()):
                raise ImmutableRuntimeMountError("runtime mountpoint is not empty after cleanup")
            metadata = self.root.lstat()
            if (metadata.st_dev, metadata.st_ino) != (
                self.mountpoint_identity["device"],
                self.mountpoint_identity["inode"],
            ):
                raise ImmutableRuntimeMountError(
                    "runtime mountpoint identity changed during cleanup"
                )
            material = {
                "schema": "aka.immutable-runtime-mount-cleanup/v1",
                "root": str(self.root),
                "unmount_returncode": unmount.returncode if unmount else None,
                "squashfuse_exit_code": self.process.returncode,
                "forced_process_stop": forced,
                "mount_absent": True,
                "mountpoint_empty": True,
                "image_memfd_closed": True,
            }
            self.cleanup_evidence = {
                **material,
                "sha256": _canonical_digest(material),
            }
            return self.cleanup_evidence
        finally:
            if image_fd >= 0:
                os.close(image_fd)


def mount_immutable_runtime(
    staging_root: str | Path,
    mountpoint: str | Path,
    inventory: Mapping[str, Any],
    *,
    timeout_seconds: float = 10.0,
) -> ImmutableRuntimeMount:
    """Build, seal, mount, and fully verify one immutable runtime tree."""

    staging_raw = Path(staging_root).expanduser()
    if not staging_raw.is_absolute():
        raise ImmutableRuntimeMountError("runtime staging tree must be absolute")
    staging_lexical = Path(os.path.abspath(staging_raw))
    try:
        staging_metadata = staging_lexical.lstat()
        staging = staging_lexical.resolve(strict=True)
    except OSError as error:
        raise ImmutableRuntimeMountError("runtime staging tree is unavailable") from error
    if (
        staging != staging_lexical
        or stat.S_ISLNK(staging_metadata.st_mode)
        or not stat.S_ISDIR(staging_metadata.st_mode)
    ):
        raise ImmutableRuntimeMountError("runtime staging tree is unsafe")
    entries = validate_image_inventory(inventory)
    _verify_tree(staging, entries, mounted=False)
    root = _canonical_empty_mountpoint(mountpoint, staging)
    image, builder = _build_image(staging, entries)
    _verify_tree(staging, entries, mounted=False)
    image_fd = squashfuse_fd = mountpoint_fd = -1
    process: subprocess.Popen[bytes] | None = None
    try:
        image_fd = _sealed_memfd(image, name="aka-runtime-squashfs", mode=0o400)
        squashfuse_fd, squashfuse = _sealed_tool(
            _SQUASHFUSE,
            expected_sha256=_SQUASHFUSE_SHA256,
            expected_size=_SQUASHFUSE_SIZE,
        )
        squashfuse["version"] = _tool_version(
            squashfuse_fd, ["--help"], _SQUASHFUSE_VERSION
        )
        mountpoint_fd = os.open(
            root,
            os.O_DIRECTORY
            | os.O_CLOEXEC
            | getattr(os, "O_PATH", os.O_RDONLY)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        mountpoint_stat = os.fstat(mountpoint_fd)
        mountpoint_identity = {
            "device": mountpoint_stat.st_dev,
            "inode": mountpoint_stat.st_ino,
            "mode": stat.S_IMODE(mountpoint_stat.st_mode),
        }
        process = subprocess.Popen(
            [
                f"/proc/self/fd/{squashfuse_fd}",
                "-f",
                "-s",
                "-o",
                "ro,nodev,nosuid,subtype=squashfuse",
                f"/proc/self/fd/{image_fd}",
                f"/proc/self/fd/{mountpoint_fd}",
            ],
            pass_fds=(squashfuse_fd, image_fd, mountpoint_fd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
        os.close(squashfuse_fd)
        squashfuse_fd = -1
        os.close(mountpoint_fd)
        mountpoint_fd = -1
        mount = _wait_for_mount(root, process, timeout_seconds)
        verification = _verify_tree(root, entries, mounted=True)
        write_errno = _read_only_probe(root)
        receipt = _receipt(root=root, inventory=inventory, image=image, mount=mount)
        evidence_material = {
            "schema": ENGINE_EVIDENCE_SCHEMA,
            "policy_id": MOUNT_POLICY_ID,
            "receipt_sha256": receipt["sha256"],
            "runtime_image_input_sha256": inventory["sha256"],
            "image": {
                "size_bytes": len(image),
                "sha256": hashlib.sha256(image).hexdigest(),
                "memfd_seals": list(_SEAL_NAMES),
            },
            "tools": {"mksquashfs": builder, "squashfuse": squashfuse},
            "process": {
                "pid": process.pid,
                "starttime": _process_starttime(process.pid),
                "foreground": True,
            },
            "mountpoint_source": mountpoint_identity,
            "mount": mount,
            "inventory_verification": verification,
            "write_probe_errno": write_errno,
        }
        evidence = {
            **evidence_material,
            "sha256": _canonical_digest(evidence_material),
        }
        result = ImmutableRuntimeMount(
            root,
            process,
            image_fd,
            receipt,
            evidence,
            mountpoint_identity,
        )
        image_fd = -1
        process = None
        return result
    except Exception as error:
        cleanup_error: ImmutableRuntimeMountError | None = None
        if process is not None:
            try:
                fusermount_fd, _fusermount = _open_pinned_tool(
                    _FUSERMOUNT,
                    expected_sha256=_FUSERMOUNT_SHA256,
                    expected_size=_FUSERMOUNT_SIZE,
                    expected_mode=0o4755,
                )
                os.close(fusermount_fd)
                unmounted = subprocess.run(
                    [str(_FUSERMOUNT), "-u", str(root)],
                    capture_output=True,
                    timeout=3,
                    check=False,
                    env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
                )
                del unmounted
            except (OSError, subprocess.TimeoutExpired, ImmutableRuntimeMountError):
                pass
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    cleanup_error = ImmutableRuntimeMountError(
                        "failed squashfuse process survived cleanup"
                    )
            remaining, nested = _mountinfo(root)
            if remaining is not None or nested:
                cleanup_error = ImmutableRuntimeMountError(
                    "failed runtime mount survived cleanup"
                )
        if cleanup_error is not None:
            raise cleanup_error from error
        raise
    finally:
        if squashfuse_fd >= 0:
            os.close(squashfuse_fd)
        if mountpoint_fd >= 0:
            os.close(mountpoint_fd)
        if image_fd >= 0:
            os.close(image_fd)


__all__ = [
    "ENGINE_EVIDENCE_SCHEMA",
    "IMAGE_INPUT_POLICY_ID",
    "IMAGE_INPUT_SCHEMA",
    "ImmutableRuntimeMount",
    "ImmutableRuntimeMountError",
    "MOUNT_POLICY_ID",
    "MOUNT_RECEIPT_SCHEMA",
    "mount_immutable_runtime",
    "validate_image_inventory",
]
