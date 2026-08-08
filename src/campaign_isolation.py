# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Per-attempt mount and home isolation for formal matched campaigns."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.gpu_device_boundary import (
    GpuBoundaryError,
    RECEIPT_SCHEMA as GPU_BOUNDARY_RECEIPT_SCHEMA,
    canonical_digest,
    load_plan,
    selected_device,
)
from src.gpu_exclusivity import GpuExclusivityError, load_receipt


class CampaignIsolationError(RuntimeError):
    """Raised when a formal attempt cannot be given a private filesystem view."""


ATTEMPT_CONTAINMENT_POLICY = "private_pid_namespace_init_pidfd_v1"
ATTEMPT_MOUNT_RECEIPT_SCHEMA = "aka.attempt-mounts/v1"
APEX_RUNTIME_MOUNT_POLICY = "validated_exact_apex_root_read_only_v1"
APEX_RUNTIME_MOUNT_SCHEMA = "aka.apex-runtime-mount/v1"
_BROAD_EXTERNAL_ROOTS = frozenset(
    {Path("/tmp"), Path("/var/tmp"), Path("/dev/shm")}
)


class WrappedAttemptCommand(list[str]):
    """Command argv plus narrowly scoped descriptors required by its mounts."""

    def __init__(
        self,
        argv: Iterable[str],
        *,
        pass_fds: Iterable[int] = (),
        boundary_status_fd: int | None = None,
        boundary_gate_fd: int | None = None,
        boundary_procfs: str = "private_attempt_procfs",
        mount_receipt: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(argv)
        self.pass_fds = tuple(pass_fds)
        self.boundary_status_fd = boundary_status_fd
        self.boundary_gate_fd = boundary_gate_fd
        self.boundary_procfs = boundary_procfs
        self.mount_receipt = mount_receipt

    def release_pass_fds(self) -> None:
        descriptors, self.pass_fds = self.pass_fds, ()
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def close_boundary_status_fd(self) -> None:
        descriptor, self.boundary_status_fd = self.boundary_status_fd, None
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def close_boundary_gate_fd(self) -> None:
        descriptor, self.boundary_gate_fd = self.boundary_gate_fd, None
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


@dataclass
class AttemptBoundary:
    """Exact pidfd identity for a formal attempt's private PID namespace init."""

    init_pidfd: int
    status_fd: int
    status: dict[str, int]
    init_starttime: int
    init_parent_pid: int
    procfs: str
    sigkill_sent: bool = False

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "aka.attempt-process-boundary/v1",
            "policy": ATTEMPT_CONTAINMENT_POLICY,
            "pid_namespace_unshared": True,
            "procfs": self.procfs,
            "namespace_init_pid": self.status["child-pid"],
            "namespace_init_starttime": self.init_starttime,
            "namespace_init_parent_pid": self.init_parent_pid,
            "namespace_init_inner_pid": 1,
            "pid_namespace_id": self.status["pid-namespace"],
            "mount_namespace_id": self.status["mnt-namespace"],
            "ipc_namespace_id": self.status["ipc-namespace"],
            "pidfd_opened": self.init_pidfd >= 0,
            "identity_source": "pinned_bubblewrap_json_status_fd",
        }

    def close(self) -> None:
        descriptor, self.init_pidfd = self.init_pidfd, -1
        if descriptor >= 0:
            os.close(descriptor)
        descriptor, self.status_fd = self.status_fd, -1
        if descriptor >= 0:
            os.close(descriptor)

    def terminate_now(self) -> None:
        """Synchronously target the exact namespace init without numeric PID reuse."""
        if self.init_pidfd < 0 or _pidfd_ready(self.init_pidfd, 0):
            return
        signal.pidfd_send_signal(self.init_pidfd, signal.SIGKILL)
        self.sigkill_sent = True


@dataclass(frozen=True)
class _NamespaceMembershipScan:
    """Visible `/proc` corroboration for authoritative pidfd teardown."""

    enumeration_completed: bool
    inaccessible_entries_count: int
    live_visible_members: tuple[dict[str, Any], ...]

    @property
    def complete(self) -> bool:
        return self.enumeration_completed and self.inaccessible_entries_count == 0


_RUNTIME_ISOLATION_SCHEMA = "aka.runtime-isolation-receipt/v5"
_ZERO_CAPABILITY_FIELDS = ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
_CODEX_REQUIREMENTS_PATH = Path("/etc/codex/requirements.toml")
_CODEX_PERMISSION_PROFILE = "aka_formal_kernel_v1"
_CODEX_REQUIREMENTS_SHA256 = "0c68db4f0ee56b42f15af2896e51f4e667d9d6f86d9d3864dfec571278572ade"
_CODEX_GPU_BWRAP_PATH = (
    Path(__file__).resolve().parents[1] / "agents" / "codex" / "bin" / "bwrap"
)
_CODEX_GPU_BWRAP_SHA256 = "9271bd346d1ea5f878c8f345537e8464a56156b82f956942b66b82feb61791ef"
_CODEX_GPU_BWRAP_SIZE_BYTES = 2381
_CODEX_GPU_REAL_BWRAP_PATH = Path("/usr/bin/bwrap")
_CODEX_GPU_REAL_BWRAP_SHA256 = "d78807229d616606e339c5988392b9e0ab4a6a6998fa51e4590837f426a12fca"
_CODEX_GPU_BWRAP_TRUSTED_DIR = Path("/tmp/aka-codex-gpu-bwrap")
_CODEX_GPU_BWRAP_TRUSTED_PATH = _CODEX_GPU_BWRAP_TRUSTED_DIR / "bwrap"
_DOCKER_MASKED_DIRECTORIES = (
    "/proc/acpi",
    "/proc/asound",
    "/proc/scsi",
    "/sys/devices/virtual/powercap",
    "/sys/firmware",
)
_DOCKER_MASKED_FILES = (
    "/proc/interrupts",
    "/proc/kcore",
    "/proc/keys",
    "/proc/latency_stats",
    "/proc/sched_debug",
    "/proc/timer_list",
    "/proc/timer_stats",
)
_DOCKER_READONLY_PATHS = (
    "/proc/bus",
    "/proc/fs",
    "/proc/irq",
    "/proc/sys",
    "/proc/sysrq-trigger",
)


def _proc_status() -> dict[str, str]:
    try:
        lines = Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CampaignIsolationError("cannot read /proc/self/status") from error
    status: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition(":")
        if separator:
            status[key] = value.strip()
    return status


def _outer_runtime_observation() -> dict[str, Any]:
    """Fail closed unless the outer Docker worker has the pinned security state."""
    status = _proc_status()
    capabilities: dict[str, int] = {}
    for field in _ZERO_CAPABILITY_FIELDS:
        raw = status.get(field, "")
        try:
            value = int(raw, 16)
        except ValueError as error:
            raise CampaignIsolationError(f"invalid {field} in /proc/self/status") from error
        if value != 0:
            raise CampaignIsolationError(f"formal Docker worker retained {field} capabilities")
        capabilities[field] = value

    try:
        no_new_privileges = int(status.get("NoNewPrivs", ""))
        seccomp_mode = int(status.get("Seccomp", ""))
        seccomp_filters = int(status.get("Seccomp_filters", ""))
    except ValueError as error:
        raise CampaignIsolationError("invalid NNP/seccomp state in /proc/self/status") from error
    if no_new_privileges != 1:
        raise CampaignIsolationError("formal Docker worker requires no-new-privileges")
    if seccomp_mode != 0 or seccomp_filters != 0:
        raise CampaignIsolationError(
            "formal Docker worker requires the pinned unconfined seccomp profile"
        )

    try:
        apparmor_profile = Path("/proc/self/attr/current").read_text(
            encoding="utf-8"
        ).strip()
    except OSError as error:
        raise CampaignIsolationError("cannot read the Docker AppArmor profile") from error
    if apparmor_profile != "unconfined":
        raise CampaignIsolationError(
            "formal Docker worker requires the pinned unconfined AppArmor profile"
        )

    try:
        yama_ptrace_scope = int(
            Path("/proc/sys/kernel/yama/ptrace_scope")
            .read_text(encoding="utf-8")
            .strip()
        )
    except (OSError, ValueError) as error:
        raise CampaignIsolationError("cannot prove the Yama ptrace policy") from error
    if yama_ptrace_scope < 1:
        raise CampaignIsolationError(
            "formal campaign requires kernel.yama.ptrace_scope >= 1"
        )
    if os.geteuid() == 0:
        raise CampaignIsolationError("formal Docker worker must run as a non-root UID")

    try:
        pid_namespace = os.readlink("/proc/self/ns/pid")
        ipc_namespace = os.readlink("/proc/self/ns/ipc")
    except OSError as error:
        raise CampaignIsolationError("cannot inspect outer Docker namespaces") from error
    return {
        "effective_uid": os.geteuid(),
        "effective_gid": os.getegid(),
        "supplementary_gids": sorted(set(os.getgroups())),
        "capabilities": capabilities,
        "no_new_privileges": True,
        "seccomp_mode": seccomp_mode,
        "seccomp_filters": seccomp_filters,
        "apparmor_profile": apparmor_profile,
        "yama_ptrace_scope": yama_ptrace_scope,
        "pid_namespace": pid_namespace,
        "ipc_namespace": ipc_namespace,
    }


_ATTEMPT_ESCAPE_PROBE = r"""
import errno
import json
import os
import pathlib
import sys

outer_pid = int(sys.argv[1])
outer_fd = int(sys.argv[2])
sentinel = pathlib.Path(sys.argv[3])
outer_pid_namespace = sys.argv[4]
outer_ipc_namespace = sys.argv[5]
nonce = bytes.fromhex(sys.argv[6])

def read_error(path):
    try:
        with open(path, "rb") as stream:
            stream.read(1)
        return None
    except OSError as error:
        return error.errno

def read_bytes(path):
    try:
        return pathlib.Path(path).read_bytes()
    except OSError:
        return None

def status_value(name):
    for line in pathlib.Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key == name:
            return value.strip()
    return ""

def mount_options(path):
    matches = []
    for line in pathlib.Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) > 6 and fields[4] == path:
            matches.append(fields[5].split(","))
    return matches

def mount_identity(path):
    matches = []
    for line in pathlib.Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 10 or fields[4] != path or "-" not in fields:
            continue
        separator = fields.index("-")
        matches.append({
            "root": fields[3],
            "options": fields[5].split(","),
            "filesystem": fields[separator + 1],
            "source": fields[separator + 2],
        })
    return matches[-1] if matches else None

def masked_directory(path):
    identity = mount_identity(path)
    if identity is None:
        return not pathlib.Path(path).exists()
    try:
        empty = pathlib.Path(path).is_dir() and not any(pathlib.Path(path).iterdir())
    except OSError:
        return False
    return (
        empty
        and identity["filesystem"] == "tmpfs"
        and identity["root"] == "/"
        and "ro" in identity["options"]
    )

def masked_file(path):
    identity = mount_identity(path)
    if identity is None:
        return not pathlib.Path(path).exists()
    return (
        identity["root"] == "/null"
        and identity["filesystem"] in {"tmpfs", "devtmpfs"}
        and "ro" in identity["options"]
    )

def mount_read_only(path):
    try:
        return bool(os.statvfs(path).f_flag & os.ST_RDONLY)
    except FileNotFoundError:
        return True
    except OSError:
        return False

parent_root = f"/proc/{outer_pid}/root{sentinel}"
parent_fd = f"/proc/{outer_pid}/fd/{outer_fd}"
proc_options = mount_options("/proc")
direct_error = read_error(sentinel)
visible_pid_namespaces = set()
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        visible_pid_namespaces.add(os.readlink(entry / "ns/pid"))
    except OSError:
        pass
result = {
    "campaign_data_hidden": direct_error == errno.ENOENT,
    "outer_pid_namespace_absent_from_private_proc": (
        outer_pid_namespace not in visible_pid_namespaces
    ),
    # Numeric PIDs can collide across nested namespaces.  Compare the secret
    # bytes, not errno, so a same-number attempt process cannot create a false
    # isolation proof.
    "parent_root_sentinel_unreachable": read_bytes(parent_root) != nonce,
    "parent_fd_sentinel_unreachable": read_bytes(parent_fd) != nonce,
    "proc_mount_read_write": bool(proc_options) and "rw" in proc_options[-1],
    "pid_namespace_unshared": os.readlink("/proc/self/ns/pid") != outer_pid_namespace,
    "ipc_namespace_unshared": os.readlink("/proc/self/ns/ipc") != outer_ipc_namespace,
    "private_shm": any(
        len(fields) > 6 and fields[4] == "/dev/shm" and "tmpfs" in fields
        for fields in (
            line.split()
            for line in pathlib.Path("/proc/self/mountinfo")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    ),
    "docker_system_paths_remasked": all(
        masked_directory(path)
        for path in (
            "/proc/acpi", "/proc/asound", "/proc/scsi",
            "/sys/devices/virtual/powercap", "/sys/firmware",
        )
    ) and all(
        masked_file(path)
        for path in (
            "/proc/interrupts", "/proc/kcore", "/proc/keys",
            "/proc/latency_stats", "/proc/sched_debug", "/proc/timer_list",
            "/proc/timer_stats",
        )
    ),
    "private_proc_control_writes_blocked": all(
        mount_read_only(path)
        for path in (
            "/proc/bus", "/proc/fs", "/proc/irq", "/proc/sys",
            "/proc/sysrq-trigger",
        )
    ),
    "no_new_privileges": status_value("NoNewPrivs") == "1",
    "effective_capabilities_zero": int(status_value("CapEff"), 16) == 0,
    "bounding_capabilities_zero": int(status_value("CapBnd"), 16) == 0,
    "all_capability_sets_zero": all(
        int(status_value(name), 16) == 0
        for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
    ),
    "seccomp_disabled": (
        status_value("Seccomp") == "0" and status_value("Seccomp_filters") == "0"
    ),
}
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
raise SystemExit(0 if all(result.values()) else 73)
"""


def _sha256_regular_file(path: Path) -> str:
    if not path.is_file():
        raise CampaignIsolationError(f"isolation executable is not a file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise CampaignIsolationError(f"cannot hash isolation executable: {path}") from error
    return digest.hexdigest()


def _codex_requirements_identity() -> tuple[Path, dict[str, Any]]:
    """Validate the immutable managed policy used by both formal treatments."""
    path = _CODEX_REQUIREMENTS_PATH
    try:
        metadata = path.lstat()
    except OSError as error:
        raise CampaignIsolationError("formal Codex requirements are unavailable") from error
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise CampaignIsolationError("formal Codex requirements file is unsafe")
    digest = _sha256_regular_file(path)
    if digest != _CODEX_REQUIREMENTS_SHA256:
        raise CampaignIsolationError("formal Codex requirements violate the pinned policy")
    resolved = path.resolve(strict=True)
    return resolved, {
        "resolved_path": str(resolved),
        "sha256": digest,
        "permission_profile": _CODEX_PERMISSION_PROFILE,
        "agent_requested_sandbox": "workspace-write_legacy_cli",
        "effective_profile_probe": "explicit_named_profile_live",
        "normalization_evidence": "managed_allowlist_plus_pinned_cli_identity",
        "workspace_write": True,
        "credential_path": "~/.codex/auth.json",
        "credential_read": "deny",
        "command_network": "deny",
        "device_access": (
            "sealed_pinned_immutable_path_bwrap_with_docker_device_boundary"
        ),
        "hooks": "disabled",
    }


def _codex_gpu_bwrap_identity() -> tuple[Path, dict[str, Any]]:
    path = _CODEX_GPU_BWRAP_PATH
    try:
        metadata = path.lstat()
    except OSError as error:
        raise CampaignIsolationError("formal Codex GPU bwrap shim is unavailable") from error
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or not metadata.st_mode & stat.S_IXUSR
    ):
        raise CampaignIsolationError("formal Codex GPU bwrap shim is unsafe")
    if metadata.st_size != _CODEX_GPU_BWRAP_SIZE_BYTES:
        raise CampaignIsolationError("formal Codex GPU bwrap shim violates its pin")
    digest = _sha256_regular_file(path)
    if digest != _CODEX_GPU_BWRAP_SHA256:
        raise CampaignIsolationError("formal Codex GPU bwrap shim violates its pin")
    resolved = path.resolve(strict=True)
    try:
        real_metadata = _CODEX_GPU_REAL_BWRAP_PATH.lstat()
    except OSError as error:
        raise CampaignIsolationError("formal Codex real bwrap is unavailable") from error
    if (
        _CODEX_GPU_REAL_BWRAP_PATH.is_symlink()
        or not stat.S_ISREG(real_metadata.st_mode)
        or real_metadata.st_nlink != 1
        or not real_metadata.st_mode & stat.S_IXUSR
    ):
        raise CampaignIsolationError("formal Codex real bwrap is unsafe")
    real_digest = _sha256_regular_file(_CODEX_GPU_REAL_BWRAP_PATH)
    if real_digest != _CODEX_GPU_REAL_BWRAP_SHA256:
        raise CampaignIsolationError("formal Codex real bwrap violates its pin")
    return resolved, {
        "resolved_path": str(resolved),
        "sha256": digest,
        "size_bytes": _CODEX_GPU_BWRAP_SIZE_BYTES,
        "interpreter": "/usr/bin/python3 -I",
        "real_bwrap": str(_CODEX_GPU_REAL_BWRAP_PATH),
        "real_bwrap_sha256": real_digest,
        "sandbox_mounted_path": str(_CODEX_GPU_BWRAP_TRUSTED_PATH),
        "mount_transport": (
            "sealed_memfd_ro_bind_data_under_remounted_ro_tmpfs"
        ),
        "device_policy": "docker_visible_kfd_and_render_nodes_only",
    }


def _sealed_codex_gpu_bwrap(source: Path) -> int:
    """Copy the pinned shim into an immutable anonymous file for exact binding."""
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    source_descriptor = -1
    sealed_descriptor = -1
    try:
        source_descriptor = os.open(source, flags)
        metadata = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise CampaignIsolationError("formal Codex GPU bwrap shim is unsafe")
        if metadata.st_size != _CODEX_GPU_BWRAP_SIZE_BYTES:
            raise CampaignIsolationError("formal Codex GPU bwrap shim violates its pin")
        payload = bytearray()
        while len(payload) <= _CODEX_GPU_BWRAP_SIZE_BYTES:
            chunk = os.read(
                source_descriptor,
                _CODEX_GPU_BWRAP_SIZE_BYTES + 1 - len(payload),
            )
            if not chunk:
                break
            payload.extend(chunk)
        if (
            len(payload) != _CODEX_GPU_BWRAP_SIZE_BYTES
            or hashlib.sha256(payload).hexdigest() != _CODEX_GPU_BWRAP_SHA256
        ):
            raise CampaignIsolationError("formal Codex GPU bwrap shim violates its pin")

        sealed_descriptor = os.memfd_create(
            "aka-codex-gpu-bwrap",
            os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
        )
        remaining = memoryview(payload)
        while remaining:
            written = os.write(sealed_descriptor, remaining)
            if written <= 0:
                raise CampaignIsolationError("cannot materialize formal Codex GPU bwrap")
            remaining = remaining[written:]
        os.fchmod(sealed_descriptor, 0o555)
        seals = (
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SEAL
        )
        fcntl.fcntl(sealed_descriptor, fcntl.F_ADD_SEALS, seals)
        if fcntl.fcntl(sealed_descriptor, fcntl.F_GET_SEALS) != seals:
            raise CampaignIsolationError("formal Codex GPU bwrap memfd is not fully sealed")
        os.lseek(sealed_descriptor, 0, os.SEEK_SET)
        result, sealed_descriptor = sealed_descriptor, -1
        return result
    except (AttributeError, OSError) as error:
        raise CampaignIsolationError(
            "cannot create sealed formal Codex GPU bwrap transport"
        ) from error
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        if sealed_descriptor >= 0:
            os.close(sealed_descriptor)


_CODEX_SANDBOX_PROBE = r"""
import errno
import json
import os
import pathlib
import socket
import sys

workspace = pathlib.Path(sys.argv[1])
credential = pathlib.Path(sys.argv[2])
outer_pid_namespace = sys.argv[3]
trusted_gpu_bwrap = pathlib.Path(sys.argv[4])
marker = workspace / "managed-sandbox-write-probe"

def read_error(path):
    try:
        pathlib.Path(path).read_bytes()
    except OSError as error:
        return error.errno
    return None

try:
    credential.read_bytes()
    credential_errno = None
except OSError as error:
    credential_errno = error.errno

try:
    marker.write_text("ok", encoding="utf-8")
    workspace_write = marker.read_text(encoding="utf-8") == "ok"
except OSError:
    workspace_write = False

network_errno = None
sock = None
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    network_errno = sock.connect_ex(("1.1.1.1", 443))
except OSError as error:
    network_errno = error.errno
finally:
    if sock is not None:
        sock.close()

mutation_denied_errors = {errno.EACCES, errno.EPERM, errno.EROFS, errno.EBUSY}
renamed_trusted_dir = trusted_gpu_bwrap.parent.with_name(
    trusted_gpu_bwrap.parent.name + "-renamed"
)
try:
    trusted_gpu_bwrap.parent.rename(renamed_trusted_dir)
    trusted_dir_rename_errno = None
except OSError as error:
    trusted_dir_rename_errno = error.errno

try:
    trusted_gpu_bwrap.unlink()
    trusted_path_unlink_errno = None
except OSError as error:
    trusted_path_unlink_errno = error.errno

renamed_trusted_path = trusted_gpu_bwrap.with_name(
    trusted_gpu_bwrap.name + "-renamed"
)
try:
    trusted_gpu_bwrap.rename(renamed_trusted_path)
    trusted_path_rename_errno = None
except OSError as error:
    trusted_path_rename_errno = error.errno

replacement = workspace / "replacement-bwrap"
replacement.write_bytes(b"malicious")
try:
    replacement.replace(trusted_gpu_bwrap)
    trusted_path_replace_errno = None
except OSError as error:
    trusted_path_replace_errno = error.errno

try:
    descriptor = os.open(trusted_gpu_bwrap, os.O_WRONLY | os.O_TRUNC)
    os.close(descriptor)
    trusted_path_write_errno = None
except OSError as error:
    trusted_path_write_errno = error.errno

render_nodes = sorted(pathlib.Path("/dev/dri").glob("renderD*"))
assigned_gpu_devices_visible = pathlib.Path("/dev/kfd").exists() and bool(render_nodes)
device_descriptors = []
device_open_error = None
try:
    for device in (pathlib.Path("/dev/kfd"), *render_nodes):
        device_descriptors.append(os.open(device, os.O_RDWR | os.O_CLOEXEC))
    assigned_gpu_devices_writable = bool(render_nodes)
except OSError as error:
    device_open_error = repr(error)
    assigned_gpu_devices_writable = False
finally:
    for descriptor in device_descriptors:
        os.close(descriptor)

torch_probe_error = None
try:
    import torch
    single_gpu_runtime_visible = torch.cuda.is_available() and torch.cuda.device_count() == 1
    if single_gpu_runtime_visible:
        values = torch.arange(16, dtype=torch.float32, device="cuda")
        gpu_compute_probe_passed = values.sum().item() == 120.0
    else:
        gpu_compute_probe_passed = False
except Exception as error:
    torch_probe_error = repr(error)
    single_gpu_runtime_visible = False
    gpu_compute_probe_passed = False

blocked_proc_errors = {errno.EACCES, errno.EPERM, errno.ENOENT}
result = {
    "workspace_write_enforced": workspace_write,
    "credential_read_denied": credential_errno in {errno.EACCES, errno.EPERM, errno.ENOENT},
    "command_network_denied": network_errno in {
        errno.EACCES,
        errno.EPERM,
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.EHOSTUNREACH,
    },
    "command_not_in_worker_pid_namespace": (
        os.readlink("/proc/self/ns/pid") != outer_pid_namespace
    ),
    # /proc/1 is the closest PID-namespace init visible to the managed command.
    # These probes close the common /proc/<pid>/root credential-bypass route
    # without assuming host and namespace PID numbers cannot collide.
    "pid1_root_alias_credential_blocked": (
        read_error(f"/proc/1/root{credential}") in blocked_proc_errors
    ),
    "pid1_environ_blocked": (
        read_error("/proc/1/environ") in blocked_proc_errors
    ),
    "pid1_mem_blocked": read_error("/proc/1/mem") in blocked_proc_errors,
    "pinned_gpu_bwrap_active": os.environ.get("AKA_CODEX_GPU_BWRAP_ACTIVE") == "1",
    "gpu_bwrap_directory_immutable": (
        trusted_dir_rename_errno in mutation_denied_errors
    ),
    "gpu_bwrap_path_immutable": all(
        value in mutation_denied_errors
        for value in (
            trusted_path_unlink_errno,
            trusted_path_rename_errno,
            trusted_path_write_errno,
        )
    ) and trusted_path_replace_errno in (
        mutation_denied_errors | {errno.EXDEV}
    ),
    "assigned_gpu_devices_visible": assigned_gpu_devices_visible,
    "assigned_gpu_devices_writable": assigned_gpu_devices_writable,
    "single_gpu_runtime_visible": single_gpu_runtime_visible,
    "gpu_compute_probe_passed": gpu_compute_probe_passed,
}
if not (
    result["gpu_bwrap_directory_immutable"]
    and result["gpu_bwrap_path_immutable"]
    and assigned_gpu_devices_visible
    and assigned_gpu_devices_writable
    and single_gpu_runtime_visible
    and gpu_compute_probe_passed
):
    device_metadata = {}
    for device in (pathlib.Path("/dev/kfd"), *render_nodes):
        try:
            metadata = device.stat()
            device_metadata[str(device)] = {
                "uid": metadata.st_uid,
                "gid": metadata.st_gid,
                "mode": oct(metadata.st_mode & 0o777),
            }
        except OSError as error:
            device_metadata[str(device)] = {"stat_error": repr(error)}
    print(json.dumps({
        "gpu_probe_diagnostic": {
            "euid": os.geteuid(),
            "egid": os.getegid(),
            "groups": os.getgroups(),
            "devices": device_metadata,
            "device_open_error": device_open_error,
            "torch_probe_error": torch_probe_error,
            "trusted_gpu_bwrap_mutation_errnos": {
                "rename_directory": trusted_dir_rename_errno,
                "unlink_path": trusted_path_unlink_errno,
                "rename_path": trusted_path_rename_errno,
                "replace_path": trusted_path_replace_errno,
                "write_path": trusted_path_write_errno,
            },
        }
    }, sort_keys=True, separators=(",", ":")), file=sys.stderr)
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
raise SystemExit(0 if all(result.values()) else 74)
"""


def _bubblewrap_identity() -> tuple[Path, dict[str, str]]:
    discovered = shutil.which("bwrap")
    if not discovered:
        raise CampaignIsolationError("formal campaign requires bubblewrap (bwrap)")
    try:
        binary = Path(discovered).resolve(strict=True)
        completed = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CampaignIsolationError("cannot identify bubblewrap") from error
    version = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0 or not version:
        raise CampaignIsolationError("bubblewrap version is unavailable")
    return binary, {
        "resolved_path": str(binary),
        "sha256": _sha256_regular_file(binary),
        "version": version,
    }


def _bubblewrap_base_command(
    binary: str | Path,
    data_root: Path,
    *,
    private_proc: bool = True,
) -> list[str]:
    """Pinned mount/namespace boundary shared by probes and real attempts."""
    command = [
        str(binary),
        "--die-with-parent",
        "--unshare-ipc",
        "--unshare-pid",
        "--ro-bind",
        "/",
        "/",
        "--dev-bind",
        "/dev",
        "/dev",
        "--tmpfs",
        "/dev/shm",
    ]
    if private_proc:
        command.extend(["--proc", "/proc"])
        # A replaced procfs no longer carries Docker's protected submounts.
        # Recreate them before the untrusted direct backend can execute.
        for path in _DOCKER_MASKED_DIRECTORIES:
            if Path(path).is_dir():
                command.extend(["--tmpfs", path, "--remount-ro", path])
        for path in _DOCKER_MASKED_FILES:
            if Path(path).exists():
                command.extend(["--ro-bind", "/dev/null", path])
        for path in _DOCKER_READONLY_PATHS:
            if Path(path).exists():
                command.extend(["--ro-bind", path, path])
    else:
        # Apex is trusted at this layer and must create a nested user namespace
        # around its untrusted backend.  The recursive read-only root bind above
        # otherwise makes /proc/<pid>/uid_map unwritable and causes inner bwrap
        # to fail before it can establish that stronger boundary.
        command.extend(["--bind", "/proc", "/proc"])
    command.extend(["--tmpfs", "/tmp", "--tmpfs", str(data_root)])
    return command


def _mount_codex_gpu_bwrap(command: list[str], descriptor: int) -> None:
    """Expose sealed shim bytes on an immutable private mount outside Codex's cwd."""
    command.extend(
        [
            "--perms",
            "0555",
            "--dir",
            str(_CODEX_GPU_BWRAP_TRUSTED_DIR),
            "--tmpfs",
            str(_CODEX_GPU_BWRAP_TRUSTED_DIR),
            "--perms",
            "0555",
            "--ro-bind-data",
            str(descriptor),
            str(_CODEX_GPU_BWRAP_TRUSTED_PATH),
            "--remount-ro",
            str(_CODEX_GPU_BWRAP_TRUSTED_DIR),
        ]
    )


def _codex_cli_identity() -> tuple[Path, dict[str, str]]:
    discovered = shutil.which("codex")
    if not discovered:
        raise CampaignIsolationError("formal campaign requires the Codex CLI")
    try:
        binary = Path(discovered).resolve(strict=True)
        completed = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CampaignIsolationError("cannot identify the Codex CLI") from error
    version = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0 or not version:
        raise CampaignIsolationError("Codex CLI version is unavailable")
    return binary, {
        "resolved_path": str(binary),
        "sha256": _sha256_regular_file(binary),
        "version": version,
    }


def _codex_sandbox_probe(
    *,
    codex_binary: Path,
    bubblewrap_binary: Path,
    gpu_bubblewrap_binary: Path,
    data_root: Path,
) -> dict[str, bool]:
    """Exercise the exact outer mount boundary and Codex managed command policy."""
    probe_dir = Path(tempfile.mkdtemp(prefix=".aka-codex-sandbox-probe-", dir=data_root))
    home = probe_dir / "home"
    workspace = probe_dir / "workspace"
    credential = home / ".codex" / "auth.json"
    gpu_bwrap_descriptor = -1
    try:
        credential.parent.mkdir(parents=True)
        workspace.mkdir()
        credential.write_text('{"fixture":"must-not-be-readable"}\n', encoding="utf-8")
        outer_pid_namespace = os.readlink("/proc/self/ns/pid")
        command = _bubblewrap_base_command(bubblewrap_binary, data_root)
        gpu_bwrap_descriptor = _sealed_codex_gpu_bwrap(gpu_bubblewrap_binary)
        _mount_codex_gpu_bwrap(command, gpu_bwrap_descriptor)
        current = data_root
        for part in probe_dir.relative_to(data_root).parts:
            current /= part
            command.extend(["--dir", str(current)])
        command.extend(
            [
                "--bind",
                str(probe_dir),
                str(probe_dir),
                "--",
                str(codex_binary),
                "sandbox",
                "--include-managed-config",
                "--permission-profile",
                _CODEX_PERMISSION_PROFILE,
                "-C",
                str(workspace),
                "--",
                sys.executable,
                "-c",
                _CODEX_SANDBOX_PROBE,
                str(workspace),
                str(credential),
                outer_pid_namespace,
                str(_CODEX_GPU_BWRAP_TRUSTED_PATH),
            ]
        )
        environment = dict(os.environ)
        environment["PATH"] = (
            f"{_CODEX_GPU_BWRAP_TRUSTED_DIR}{os.pathsep}"
            f"{environment.get('PATH', '')}"
        )
        environment.update(
            {
                "HOME": str(home),
                "CODEX_HOME": str(home / ".codex"),
                "XDG_CONFIG_HOME": str(home / ".config"),
                "XDG_STATE_HOME": str(home / ".local/state"),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
                env=environment,
                pass_fds=(gpu_bwrap_descriptor,),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise CampaignIsolationError("Codex managed sandbox probe failed to run") from error
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        try:
            result = json.loads(lines[-1]) if lines else None
        except json.JSONDecodeError as error:
            raise CampaignIsolationError(
                f"Codex managed sandbox probe emitted invalid evidence: {completed.stderr.strip()}"
            ) from error
        expected_keys = {
            "workspace_write_enforced",
            "credential_read_denied",
            "command_network_denied",
            "command_not_in_worker_pid_namespace",
            "pid1_root_alias_credential_blocked",
            "pid1_environ_blocked",
            "pid1_mem_blocked",
            "pinned_gpu_bwrap_active",
            "gpu_bwrap_directory_immutable",
            "gpu_bwrap_path_immutable",
            "assigned_gpu_devices_visible",
            "assigned_gpu_devices_writable",
            "single_gpu_runtime_visible",
            "gpu_compute_probe_passed",
        }
        if (
            completed.returncode != 0
            or not isinstance(result, dict)
            or set(result) != expected_keys
            or any(result.get(key) is not True for key in expected_keys)
        ):
            raise CampaignIsolationError(
                "Codex managed sandbox probe failed closed: "
                f"result={result!r} stderr={completed.stderr.strip()!r}"
            )
        return result
    finally:
        if gpu_bwrap_descriptor >= 0:
            os.close(gpu_bwrap_descriptor)
        shutil.rmtree(probe_dir, ignore_errors=False)


def _attempt_escape_probe(
    *, binary: Path, data_root: Path, outer: dict[str, Any]
) -> dict[str, bool]:
    raw_root = Path(os.environ.get("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", ""))
    if not raw_root.is_absolute() or raw_root == Path("/") or raw_root.is_symlink():
        raise CampaignIsolationError("formal campaign data root is unsafe")
    try:
        resolved_root = raw_root.resolve(strict=True)
    except OSError as error:
        raise CampaignIsolationError("formal campaign data root is unavailable") from error
    if resolved_root != data_root or not data_root.is_dir():
        raise CampaignIsolationError("formal campaign data root changed during isolation proof")

    probe_dir = Path(tempfile.mkdtemp(prefix=".aka-isolation-probe-", dir=data_root))
    sentinel = probe_dir / "sentinel"
    descriptor = -1
    try:
        nonce = os.urandom(32)
        sentinel.write_bytes(nonce)
        descriptor = os.open(sentinel, os.O_RDONLY | os.O_CLOEXEC)
        parent_root_alias = Path(f"/proc/{os.getpid()}/root{sentinel}")
        parent_fd_alias = Path(f"/proc/{os.getpid()}/fd/{descriptor}")
        try:
            root_bytes = parent_root_alias.read_bytes()
            fd_bytes = parent_fd_alias.read_bytes()
        except OSError as error:
            raise CampaignIsolationError(
                "outer runtime cannot establish readable parent /proc aliases"
            ) from error
        if root_bytes != nonce or fd_bytes != nonce:
            raise CampaignIsolationError("outer parent /proc aliases do not bind the sentinel")
        command = _bubblewrap_base_command(binary, data_root) + [
            "--",
            sys.executable,
            "-c",
            _ATTEMPT_ESCAPE_PROBE,
            str(os.getpid()),
            str(descriptor),
            str(sentinel),
            str(outer["pid_namespace"]),
            str(outer["ipc_namespace"]),
            nonce.hex(),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise CampaignIsolationError("bubblewrap isolation probe failed to run") from error
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise CampaignIsolationError(
                f"bubblewrap isolation probe emitted invalid evidence: {completed.stderr.strip()}"
            ) from error
        if completed.returncode != 0 or not isinstance(result, dict):
            raise CampaignIsolationError(
                f"bubblewrap isolation probe failed closed: {result!r}"
            )
        expected_keys = {
            "campaign_data_hidden",
            "outer_pid_namespace_absent_from_private_proc",
            "parent_root_sentinel_unreachable",
            "parent_fd_sentinel_unreachable",
            "proc_mount_read_write",
            "pid_namespace_unshared",
            "ipc_namespace_unshared",
            "private_shm",
            "docker_system_paths_remasked",
            "private_proc_control_writes_blocked",
            "no_new_privileges",
            "effective_capabilities_zero",
            "bounding_capabilities_zero",
            "all_capability_sets_zero",
            "seccomp_disabled",
        }
        if set(result) != expected_keys or any(
            not isinstance(result[key], bool) for key in expected_keys
        ):
            raise CampaignIsolationError("bubblewrap isolation probe evidence is malformed")
        if any(result[key] is not True for key in expected_keys):
            raise CampaignIsolationError("bubblewrap isolation proof is incomplete")
        return result
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        sentinel.unlink(missing_ok=True)
        try:
            probe_dir.rmdir()
        except OSError as error:
            raise CampaignIsolationError("cannot remove the isolation probe directory") from error


def runtime_isolation_receipt() -> dict[str, Any]:
    """Return invariant, live evidence for the formal Docker+bwrap boundary."""
    raw_root = Path(os.environ.get("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", ""))
    try:
        data_root = raw_root.resolve(strict=True)
    except OSError as error:
        raise CampaignIsolationError("formal campaign data root is unavailable") from error
    outer = _outer_runtime_observation()
    binary, bubblewrap = _bubblewrap_identity()
    _requirements_path, codex_requirements = _codex_requirements_identity()
    gpu_bubblewrap_binary, gpu_bubblewrap = _codex_gpu_bwrap_identity()
    codex_binary, codex_cli = _codex_cli_identity()
    attempt = _attempt_escape_probe(binary=binary, data_root=data_root, outer=outer)
    codex_sandbox = _codex_sandbox_probe(
        codex_binary=codex_binary,
        bubblewrap_binary=binary,
        gpu_bubblewrap_binary=gpu_bubblewrap_binary,
        data_root=data_root,
    )
    # Namespace inode identifiers are deliberately used only by the live probe:
    # they differ for every Docker worker and would make the Apex/Codex
    # comparison contract run-specific. The manifest records only the verified
    # relationships (attempt PID/IPC unshared and private procfs active).
    recorded_outer = {
        key: value
        for key, value in outer.items()
        if key not in {"pid_namespace", "ipc_namespace"}
    }
    return {
        "schema": _RUNTIME_ISOLATION_SCHEMA,
        "policy": {
            "docker_user": "non_root",
            "docker_capabilities": "drop_all",
            "docker_no_new_privileges": True,
            "docker_apparmor": "unconfined_for_rootless_userns",
            "docker_seccomp": "unconfined_for_rootless_userns",
            "docker_systempaths": "unconfined_for_private_attempt_procfs",
            "docker_masked_paths_rebuilt": [
                *_DOCKER_MASKED_DIRECTORIES,
                *_DOCKER_MASKED_FILES,
            ],
            "docker_readonly_paths_rebuilt": list(_DOCKER_READONLY_PATHS),
            "docker_pid_namespace": "private_default",
            "attempt_mount_namespace": "bubblewrap",
            "attempt_pid_namespace": "private_per_attempt_with_bwrap_reaper_pid1",
            "attempt_ipc_namespace": "unshared",
            "attempt_proc": "private_procfs_for_attempt_pid_namespace",
            "direct_agent_proc": "aka_outer_private_attempt_procfs",
            "apex_outer_proc": (
                "trusted_orchestrator_inherited_worker_procfs_nested_userns_writable"
            ),
            "apex_backend_proc": "apex_inner_private_attempt_procfs_required",
            "process_lifetime_boundary": "namespace_init_pidfd_v1",
            "proc_escape_guard": "outer_process_absent_from_private_procfs_v1",
            "command_sandbox": "codex_managed_permission_profile_bwrap",
            "command_pid_namespace": (
                "nested_codex_unshared_inside_private_attempt_pidns_v1"
            ),
            "command_network": "managed_profile_denied_live_probe_v1",
            "command_gpu_access": (
                "sealed_memfd_immutable_path_bwrap_and_single_gpu_probe_v1"
            ),
            "credential_read": "denied_by_managed_permission_profile",
        },
        "outer_runtime": recorded_outer,
        "bubblewrap": bubblewrap,
        "codex_gpu_bubblewrap": gpu_bubblewrap,
        "codex_cli": codex_cli,
        "codex_requirements": codex_requirements,
        "attempt_probe": attempt,
        "codex_sandbox_probe": codex_sandbox,
    }


def is_formal_campaign(eval_config: dict[str, Any]) -> bool:
    campaign = eval_config.get("campaign") or {}
    attempt = eval_config.get("campaign_attempt") or {}
    return (
        isinstance(campaign, dict)
        and campaign.get("comparison") == "apex_vs_codex"
        and isinstance(attempt, dict)
        and attempt.get("fresh_session") is True
    )


def formal_gpu_evidence(eval_config: dict[str, Any]) -> dict[str, Any] | None:
    """Validate runner-owned physical-device and exclusivity receipts."""
    if not is_formal_campaign(eval_config):
        return None
    plan_path = Path(os.environ.get("AGENT_KERNEL_ARENA_GPU_BOUNDARY_PLAN", ""))
    boundary_receipt_path = Path(
        os.environ.get("AGENT_KERNEL_ARENA_GPU_BOUNDARY_RECEIPT", "")
    )
    lease_receipt_path = Path(
        os.environ.get("AGENT_KERNEL_ARENA_GPU_EXCLUSIVITY_RECEIPT", "")
    )
    expected_plan_digest = os.environ.get(
        "AGENT_KERNEL_ARENA_GPU_BOUNDARY_PLAN_SHA256", ""
    )
    expected_lease_digest = os.environ.get(
        "AGENT_KERNEL_ARENA_GPU_EXCLUSIVITY_RECEIPT_SHA256", ""
    )
    host_gpu_id = os.environ.get("AGENT_KERNEL_ARENA_HOST_GPU_ID", "")
    if any(
        not path.is_absolute()
        for path in (plan_path, boundary_receipt_path, lease_receipt_path)
    ):
        raise CampaignIsolationError("formal GPU evidence paths must be absolute")
    try:
        plan = load_plan(plan_path)
        selected = selected_device(plan, host_gpu_id)
        boundary = json.loads(boundary_receipt_path.read_text(encoding="utf-8"))
        lease = load_receipt(
            lease_receipt_path, expected_plan_sha256=expected_plan_digest
        )
    except (GpuBoundaryError, GpuExclusivityError, OSError, json.JSONDecodeError) as error:
        raise CampaignIsolationError(f"formal GPU evidence is invalid: {error}") from error
    if not isinstance(boundary, dict):
        raise CampaignIsolationError("formal GPU boundary receipt must be an object")
    boundary_material = {key: value for key, value in boundary.items() if key != "sha256"}
    if (
        plan.get("sha256") != expected_plan_digest
        or boundary.get("schema") != GPU_BOUNDARY_RECEIPT_SCHEMA
        or boundary.get("sha256") != canonical_digest(boundary_material)
        or boundary.get("plan_sha256") != expected_plan_digest
        or boundary.get("host_gpu_id") != host_gpu_id
        or boundary.get("unique_id") != selected.get("unique_id")
        or boundary.get("verified") is not True
        or boundary.get("runtime_verified") is not True
        or lease.get("sha256") != expected_lease_digest
        or lease.get("exclusivity_verified") is not True
    ):
        raise CampaignIsolationError("formal GPU evidence does not match the selected worker")
    return {
        "policy": "physical_device_boundary_with_host_exclusivity_v1",
        "plan_sha256": expected_plan_digest,
        "boundary_receipt_sha256": boundary["sha256"],
        "exclusivity_receipt_sha256": lease["sha256"],
        "exclusivity_verified": True,
        "host_gpu_id": host_gpu_id,
        "unique_id": selected["unique_id"],
        "allowed_render_nodes": [
            render["path"] for render in selected["render_nodes"]
        ],
        "observed_devices": boundary.get("observed_devices"),
        "runtime_identity": boundary.get("runtime_identity"),
    }


def prepare_attempt_home(
    eval_config: dict[str, Any],
    *,
    backend: str,
) -> Path | None:
    """Copy only the selected backend login state into a fresh attempt home."""
    if not is_formal_campaign(eval_config):
        return None
    attempt = eval_config["campaign_attempt"]
    receipt_path = Path(str(attempt.get("receipt_path", "")))
    if not receipt_path.is_absolute():
        raise CampaignIsolationError("formal campaign receipt_path must be absolute")
    home = receipt_path.parent / ".agent-home"
    if home.exists():
        raise CampaignIsolationError(f"attempt home already exists: {home}")
    home.mkdir(mode=0o700)

    state_root = Path(os.environ.get("AGENT_STATE_MOUNT_ROOT", "/opt/aka-agent-state"))
    selected = {
        # A formal session receives authentication only. Copying the complete
        # Codex home would expose prior transcripts, memories, caches, rules,
        # and user configuration even when the CLI is asked not to load them.
        "codex": (".codex/auth.json",),
        "claude": (".claude", ".claude.json"),
        "cursor": (".cursor", ".config/cursor"),
    }.get(backend)
    if selected is None:
        raise CampaignIsolationError(f"unsupported isolated backend: {backend}")
    for relative in selected:
        source = state_root / relative
        if not source.exists():
            continue
        _assert_safe_source(source)
        destination = home / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, symlinks=False)
        elif source.is_file() and not source.is_symlink():
            shutil.copy2(source, destination, follow_symlinks=False)
        else:
            raise CampaignIsolationError(f"unsafe backend state path: {source}")
    _make_owner_writable(home)
    return home


def _assert_safe_source(source: Path) -> None:
    source_metadata = source.lstat()
    if stat.S_ISLNK(source_metadata.st_mode) or not (
        stat.S_ISDIR(source_metadata.st_mode)
        or stat.S_ISREG(source_metadata.st_mode)
    ):
        raise CampaignIsolationError(f"unsafe backend state path: {source}")
    candidates = tuple(source.rglob("*")) if stat.S_ISDIR(source_metadata.st_mode) else ()
    for path in candidates:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not (
            stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
        ):
            raise CampaignIsolationError(f"unsafe backend state path: {path}")


def isolated_environment(environment: dict[str, str], home: Path | None) -> dict[str, str]:
    if home is None:
        return environment
    updated = dict(environment)
    updated["HOME"] = str(home)
    updated["CODEX_HOME"] = str(home / ".codex")
    updated["XDG_CONFIG_HOME"] = str(home / ".config")
    updated["XDG_STATE_HOME"] = str(home / ".local/state")
    updated["PYTHONDONTWRITEBYTECODE"] = "1"
    _gpu_bubblewrap, _identity = _codex_gpu_bwrap_identity()
    updated["PATH"] = (
        f"{_CODEX_GPU_BWRAP_TRUSTED_DIR}{os.pathsep}{updated.get('PATH', '')}"
    )
    return updated


def wrap_attempt_command(
    command: list[str],
    *,
    eval_config: dict[str, Any],
    writable_roots: Iterable[Path],
    read_only_roots: Iterable[Path] = (),
    trusted_read_only_roots: Iterable[Path] = (),
    private_proc: bool = True,
) -> list[str]:
    """Hide campaign data and expose attempt roots plus pre-authenticated roots.

    ``trusted_read_only_roots`` is intentionally not a general escape hatch: its
    caller must bind each root to an independent identity contract and persist the
    returned mount receipt.
    """
    if not is_formal_campaign(eval_config):
        return command
    # Re-resolve, hash, and version-check the read-only mounted executable at
    # attempt construction time instead of trusting a prior PATH lookup.
    binary, _identity = _bubblewrap_identity()
    _codex_requirements_identity()
    gpu_bubblewrap, _gpu_bubblewrap_identity = _codex_gpu_bwrap_identity()
    data_root_raw = os.environ.get("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", "")
    data_root = Path(data_root_raw)
    if not data_root.is_absolute() or not data_root.is_dir():
        raise CampaignIsolationError("formal campaign data root is unavailable")
    data_root = data_root.resolve(strict=True)

    writable = _validated_attempt_roots(
        writable_roots, data_root=data_root, label="writable"
    )
    read_only = _validated_attempt_roots(
        read_only_roots, data_root=data_root, label="read-only"
    )
    trusted_read_only = _validated_trusted_read_only_roots(
        trusted_read_only_roots, data_root=data_root
    )
    if not writable:
        raise CampaignIsolationError("formal campaign has no attempt writable root")
    _reject_overlapping_roots((*writable, *read_only, *trusted_read_only))

    wrapped = _bubblewrap_base_command(
        binary, data_root, private_proc=private_proc
    )
    state_root = Path(
        os.environ.get("AGENT_STATE_MOUNT_ROOT", "/opt/aka-agent-state")
    )
    if state_root.is_absolute() and state_root.is_dir():
        # Authentication was copied into the fresh attempt home. Hide the
        # original read-only mount so prior history, memories, rules, caches,
        # and user configuration cannot be recovered through its old path.
        wrapped.extend(["--tmpfs", str(state_root.resolve(strict=True))])
    created: set[Path] = set()
    for root in (*read_only, *writable):
        relative = root.relative_to(data_root)
        current = data_root
        for part in relative.parts:
            current /= part
            if current not in created:
                wrapped.extend(["--dir", str(current)])
                created.add(current)
    for root in trusted_read_only:
        if root.is_relative_to(Path("/tmp")):
            current = Path("/tmp")
            for part in root.relative_to(current).parts:
                current /= part
                if current not in created:
                    wrapped.extend(["--dir", str(current)])
                    created.add(current)
    # bubblewrap opens bind sources in its parent namespace before applying the
    # destination mounts, so these still refer to host roots after data_root is
    # hidden by the tmpfs above.
    for root in read_only:
        wrapped.extend(["--ro-bind", str(root), str(root)])
    for root in trusted_read_only:
        wrapped.extend(["--ro-bind", str(root), str(root)])
    for root in writable:
        wrapped.extend(["--bind", str(root), str(root)])

    # These are user-owned trees and are intentionally excluded from the formal
    # Git-clean check. Hide them rather than exposing unmanifested observations.
    workdir = Path(os.environ.get("AGENT_KERNEL_ARENA_WORKDIR", "/workspace"))
    for relative in (".eval-tool-artifacts", "experiments"):
        path = workdir / relative
        if path.is_dir():
            wrapped.extend(["--tmpfs", str(path)])
    gpu_bwrap_descriptor = _sealed_codex_gpu_bwrap(gpu_bubblewrap)
    status_read = status_write = gate_read = gate_write = -1
    try:
        status_read, status_write = os.pipe2(os.O_CLOEXEC)
        gate_read, gate_write = os.pipe2(os.O_CLOEXEC)
        _mount_codex_gpu_bwrap(wrapped, gpu_bwrap_descriptor)
        wrapped.extend(
            [
                "--json-status-fd",
                str(status_write),
                "--block-fd",
                str(gate_read),
            ]
        )
        wrapped.extend(["--", *command])
        result = WrappedAttemptCommand(
            wrapped,
            pass_fds=(gpu_bwrap_descriptor, status_write, gate_read),
            boundary_status_fd=status_read,
            boundary_gate_fd=gate_write,
            boundary_procfs=(
                "private_attempt_procfs"
                if private_proc
                else "trusted_orchestrator_inherited_procfs"
            ),
            mount_receipt=_build_attempt_mount_receipt(
                data_root=data_root,
                writable=writable,
                read_only=read_only,
                trusted_read_only=trusted_read_only,
            ),
        )
        gpu_bwrap_descriptor = status_write = gate_read = status_read = gate_write = -1
        return result
    except Exception:
        raise
    finally:
        for descriptor in (
            gpu_bwrap_descriptor,
            status_read,
            status_write,
            gate_read,
            gate_write,
        ):
            if descriptor >= 0:
                os.close(descriptor)


def attempt_command_pass_fds(command: list[str]) -> tuple[int, ...]:
    """Return only descriptors explicitly owned by a wrapped attempt command."""
    if isinstance(command, WrappedAttemptCommand):
        return command.pass_fds
    return ()


def release_attempt_command_fds(command: list[str]) -> None:
    """Release parent copies after subprocess creation; safe to call repeatedly."""
    if isinstance(command, WrappedAttemptCommand):
        command.release_pass_fds()


def _pidfd_ready(descriptor: int, timeout_seconds: float) -> bool:
    poller = select.poll()
    poller.register(descriptor, select.POLLIN | select.POLLHUP | select.POLLERR)
    return bool(poller.poll(max(0, int(timeout_seconds * 1000))))


def _process_starttime(pid: int) -> int:
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat_line[stat_line.rfind(")") + 2 :].split()
        value = int(fields[19])
    except (OSError, ValueError, IndexError) as error:
        raise CampaignIsolationError("cannot bind attempt namespace init starttime") from error
    if value <= 0:
        raise CampaignIsolationError("attempt namespace init starttime is invalid")
    return value


def _process_namespace_identity(pid: int) -> tuple[int, int]:
    try:
        lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
        values = {
            key: value.strip()
            for line in lines
            for key, separator, value in [line.partition(":")]
            if separator and key in {"PPid", "NSpid"}
        }
        parent = int(values["PPid"])
        namespace_pid = int(values["NSpid"].split()[-1])
    except (OSError, KeyError, ValueError, IndexError) as error:
        raise CampaignIsolationError("cannot bind attempt namespace process identity") from error
    return parent, namespace_pid


def _pid_namespace_membership_scan(
    namespace_inode: int,
) -> _NamespaceMembershipScan:
    """Corroborate pidfd teardown across every supervisor-visible `/proc` entry."""
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return _NamespaceMembershipScan(False, 0, ())
    members: list[dict[str, Any]] = []
    inaccessible_entries = 0
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            if (entry / "ns/pid").stat().st_ino != namespace_inode:
                continue
        except FileNotFoundError:
            # A process may exit between enumerating /proc and inspecting it.
            continue
        except PermissionError:
            # Yama and sibling user namespaces can conceal otherwise visible
            # entries, including entries owned by the same host uid. Never call
            # that a complete scan: namespace-init pidfd exit is the authority,
            # while this scan only corroborates supervisor-visible membership.
            inaccessible_entries += 1
            continue
        except OSError:
            return _NamespaceMembershipScan(
                False, inaccessible_entries, tuple(members)
            )
        try:
            stat_line = (entry / "stat").read_text(encoding="utf-8")
            fields = stat_line[stat_line.rfind(")") + 2 :].split()
            if fields[0] != "Z":
                members.append(
                    {
                        "pid": int(entry.name),
                        "state": fields[0],
                        "starttime": int(fields[19]),
                    }
                )
        except FileNotFoundError:
            continue
        except (OSError, ValueError, IndexError):
            return _NamespaceMembershipScan(
                False, inaccessible_entries, tuple(members)
            )
    return _NamespaceMembershipScan(
        True,
        inaccessible_entries,
        tuple(sorted(members, key=lambda item: item["pid"])),
    )


def establish_attempt_boundary(
    command: list[str],
    process: subprocess.Popen[Any],
    *,
    timeout_seconds: float = 5.0,
) -> AttemptBoundary | None:
    """Bind a formal bwrap PID namespace before allowing its command to exec."""
    if not isinstance(command, WrappedAttemptCommand):
        return None
    status_fd = command.boundary_status_fd
    gate_fd = command.boundary_gate_fd
    if status_fd is None or gate_fd is None:
        raise CampaignIsolationError("formal attempt boundary descriptors are missing")
    boundary: AttemptBoundary | None = None
    try:
        ready, _, _ = select.select([status_fd], [], [], timeout_seconds)
        if not ready:
            raise CampaignIsolationError("formal attempt boundary status timed out")
        payload = os.read(status_fd, 4097)
        if len(payload) > 4096 or not payload.endswith(b"\n"):
            raise CampaignIsolationError("formal attempt boundary status is malformed")
        try:
            status = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CampaignIsolationError(
                "formal attempt boundary status is not canonical JSON"
            ) from error
        expected = {
            "child-pid",
            "ipc-namespace",
            "mnt-namespace",
            "pid-namespace",
        }
        if (
            not isinstance(status, dict)
            or set(status) != expected
            or any(type(status.get(key)) is not int or status[key] <= 0 for key in expected)
        ):
            raise CampaignIsolationError("formal attempt boundary status is invalid")
        init_pid = status["child-pid"]
        try:
            init_pidfd = os.pidfd_open(init_pid, 0)
        except OSError as error:
            raise CampaignIsolationError("cannot open attempt namespace init pidfd") from error
        parent_pid, namespace_pid = _process_namespace_identity(init_pid)
        boundary = AttemptBoundary(
            init_pidfd=init_pidfd,
            status_fd=status_fd,
            status=status,
            init_starttime=_process_starttime(init_pid),
            init_parent_pid=parent_pid,
            procfs=command.boundary_procfs,
        )
        command.boundary_status_fd = None
        try:
            namespace_inode = os.stat(f"/proc/{init_pid}/ns/pid").st_ino
            mount_inode = os.stat(f"/proc/{init_pid}/ns/mnt").st_ino
            ipc_inode = os.stat(f"/proc/{init_pid}/ns/ipc").st_ino
        except OSError as error:
            raise CampaignIsolationError(
                "cannot bind attempt namespace identity"
            ) from error
        if (
            namespace_inode != status["pid-namespace"]
            or mount_inode != status["mnt-namespace"]
            or ipc_inode != status["ipc-namespace"]
            or parent_pid != process.pid
            or namespace_pid != 1
            or _pidfd_ready(init_pidfd, 0)
            or _process_starttime(init_pid) != boundary.init_starttime
        ):
            raise CampaignIsolationError("attempt namespace init identity changed")
        os.write(gate_fd, b"1")
        return boundary
    except Exception:
        if boundary is not None:
            boundary.close()
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise
    finally:
        command.close_boundary_status_fd()
        command.close_boundary_gate_fd()


def finalize_attempt_boundary(
    process: subprocess.Popen[Any],
    boundary: AttemptBoundary,
    *,
    reason: str,
    terminate: bool,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """End or verify a PID namespace through its exact, non-reusable pidfd."""
    evidence = {
        "required": terminate,
        "reason": reason,
        "scope": "private_pid_namespace",
        "method": "namespace_init_pidfd_v1",
        "boundary": boundary.receipt(),
        "sigkill_sent": False,
        "outer_supervisor_force_killed": False,
        "verification_performed": True,
        "namespace_init_exit_verified": False,
        "verified_absent": False,
        "kernel_semantics": (
            "linux_pid_namespace_init_exit_sigkill_all_members"
        ),
        "bubblewrap_terminal_status_verified": False,
        "bubblewrap_status_eof_verified": False,
    }
    try:
        if terminate:
            boundary.terminate_now()
        evidence["sigkill_sent"] = boundary.sigkill_sent
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            evidence["outer_supervisor_force_killed"] = True
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                pass
        evidence["namespace_init_exit_verified"] = _pidfd_ready(
            boundary.init_pidfd, timeout_seconds
        )
        evidence["outer_supervisor_exit_code"] = process.poll()
        terminal_payload = os.read(boundary.status_fd, 4097)
        terminal_absent = terminal_payload == b""
        if len(terminal_payload) <= 4096 and terminal_payload.endswith(b"\n"):
            lines = terminal_payload.splitlines()
            try:
                terminal = json.loads(lines[0]) if len(lines) == 1 else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                terminal = None
            expected_exit = process.poll()
            if (
                isinstance(terminal, dict)
                and set(terminal) == {"exit-code"}
                and type(terminal.get("exit-code")) is int
                and terminal["exit-code"] == expected_exit
            ):
                evidence["bubblewrap_terminal_status_verified"] = True
                evidence["bubblewrap_terminal_status"] = terminal
        evidence["bubblewrap_status_eof_verified"] = (
            os.read(boundary.status_fd, 1) == b""
        )
        evidence["bubblewrap_terminal_status_absent_after_sigkill"] = (
            terminal_absent and boundary.sigkill_sent
        )
        evidence["teardown_mode"] = (
            "pidfd_sigkill" if boundary.sigkill_sent else "natural_exit"
        )
        membership_scan = _pid_namespace_membership_scan(
            boundary.status["pid-namespace"]
        )
        evidence["namespace_membership_enumeration_completed"] = (
            membership_scan.enumeration_completed
        )
        evidence["namespace_membership_scan_complete"] = membership_scan.complete
        evidence["namespace_membership_inaccessible_entries_count"] = (
            membership_scan.inaccessible_entries_count
        )
        evidence["live_visible_namespace_members_after"] = list(
            membership_scan.live_visible_members
        )
        terminal_evidence_valid = (
            evidence["bubblewrap_terminal_status_verified"] is True
            or evidence["bubblewrap_terminal_status_absent_after_sigkill"] is True
        )
        evidence["verified_absent"] = (
            evidence["namespace_init_exit_verified"] is True
            and process.poll() is not None
            and evidence["outer_supervisor_force_killed"] is False
            and terminal_evidence_valid
            and evidence["bubblewrap_status_eof_verified"] is True
            and membership_scan.enumeration_completed
            and not membership_scan.live_visible_members
        )
        return evidence
    except (OSError, ProcessLookupError) as error:
        evidence["error"] = f"{type(error).__name__}: {error}"
        evidence["outer_supervisor_exit_code"] = process.poll()
        return evidence
    finally:
        boundary.close()


def attempt_cleanup_verified(
    cleanup: Any,
    *,
    exit_code: Any,
    allowed_reasons: set[str],
    required_procfs: str,
) -> bool:
    """Fail-closed verifier for a serialized formal attempt boundary."""
    if not isinstance(cleanup, dict):
        return False
    boundary = cleanup.get("boundary")
    inaccessible_entries = cleanup.get(
        "namespace_membership_inaccessible_entries_count"
    )
    scan_complete = cleanup.get("namespace_membership_scan_complete")
    reason = cleanup.get("reason")
    common = (
        reason in allowed_reasons
        and cleanup.get("required") is (reason != "normal_exit")
        and cleanup.get("scope") == "private_pid_namespace"
        and cleanup.get("method") == "namespace_init_pidfd_v1"
        and cleanup.get("verification_performed") is True
        and cleanup.get("namespace_init_exit_verified") is True
        and cleanup.get("namespace_membership_enumeration_completed") is True
        and type(scan_complete) is bool
        and type(inaccessible_entries) is int
        and inaccessible_entries >= 0
        and scan_complete == (inaccessible_entries == 0)
        and cleanup.get("live_visible_namespace_members_after") == []
        and cleanup.get("verified_absent") is True
        and cleanup.get("outer_supervisor_force_killed") is False
        and cleanup.get("bubblewrap_status_eof_verified") is True
        and cleanup.get("outer_supervisor_exit_code") == exit_code
        and cleanup.get("kernel_semantics")
        == "linux_pid_namespace_init_exit_sigkill_all_members"
        and isinstance(boundary, dict)
        and boundary.get("schema") == "aka.attempt-process-boundary/v1"
        and boundary.get("policy") == ATTEMPT_CONTAINMENT_POLICY
        and boundary.get("pid_namespace_unshared") is True
        and boundary.get("procfs") == required_procfs
        and boundary.get("pidfd_opened") is True
        and boundary.get("identity_source")
        == "pinned_bubblewrap_json_status_fd"
        and all(
            type(boundary.get(field)) is int and boundary[field] > 0
            for field in (
                "namespace_init_pid",
                "namespace_init_starttime",
                "namespace_init_parent_pid",
                "pid_namespace_id",
                "mount_namespace_id",
                "ipc_namespace_id",
            )
        )
        and boundary.get("namespace_init_inner_pid") == 1
    )
    if not common:
        return False
    if cleanup.get("teardown_mode") == "pidfd_sigkill":
        terminal_status_valid = (
            cleanup.get("bubblewrap_terminal_status_verified") is True
            and cleanup.get("bubblewrap_terminal_status")
            == {"exit-code": exit_code}
            and cleanup.get("bubblewrap_terminal_status_absent_after_sigkill")
            is False
        )
        terminal_status_absent = (
            cleanup.get("bubblewrap_terminal_status_verified") is False
            and cleanup.get("bubblewrap_terminal_status") is None
            and cleanup.get("bubblewrap_terminal_status_absent_after_sigkill")
            is True
        )
        return (
            cleanup.get("sigkill_sent") is True
            and exit_code == 128 + int(signal.SIGKILL)
            and reason in {"exact_turn_boundary", "timeout"}
            and (terminal_status_valid or terminal_status_absent)
        )
    return (
        cleanup.get("teardown_mode") == "natural_exit"
        and cleanup.get("sigkill_sent") is False
        and type(exit_code) is int
        and cleanup.get("bubblewrap_terminal_status_verified") is True
        and cleanup.get("bubblewrap_terminal_status") == {"exit-code": exit_code}
        and cleanup.get("bubblewrap_terminal_status_absent_after_sigkill") is False
    )


def _validated_attempt_roots(
    raw_roots: Iterable[Path], *, data_root: Path, label: str
) -> list[Path]:
    roots: list[Path] = []
    for raw in raw_roots:
        try:
            root = Path(raw).resolve(strict=True)
        except OSError as error:
            raise CampaignIsolationError(
                f"attempt {label} root is unavailable: {raw}"
            ) from error
        try:
            relative = root.relative_to(data_root)
        except ValueError as error:
            raise CampaignIsolationError(
                f"attempt {label} root is outside campaign data root: {root}"
            ) from error
        if not relative.parts or not root.is_dir():
            raise CampaignIsolationError(
                f"attempt {label} root must be a specific directory: {root}"
            )
        if root not in roots:
            roots.append(root)
    return roots


def _validated_trusted_read_only_roots(
    raw_roots: Iterable[Path], *, data_root: Path
) -> list[Path]:
    """Resolve exact external roots that the caller already authenticated.

    Formal attempts hide both ``/tmp`` and the complete campaign data root.
    Apex may itself live below ``/tmp``, so its already provenance-bound checkout
    needs one explicit read-only rebind.  This path class is deliberately separate
    from attempt-owned roots: it must be canonical, external to campaign data, and
    narrower than a top-level filesystem directory.
    """

    roots: list[Path] = []
    for raw in raw_roots:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            raise CampaignIsolationError(
                f"trusted read-only root must be absolute: {raw}"
            )
        lexical = Path(os.path.abspath(candidate))
        try:
            metadata = lexical.lstat()
            root = lexical.resolve(strict=True)
        except OSError as error:
            raise CampaignIsolationError(
                f"trusted read-only root is unavailable: {raw}"
            ) from error
        if (
            root != lexical
            or stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or len(root.parts) < 3
            or root in _BROAD_EXTERNAL_ROOTS
        ):
            raise CampaignIsolationError(
                f"trusted read-only root must be a specific canonical directory: {root}"
            )
        try:
            root.relative_to(data_root)
        except ValueError:
            pass
        else:
            raise CampaignIsolationError(
                "trusted read-only root must be outside campaign data root: "
                f"{root}"
            )
        if root not in roots:
            roots.append(root)
    return roots


def _build_attempt_mount_receipt(
    *,
    data_root: Path,
    writable: list[Path],
    read_only: list[Path],
    trusted_read_only: list[Path],
) -> dict[str, Any]:
    material: dict[str, Any] = {
        "schema": ATTEMPT_MOUNT_RECEIPT_SCHEMA,
        "campaign_data_root": str(data_root),
        "campaign_data_root_hidden": True,
        "writable_roots": [str(path) for path in writable],
        "read_only_roots": [str(path) for path in read_only],
        "trusted_external_read_only_roots": [
            str(path) for path in trusted_read_only
        ],
    }
    return {**material, "sha256": canonical_digest(material)}


def attempt_mount_receipt(command: list[str]) -> dict[str, Any] | None:
    """Return a detached copy of the mount receipt carried by a wrapped argv."""

    if not isinstance(command, WrappedAttemptCommand):
        return None
    receipt = command.mount_receipt
    if not isinstance(receipt, dict):
        return None
    return json.loads(json.dumps(receipt, sort_keys=True))


def _reject_overlapping_roots(roots: tuple[Path, ...]) -> None:
    for index, root in enumerate(roots):
        for other in roots[index + 1 :]:
            try:
                root.relative_to(other)
                overlaps = True
            except ValueError:
                try:
                    other.relative_to(root)
                    overlaps = True
                except ValueError:
                    overlaps = False
            if overlaps:
                raise CampaignIsolationError(
                    f"attempt mount roots overlap: {root} and {other}"
                )


def _make_owner_writable(root: Path) -> None:
    for path in (root, *root.rglob("*")):
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise CampaignIsolationError(f"cannot inspect attempt home: {path}") from error
        if stat.S_ISLNK(mode):
            raise CampaignIsolationError(f"attempt home contains a symlink: {path}")
        if stat.S_ISDIR(mode):
            path.chmod((mode & 0o777) | 0o700)
        elif stat.S_ISREG(mode):
            path.chmod((mode & 0o777) | 0o600)
        else:
            raise CampaignIsolationError(f"attempt home contains an unsafe file: {path}")


__all__ = [
    "APEX_RUNTIME_MOUNT_POLICY",
    "APEX_RUNTIME_MOUNT_SCHEMA",
    "ATTEMPT_CONTAINMENT_POLICY",
    "ATTEMPT_MOUNT_RECEIPT_SCHEMA",
    "CampaignIsolationError",
    "AttemptBoundary",
    "WrappedAttemptCommand",
    "attempt_cleanup_verified",
    "attempt_command_pass_fds",
    "attempt_mount_receipt",
    "establish_attempt_boundary",
    "finalize_attempt_boundary",
    "is_formal_campaign",
    "formal_gpu_evidence",
    "isolated_environment",
    "prepare_attempt_home",
    "release_attempt_command_fds",
    "runtime_isolation_receipt",
    "wrap_attempt_command",
]
