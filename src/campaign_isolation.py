# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Per-attempt mount and home isolation for formal matched campaigns."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
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


_RUNTIME_ISOLATION_SCHEMA = "aka.runtime-isolation-receipt/v1"
_ZERO_CAPABILITY_FIELDS = ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")


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

def read_error(path):
    try:
        with open(path, "rb") as stream:
            stream.read(1)
        return None
    except OSError as error:
        return error.errno

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

parent_root = f"/proc/{outer_pid}/root{sentinel}"
parent_fd = f"/proc/{outer_pid}/fd/{outer_fd}"
proc_options = mount_options("/proc")
direct_error = read_error(sentinel)
parent_root_error = read_error(parent_root)
parent_fd_error = read_error(parent_fd)
result = {
    "campaign_data_hidden": direct_error == errno.ENOENT,
    "parent_process_visible_in_inherited_proc": read_error(f"/proc/{outer_pid}/status") is None,
    "parent_root_escape_blocked": parent_root_error in {errno.EACCES, errno.EPERM},
    "parent_fd_escape_blocked": parent_fd_error in {errno.EACCES, errno.EPERM},
    "proc_mount_read_only": bool(proc_options) and all("ro" in item for item in proc_options),
    "pid_namespace_unshared": os.readlink("/proc/self/ns/pid") != outer_pid_namespace,
    "ipc_namespace_unshared": os.readlink("/proc/self/ns/ipc") != outer_ipc_namespace,
    "private_shm": any(
        len(fields) > 6 and fields[4] == "/dev/shm" and "tmpfs" in fields
        for fields in (
            line.split()
            for line in pathlib.Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
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


def _bubblewrap_base_command(binary: str | Path, data_root: Path) -> list[str]:
    """Pinned mount/namespace boundary shared by probes and real attempts."""
    return [
        str(binary),
        "--die-with-parent",
        "--unshare-pid",
        "--unshare-ipc",
        "--ro-bind",
        "/",
        "/",
        "--dev-bind",
        "/dev",
        "/dev",
        "--tmpfs",
        "/dev/shm",
        "--ro-bind",
        "/proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        str(data_root),
    ]


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
            "parent_process_visible_in_inherited_proc",
            "parent_root_escape_blocked",
            "parent_fd_escape_blocked",
            "proc_mount_read_only",
            "pid_namespace_unshared",
            "ipc_namespace_unshared",
            "private_shm",
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
    attempt = _attempt_escape_probe(binary=binary, data_root=data_root, outer=outer)
    # Namespace inode identifiers are deliberately used only by the live probe:
    # they differ for every Docker worker and would make the Apex/Codex
    # comparison contract run-specific.  The manifest records the verified
    # relationship instead (the two `*_namespace_unshared` booleans).
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
            "docker_pid_namespace": "private_default",
            "attempt_mount_namespace": "bubblewrap",
            "attempt_pid_namespace": "unshared",
            "attempt_ipc_namespace": "unshared",
            "attempt_proc": "read_only_bind_of_docker_private_procfs",
            "proc_escape_guard": "yama_ptrace_scope_and_live_parent_root_fd_probe_v1",
        },
        "outer_runtime": recorded_outer,
        "bubblewrap": bubblewrap,
        "attempt_probe": attempt,
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
    return updated


def wrap_attempt_command(
    command: list[str],
    *,
    eval_config: dict[str, Any],
    writable_roots: Iterable[Path],
    read_only_roots: Iterable[Path] = (),
) -> list[str]:
    """Hide other campaign data and expose only explicitly scoped roots."""
    if not is_formal_campaign(eval_config):
        return command
    # Re-resolve, hash, and version-check the read-only mounted executable at
    # attempt construction time instead of trusting a prior PATH lookup.
    binary, _identity = _bubblewrap_identity()
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
    if not writable:
        raise CampaignIsolationError("formal campaign has no attempt writable root")
    _reject_overlapping_roots((*writable, *read_only))

    wrapped = _bubblewrap_base_command(binary, data_root)
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
    # bubblewrap opens bind sources in its parent namespace before applying the
    # destination mounts, so these still refer to host roots after data_root is
    # hidden by the tmpfs above.
    for root in read_only:
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
    wrapped.extend(["--", *command])
    return wrapped


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
    "CampaignIsolationError",
    "is_formal_campaign",
    "formal_gpu_evidence",
    "isolated_environment",
    "prepare_attempt_home",
    "runtime_isolation_receipt",
    "wrap_attempt_command",
]
