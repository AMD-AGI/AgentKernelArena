# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Per-attempt mount and home isolation for formal matched campaigns."""

from __future__ import annotations

import os
import json
import shutil
import stat
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
    binary = shutil.which("bwrap")
    if not binary:
        raise CampaignIsolationError("formal campaign requires bubblewrap (bwrap)")
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

    wrapped = [
        binary,
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
    "wrap_attempt_command",
]
