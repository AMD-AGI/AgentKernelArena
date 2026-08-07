#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Host-side cooperative GPU leases and KFD ownership preflight evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    from src.gpu_device_boundary import GpuBoundaryError, load_plan
    from src.kfd_process_inventory import (
        KfdProcessInventoryError,
        parse_inventory,
    )
except ModuleNotFoundError:  # direct ``python src/gpu_exclusivity.py`` execution
    from gpu_device_boundary import GpuBoundaryError, load_plan
    from kfd_process_inventory import KfdProcessInventoryError, parse_inventory


SCHEMA = "aka.gpu-exclusivity-receipt/v1"
POLICY = "physical_unique_id_flock_plus_kfd_preflight_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class GpuExclusivityError(RuntimeError):
    """Raised when formal GPU exclusivity cannot be established."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def lease_keys(plan: dict[str, Any]) -> tuple[str, ...]:
    keys = tuple(str(device.get("unique_id", "")) for device in plan.get("devices", []))
    if not keys or len(set(keys)) != len(keys):
        raise GpuExclusivityError("GPU plan has missing or duplicate physical unique IDs")
    if any(not re.fullmatch(r"0x[0-9a-f]+", key) for key in keys):
        raise GpuExclusivityError("GPU unique IDs are not canonical lowercase hexadecimal")
    return keys


def protected_device_paths(plan: dict[str, Any]) -> tuple[str, ...]:
    raw = [str(plan.get("kfd_device", {}).get("path", ""))]
    for device in plan.get("devices", []):
        raw.extend(str(render.get("path", "")) for render in device.get("render_nodes", []))
    paths = tuple(sorted(set(raw)))
    if not paths or any(not path.startswith("/dev/") for path in paths):
        raise GpuExclusivityError("GPU plan contains an unsafe protected device path")
    return paths


def parse_kfd_process_inventory(raw: str | bytes) -> dict[str, Any]:
    """Validate an authoritative direct-API inventory, fail closed."""
    try:
        return parse_inventory(raw)
    except KfdProcessInventoryError as error:
        raise GpuExclusivityError(str(error)) from error


def _proc_device_audit(
    device_paths: tuple[str, ...], *, proc_root: Path
) -> dict[str, Any]:
    protected = set(device_paths)
    owners: list[dict[str, Any]] = []
    inaccessible: list[int] = []
    try:
        entries = sorted(proc_root.iterdir(), key=lambda path: path.name)
    except OSError as error:
        raise GpuExclusivityError(f"cannot scan process table: {error}") from error
    for entry in entries:
        if not entry.name.isdigit():
            continue
        fd_root = entry / "fd"
        try:
            descriptors = list(fd_root.iterdir())
        except OSError:
            inaccessible.append(int(entry.name))
            continue
        matched: set[str] = set()
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            target = target.removesuffix(" (deleted)")
            if target in protected:
                matched.add(target)
        if not matched:
            continue
        try:
            command = (entry / "cmdline").read_bytes()[:512].replace(b"\x00", b" ")
            command_text = command.decode("utf-8", errors="replace").strip()
        except OSError:
            command_text = ""
        owners.append(
            {"pid": int(entry.name), "device_paths": sorted(matched), "command": command_text}
        )
    inaccessible.sort()
    return {
        "owners": owners,
        "complete": not inaccessible,
        "inaccessible_pid_count": len(inaccessible),
        "inaccessible_pids_sha256": _digest(inaccessible),
        "inaccessible_pids_sample": inaccessible[:64],
    }


def scan_device_owners(
    device_paths: tuple[str, ...], *, proc_root: Path = Path("/proc")
) -> list[dict[str, Any]]:
    """Return open-device owners only when every live PID was auditable."""
    audit = _proc_device_audit(device_paths, proc_root=proc_root)
    if not audit["complete"]:
        raise GpuExclusivityError(
            "supplementary /proc GPU-owner audit is incomplete for "
            f"{audit['inaccessible_pid_count']} live PIDs"
        )
    return audit["owners"]


def build_receipt(
    *,
    plan: dict[str, Any],
    run_name: str,
    runner_pid: int,
    locks: dict[str, str],
    kfd_process_inventory_path: Path,
    proc_root: Path = Path("/proc"),
    observed_at_ns: int | None = None,
) -> dict[str, Any]:
    keys = lease_keys(plan)
    if set(locks) != set(keys):
        raise GpuExclusivityError("lease lock set differs from physical GPU unique IDs")
    if any(not Path(path).is_absolute() for path in locks.values()):
        raise GpuExclusivityError("lease lock paths must be absolute")
    try:
        inventory_metadata = kfd_process_inventory_path.lstat()
        inventory_raw = kfd_process_inventory_path.read_bytes()
    except OSError as error:
        raise GpuExclusivityError(f"cannot read KFD process inventory: {error}") from error
    if (
        not kfd_process_inventory_path.is_absolute()
        or not kfd_process_inventory_path.is_file()
        or kfd_process_inventory_path.is_symlink()
        or inventory_metadata.st_nlink != 1
        or inventory_metadata.st_mode & 0o222
    ):
        raise GpuExclusivityError("KFD process inventory artifact is unsafe or mutable")
    kfd_inventory = parse_kfd_process_inventory(inventory_raw)
    if not kfd_inventory["verified_empty"]:
        raise GpuExclusivityError(
            "authoritative KFD inventory reports active GPU PIDs: "
            + ", ".join(str(pid) for pid in kfd_inventory["pids"])
        )
    proc_audit = _proc_device_audit(protected_device_paths(plan), proc_root=proc_root)
    owners = proc_audit["owners"]
    if owners:
        details = ", ".join(str(owner["pid"]) for owner in owners)
        raise GpuExclusivityError(f"selected GPU devices are already open by PIDs: {details}")
    material = {
        "schema": SCHEMA,
        "policy": POLICY,
        "gpu_boundary_plan_sha256": plan["sha256"],
        "run_name": run_name,
        "runner_pid": runner_pid,
        "observed_at_ns": observed_at_ns if observed_at_ns is not None else time.time_ns(),
        "leases": [
            {"unique_id": key, "lock_path": locks[key]} for key in sorted(keys)
        ],
        "protected_device_paths": list(protected_device_paths(plan)),
        "foreign_device_owners": [],
        "authoritative_kfd_process_inventory": {
            **kfd_inventory,
            "path": str(kfd_process_inventory_path),
        },
        "supplementary_proc_audit": proc_audit,
        "proof_basis": "rocm_smi_kfd_process_api_v1",
        "exclusivity_verified": True,
    }
    return {**material, "sha256": _digest(material)}


def load_receipt(
    path: Path, *, expected_plan_sha256: str | None = None
) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GpuExclusivityError(f"cannot read GPU exclusivity receipt: {error}") from error
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
        raise GpuExclusivityError("GPU exclusivity receipt schema mismatch")
    observed_digest = receipt.get("sha256")
    material = {key: value for key, value in receipt.items() if key != "sha256"}
    if not isinstance(observed_digest, str) or observed_digest != _digest(material):
        raise GpuExclusivityError("GPU exclusivity receipt digest mismatch")
    if expected_plan_sha256 is not None and receipt.get(
        "gpu_boundary_plan_sha256"
    ) != expected_plan_sha256:
        raise GpuExclusivityError("GPU exclusivity receipt is bound to another plan")
    if (
        receipt.get("policy") != POLICY
        or receipt.get("exclusivity_verified") is not True
        or receipt.get("foreign_device_owners") != []
        or not isinstance(receipt.get("leases"), list)
        or not receipt["leases"]
    ):
        raise GpuExclusivityError("GPU exclusivity receipt is not verified")
    inventory = receipt.get("authoritative_kfd_process_inventory")
    if not isinstance(inventory, dict) or inventory.get("verified_empty") is not True:
        raise GpuExclusivityError("authoritative KFD process evidence is missing")
    inventory_path = Path(str(inventory.get("path", "")))
    try:
        metadata = inventory_path.lstat()
        raw = inventory_path.read_bytes()
    except OSError as error:
        raise GpuExclusivityError(f"cannot re-read KFD process inventory: {error}") from error
    if (
        not inventory_path.is_absolute()
        or not inventory_path.is_file()
        or inventory_path.is_symlink()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o222
    ):
        raise GpuExclusivityError("KFD process inventory artifact became unsafe")
    parsed = parse_kfd_process_inventory(raw)
    if parsed != {key: inventory.get(key) for key in parsed}:
        raise GpuExclusivityError("KFD process inventory no longer matches its receipt")
    return receipt


def _load_plan(path: Path) -> dict[str, Any]:
    try:
        return load_plan(path)
    except GpuBoundaryError as error:
        raise GpuExclusivityError(str(error)) from error


def _parse_locks(values: list[str]) -> dict[str, str]:
    locks: dict[str, str] = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or not key or not path or key in locks:
            raise GpuExclusivityError("--lock must be a unique UNIQUE_ID=/absolute/path")
        locks[key] = path
    return locks


def _main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    keys_parser = subparsers.add_parser("lease-keys")
    keys_parser.add_argument("--plan", type=Path, required=True)
    receipt_parser = subparsers.add_parser("create-receipt")
    receipt_parser.add_argument("--plan", type=Path, required=True)
    receipt_parser.add_argument("--run-name", required=True)
    receipt_parser.add_argument("--runner-pid", type=int, required=True)
    receipt_parser.add_argument("--lock", action="append", default=[])
    receipt_parser.add_argument("--kfd-process-inventory", type=Path, required=True)
    receipt_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify-receipt")
    verify_parser.add_argument("--receipt", type=Path, required=True)
    verify_parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    try:
        if args.command == "lease-keys":
            print("\n".join(lease_keys(_load_plan(args.plan))))
        elif args.command == "create-receipt":
            receipt = build_receipt(
                plan=_load_plan(args.plan),
                run_name=args.run_name,
                runner_pid=args.runner_pid,
                locks=_parse_locks(args.lock),
                kfd_process_inventory_path=args.kfd_process_inventory,
            )
            args.output.write_bytes(_canonical_bytes(receipt) + b"\n")
        else:
            receipt = load_receipt(
                args.receipt, expected_plan_sha256=args.plan_sha256
            )
            print(receipt["sha256"])
    except (GpuExclusivityError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
