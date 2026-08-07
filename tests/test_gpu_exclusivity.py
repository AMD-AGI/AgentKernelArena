"""CPU-only tests for formal physical-GPU lease evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.gpu_exclusivity import (
    GpuExclusivityError,
    build_receipt,
    load_receipt,
    parse_kfd_process_inventory,
    protected_device_paths,
    scan_device_owners,
)
from src.kfd_process_inventory import query_process_inventory


class _FakeInventoryApi:
    def __init__(self, pids: list[int]) -> None:
        self._pids = pids

    def init(self) -> int:
        return 0

    def process_count(self) -> tuple[int, int]:
        return 0, len(self._pids)

    def process_pids(self, capacity: int) -> tuple[int, list[int]]:
        assert capacity >= len(self._pids)
        return 0, list(self._pids)

    def shutdown(self) -> int:
        return 0


def _plan() -> dict:
    return {
        "sha256": "a" * 64,
        "kfd_device": {"path": "/dev/kfd", "major": 235, "minor": 0},
        "devices": [
            {
                "host_gpu_id": "0",
                "unique_id": "0x0000000000000001",
                "render_nodes": [
                    {"path": "/dev/dri/renderD128"},
                    {"path": "/dev/dri/renderD136"},
                ],
            }
        ],
    }


def _kfd_inventory(tmp_path: Path, pids: list[int], name: str) -> Path:
    path = tmp_path / name
    document = query_process_inventory(
        _FakeInventoryApi(pids),
        library_path=Path("/opt/rocm/lib/librocm_smi64.so.7"),
        library_sha256="b" * 64,
        observed_at_ns=123,
    )
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path


def _empty_kfd_inventory(tmp_path: Path) -> Path:
    return _kfd_inventory(tmp_path, [], "kfd-process-inventory.json")


def test_receipt_binds_physical_unique_id_and_empty_owner_preflight(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    lock = tmp_path / "0x0000000000000001.lock"
    receipt = build_receipt(
        plan=_plan(),
        run_name="formal-run",
        runner_pid=123,
        locks={"0x0000000000000001": str(lock)},
        kfd_process_inventory_path=_empty_kfd_inventory(tmp_path),
        proc_root=proc,
        observed_at_ns=456,
    )
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")

    loaded = load_receipt(path, expected_plan_sha256="a" * 64)

    assert loaded["exclusivity_verified"] is True
    assert loaded["leases"] == [
        {"unique_id": "0x0000000000000001", "lock_path": str(lock)}
    ]
    assert protected_device_paths(_plan()) == (
        "/dev/dri/renderD128",
        "/dev/dri/renderD136",
        "/dev/kfd",
    )


def test_open_kfd_or_render_descriptor_fails_preflight(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    fd_root = proc / "4242/fd"
    fd_root.mkdir(parents=True)
    (proc / "4242/cmdline").write_bytes(b"foreign-worker\x00--serve\x00")
    (fd_root / "7").symlink_to("/dev/dri/renderD136")

    with pytest.raises(GpuExclusivityError, match="4242"):
        build_receipt(
            plan=_plan(),
            run_name="formal-run",
            runner_pid=123,
            locks={"0x0000000000000001": str(tmp_path / "lease.lock")},
            kfd_process_inventory_path=_empty_kfd_inventory(tmp_path),
            proc_root=proc,
        )


def test_receipt_digest_or_plan_tampering_is_rejected(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    receipt = build_receipt(
        plan=_plan(),
        run_name="formal-run",
        runner_pid=123,
        locks={"0x0000000000000001": str(tmp_path / "lease.lock")},
        kfd_process_inventory_path=_empty_kfd_inventory(tmp_path),
        proc_root=proc,
    )
    receipt["exclusivity_verified"] = False
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(GpuExclusivityError, match="digest"):
        load_receipt(path, expected_plan_sha256="a" * 64)


def test_unreadable_live_proc_fd_directory_is_not_silently_skipped(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    process = proc / "4242"
    process.mkdir(parents=True)
    (process / "fd").write_text("not a directory", encoding="utf-8")

    with pytest.raises(GpuExclusivityError, match="incomplete.*1 live PIDs"):
        scan_device_owners(("/dev/kfd",), proc_root=proc)


def test_authoritative_kfd_inventory_catches_owner_hidden_from_proc(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    process = proc / "4242"
    process.mkdir(parents=True)
    (process / "fd").write_text("unreadable", encoding="utf-8")
    inventory = _kfd_inventory(tmp_path, [4242], "active-kfd-processes.json")

    parsed = parse_kfd_process_inventory(inventory.read_bytes())
    assert parsed["pids"] == [4242]
    with pytest.raises(GpuExclusivityError, match="active GPU PIDs: 4242"):
        build_receipt(
            plan=_plan(),
            run_name="formal-run",
            runner_pid=123,
            locks={"0x0000000000000001": str(tmp_path / "lease.lock")},
            kfd_process_inventory_path=inventory,
            proc_root=proc,
        )
