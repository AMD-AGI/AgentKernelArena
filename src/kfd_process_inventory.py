#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Fail-closed KFD process inventory from the ROCm SMI library API."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Protocol


SCHEMA = "aka.kfd-process-inventory/v1"
SOURCE = "librocm_smi64.rsmi_compute_process_info_get"
_FETCH_HEADROOM = 64
_MAX_INVENTORY_BYTES = 4 * 1024 * 1024
_MAX_PROCESS_RECORDS = 262_144
_SHA256 = re.compile(r"[0-9a-f]{64}")


class KfdProcessInventoryError(RuntimeError):
    """Raised when KFD process state cannot be proved from ROCm SMI."""


class _RsmiProcessInfo(ctypes.Structure):
    _fields_ = [
        ("process_id", ctypes.c_uint32),
        ("pasid", ctypes.c_uint32),
        ("vram_usage", ctypes.c_uint64),
        ("sdma_usage", ctypes.c_uint64),
        ("cu_occupancy", ctypes.c_uint32),
    ]


class ProcessInventoryApi(Protocol):
    """Narrow API used by the inventory query and CPU-only fakes."""

    def init(self) -> int: ...

    def process_count(self) -> tuple[int, int]: ...

    def process_pids(self, capacity: int) -> tuple[int, list[int]]: ...

    def shutdown(self) -> int: ...


class RocmSmiApi:
    """Checked ctypes wrapper around the process-inventory API."""

    def __init__(self, library_path: Path) -> None:
        try:
            library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_LOCAL)
        except OSError as error:
            raise KfdProcessInventoryError(
                f"cannot load ROCm SMI library {library_path}: {error}"
            ) from error
        self._library = library
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        try:
            self._library.rsmi_init.argtypes = [ctypes.c_uint64]
            self._library.rsmi_init.restype = ctypes.c_int
            self._library.rsmi_shut_down.argtypes = []
            self._library.rsmi_shut_down.restype = ctypes.c_int
            process_info = self._library.rsmi_compute_process_info_get
            process_info.argtypes = [
                ctypes.POINTER(_RsmiProcessInfo),
                ctypes.POINTER(ctypes.c_uint32),
            ]
            process_info.restype = ctypes.c_int
        except AttributeError as error:
            raise KfdProcessInventoryError(
                "ROCm SMI library lacks the required process API"
            ) from error

    def init(self) -> int:
        return int(self._library.rsmi_init(0))

    def process_count(self) -> tuple[int, int]:
        count = ctypes.c_uint32(0)
        status = self._library.rsmi_compute_process_info_get(
            None, ctypes.byref(count)
        )
        return int(status), int(count.value)

    def process_pids(self, capacity: int) -> tuple[int, list[int]]:
        records = (_RsmiProcessInfo * capacity)()
        count = ctypes.c_uint32(capacity)
        status = int(
            self._library.rsmi_compute_process_info_get(
                records, ctypes.byref(count)
            )
        )
        if count.value > capacity:
            raise KfdProcessInventoryError(
                "ROCm SMI returned more process records than the supplied capacity"
            )
        return status, [int(records[index].process_id) for index in range(count.value)]

    def shutdown(self) -> int:
        return int(self._library.rsmi_shut_down())


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise KfdProcessInventoryError(f"cannot hash ROCm SMI library: {error}") from error
    return digest.hexdigest()


def resolve_library(explicit: Path | None = None) -> Path:
    """Resolve one concrete ROCm SMI shared library without importing ROCm Python."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    elif os.environ.get("ROCM_SMI_LIB_PATH"):
        candidates.append(Path(os.environ["ROCM_SMI_LIB_PATH"]))
    else:
        candidates.extend(
            [
                Path("/opt/rocm/lib/librocm_smi64.so.7"),
                Path("/opt/rocm/lib/librocm_smi64.so"),
            ]
        )
        candidates.extend(
            sorted(Path("/opt").glob("rocm-*/lib/librocm_smi64.so.7"))
        )
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_absolute() and resolved.is_file():
            return resolved
    raise KfdProcessInventoryError("cannot resolve a ROCm SMI shared library")


def _raise_query_error(error: Exception, shutdown_status: int) -> None:
    suffix = ""
    if shutdown_status != 0:
        suffix = f"; rsmi_shut_down status={shutdown_status}"
    if isinstance(error, KfdProcessInventoryError):
        raise KfdProcessInventoryError(f"{error}{suffix}") from error
    raise KfdProcessInventoryError(f"ROCm SMI process query failed: {error}{suffix}") from error


def query_process_inventory(
    api: ProcessInventoryApi,
    *,
    library_path: Path,
    library_sha256: str,
    observed_at_ns: int | None = None,
) -> dict[str, Any]:
    """Query twice, checking every API status and failing on races or anomalies."""
    if not library_path.is_absolute() or not _SHA256.fullmatch(library_sha256):
        raise KfdProcessInventoryError("ROCm SMI library identity is invalid")
    init_status = api.init()
    if init_status != 0:
        raise KfdProcessInventoryError(f"rsmi_init failed with status={init_status}")

    query_error: Exception | None = None
    count_status = -1
    count_hint = -1
    fetch_status = -1
    capacity = -1
    pids: list[int] = []
    try:
        count_status, count_hint = api.process_count()
        if count_status != 0:
            raise KfdProcessInventoryError(
                "rsmi_compute_process_info_get count failed with "
                f"status={count_status}"
            )
        if count_hint < 0 or count_hint > _MAX_PROCESS_RECORDS:
            raise KfdProcessInventoryError("ROCm SMI returned an unsafe process count")
        capacity = max(count_hint + _FETCH_HEADROOM, _FETCH_HEADROOM)
        fetch_status, pids = api.process_pids(capacity)
        if fetch_status != 0:
            raise KfdProcessInventoryError(
                "rsmi_compute_process_info_get fetch failed with "
                f"status={fetch_status}"
            )
        if len(pids) > capacity or len(pids) > _MAX_PROCESS_RECORDS:
            raise KfdProcessInventoryError("ROCm SMI returned too many process records")
        if any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in pids):
            raise KfdProcessInventoryError("ROCm SMI returned an invalid process ID")
        if len(set(pids)) != len(pids):
            raise KfdProcessInventoryError("ROCm SMI returned duplicate process IDs")
    except Exception as error:  # normalized below after mandatory shutdown
        query_error = error

    try:
        shutdown_status = api.shutdown()
    except Exception as error:
        if query_error is None:
            query_error = error
        shutdown_status = -1
    if query_error is not None:
        _raise_query_error(query_error, shutdown_status)
    if shutdown_status != 0:
        raise KfdProcessInventoryError(
            f"rsmi_shut_down failed with status={shutdown_status}"
        )

    sorted_pids = sorted(pids)
    material = {
        "schema": SCHEMA,
        "source": SOURCE,
        "library": {
            "path": str(library_path),
            "sha256": library_sha256,
        },
        "observed_at_ns": observed_at_ns if observed_at_ns is not None else time.time_ns(),
        "query": {
            "init_status": init_status,
            "count_status": count_status,
            "count_hint": count_hint,
            "fetch_status": fetch_status,
            "fetch_capacity": capacity,
            "fetched_count": len(sorted_pids),
            "shutdown_status": shutdown_status,
        },
        "pids": sorted_pids,
        "process_count": len(sorted_pids),
    }
    return {**material, "sha256": _digest(material)}


def parse_inventory(raw: str | bytes) -> dict[str, Any]:
    """Validate a canonical inventory artifact and return its receipt summary."""
    encoded = raw.encode("utf-8") if isinstance(raw, str) else raw
    if not encoded or len(encoded) > _MAX_INVENTORY_BYTES:
        raise KfdProcessInventoryError("KFD process inventory is empty or oversized")
    try:
        document = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise KfdProcessInventoryError("KFD process inventory is not canonical JSON") from error
    expected_keys = {
        "schema",
        "source",
        "library",
        "observed_at_ns",
        "query",
        "pids",
        "process_count",
        "sha256",
    }
    if not isinstance(document, dict) or set(document) != expected_keys:
        raise KfdProcessInventoryError("KFD process inventory fields are invalid")
    material = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != _digest(material):
        raise KfdProcessInventoryError("KFD process inventory digest mismatch")
    _validate_inventory_material(material)
    return {
        "source": SOURCE,
        "artifact_sha256": hashlib.sha256(encoded).hexdigest(),
        "document_sha256": document["sha256"],
        "library": document["library"],
        "observed_at_ns": document["observed_at_ns"],
        "query": document["query"],
        "pids": document["pids"],
        "process_count": document["process_count"],
        "verified_empty": document["process_count"] == 0,
    }


def _validate_inventory_material(material: dict[str, Any]) -> None:
    if material.get("schema") != SCHEMA or material.get("source") != SOURCE:
        raise KfdProcessInventoryError("KFD process inventory schema or source mismatch")
    library = material.get("library")
    if not isinstance(library, dict) or set(library) != {"path", "sha256"}:
        raise KfdProcessInventoryError("KFD process inventory library identity is invalid")
    if not Path(str(library.get("path", ""))).is_absolute() or not _SHA256.fullmatch(
        str(library.get("sha256", ""))
    ):
        raise KfdProcessInventoryError("KFD process inventory library identity is unsafe")
    observed = material.get("observed_at_ns")
    if isinstance(observed, bool) or not isinstance(observed, int) or observed <= 0:
        raise KfdProcessInventoryError("KFD process inventory timestamp is invalid")

    query = material.get("query")
    expected_query_keys = {
        "init_status",
        "count_status",
        "count_hint",
        "fetch_status",
        "fetch_capacity",
        "fetched_count",
        "shutdown_status",
    }
    if not isinstance(query, dict) or set(query) != expected_query_keys:
        raise KfdProcessInventoryError("KFD process inventory query evidence is invalid")
    if any(
        isinstance(query[key], bool) or not isinstance(query[key], int)
        for key in expected_query_keys
    ):
        raise KfdProcessInventoryError("KFD process inventory query values are invalid")
    status_keys = ("init_status", "count_status", "fetch_status", "shutdown_status")
    if any(query[key] != 0 for key in status_keys):
        raise KfdProcessInventoryError("KFD process inventory contains a failed API status")
    expected_capacity = max(query["count_hint"] + _FETCH_HEADROOM, _FETCH_HEADROOM)
    if (
        query["count_hint"] < 0
        or query["count_hint"] > _MAX_PROCESS_RECORDS
        or query["fetch_capacity"] != expected_capacity
    ):
        raise KfdProcessInventoryError("KFD process inventory capacity evidence is invalid")

    pids = material.get("pids")
    if not isinstance(pids, list) or any(
        isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in pids
    ):
        raise KfdProcessInventoryError("KFD process inventory PID list is invalid")
    if pids != sorted(set(pids)):
        raise KfdProcessInventoryError("KFD process inventory PID list is not canonical")
    if (
        material.get("process_count") != len(pids)
        or query["fetched_count"] != len(pids)
        or query["fetch_capacity"] < len(pids)
        or len(pids) > _MAX_PROCESS_RECORDS
    ):
        raise KfdProcessInventoryError("KFD process inventory counts are inconsistent")


def collect_inventory(
    *, library: Path | None = None, observed_at_ns: int | None = None
) -> dict[str, Any]:
    library_path = resolve_library(library)
    return query_process_inventory(
        RocmSmiApi(library_path),
        library_path=library_path,
        library_sha256=_file_sha256(library_path),
        observed_at_ns=observed_at_ns,
    )


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        document = collect_inventory(library=args.library)
        args.output.write_bytes(_canonical_bytes(document) + b"\n")
    except (KfdProcessInventoryError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
