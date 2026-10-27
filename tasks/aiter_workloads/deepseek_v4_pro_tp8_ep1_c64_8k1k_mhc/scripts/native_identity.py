"""Bind an explicitly declared artifact set to observed on-disk bytes.

No discovery, module loading, compilation, or GPU calls occur here. Identifying
a file does not prove its dispatch, execution, loaded-memory contents, or build
provenance. Callers must include unobservable required artifacts as missing.
"""
from __future__ import annotations

import ctypes
import hashlib
from importlib.machinery import EXTENSION_SUFFIXES, ExtensionFileLoader
import os
from pathlib import Path
import re
import stat
from types import ModuleType


EVIDENCE_KINDS = {
    "python_extension": "python_module_file_attribute",
    "ctypes_library": "ctypes_handle_name_attribute",
    "device_code": "explicit_device_code_file_path",
    "table": "explicit_table_file_path",
}
SCOPE = "Declared on-disk artifacts only; no dispatch, execution, memory-image, or build-provenance attestation."


class NativeIdentityError(RuntimeError):
    """Required native artifact evidence is incomplete or differs."""


def _requirements(required):
    if not isinstance(required, dict) or not required:
        raise ValueError("Declare a nonempty artifact ID-to-kind mapping")
    if any(not isinstance(key, str) or not key or not isinstance(kind, str) or kind not in EVIDENCE_KINDS
           for key, kind in required.items()):
        raise ValueError("Every artifact needs a nonempty ID and a supported kind")
    if set(required.values()) == {"table"}:
        raise ValueError("A table-only declaration cannot establish native artifact identity")


def _path(location, roots):
    if isinstance(location, dict):
        if (set(location) != {"root", "path"} or not isinstance(location["root"], str)
                or location["root"] not in roots):
            raise NativeIdentityError("Relative artifact path needs an explicit root mapping")
        root = Path(roots[location["root"]])
        relative = Path(location["path"])
        if not root.is_absolute() or relative.is_absolute():
            raise NativeIdentityError("Root must be absolute and its artifact path relative")
        bound_root = root.resolve()
        path = (bound_root / relative).resolve()
        if not path.is_relative_to(bound_root):
            raise NativeIdentityError("Artifact path escapes its declared root")
        return path
    if not isinstance(location, (str, os.PathLike)):
        raise NativeIdentityError("Artifact path is unavailable")
    path = Path(location)
    if not path.is_absolute():
        raise NativeIdentityError("Loader resolution is unknown for a relative path or soname")
    return path


def _stat_identity(value):
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _hash_regular(path, per_artifact_limit, budget, *, native, bound_root=None):
    resolved = path.resolve(strict=True)
    if bound_root is not None and not resolved.is_relative_to(bound_root):
        raise NativeIdentityError("Artifact path escapes its declared root")
    # O_NONBLOCK also prevents a raced replacement by a FIFO from blocking.
    descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise NativeIdentityError("Artifact is not a regular file")
        if before.st_size > per_artifact_limit or before.st_size > budget[0]:
            raise NativeIdentityError("Artifact exceeds the per-artifact or total byte cap")
        budget[0] -= before.st_size
        digest = hashlib.sha256()
        remaining = before.st_size
        header = b""
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise NativeIdentityError("Artifact became shorter while hashing")
            digest.update(chunk)
            if len(header) < 4:
                header += chunk[:4 - len(header)]
            remaining -= len(chunk)
        after = os.fstat(handle.fileno())
    if (_stat_identity(before) != _stat_identity(after)
            or path.resolve(strict=True) != resolved
            or _stat_identity(after) != _stat_identity(resolved.stat())):
        raise NativeIdentityError("Artifact changed while hashing")
    if native and header != b"\x7fELF":
        raise NativeIdentityError("Native artifact does not have an ELF header")
    return {"path": str(resolved), "size_bytes": before.st_size,
            "sha256": digest.hexdigest(), "format_check": "elf_magic" if native else "not_checked"}


def capture_native_identity(
    required, *, python_modules=None, ctypes_libraries=None,
    device_code_paths=None, table_paths=None, roots=None,
    max_artifact_bytes, max_total_bytes,
):
    """Hash only required observed artifacts; missing values remain incomplete.

    ``required`` maps stable IDs to python_extension/ctypes_library/device_code/
    table. Native module/handle inputs must already exist; do not invoke an
    AITER loader or table resolver to populate them. Explicit path inputs may
    use {"root": name, "path": relative_path} with an absolute ``roots`` mapping.
    """
    _requirements(required)
    for cap in (max_artifact_bytes, max_total_bytes):
        if type(cap) is not int or cap <= 0:
            raise ValueError("Explicit positive integer byte caps are required")
    inputs = {"python_extension": python_modules or {},
              "ctypes_library": ctypes_libraries or {},
              "device_code": device_code_paths or {}, "table": table_paths or {}}
    for kind, entries in inputs.items():
        if not isinstance(entries, dict) or any(required.get(key) != kind for key in entries):
            raise ValueError("Observed artifact IDs and kinds must match the declaration")
    roots = roots or {}
    if not isinstance(roots, dict) or any(not Path(value).is_absolute() for value in roots.values()):
        raise ValueError("Root mappings must contain absolute paths")
    roots = {key: Path(value).resolve(strict=True) for key, value in roots.items()}
    artifacts = {}
    budget = [max_total_bytes]
    for key, kind in required.items():
        row = {"kind": kind, "evidence_kind": EVIDENCE_KINDS[kind]}
        artifacts[key] = row
        try:
            value = inputs[kind].get(key)
            if kind == "python_extension":
                if not isinstance(value, ModuleType):
                    raise NativeIdentityError("Loaded Python extension module is unavailable")
                location = vars(value).get("__file__")
                row["module_name"] = vars(value).get("__name__")
                loader = vars(value).get("__loader__")
                if not isinstance(loader, ExtensionFileLoader):
                    raise NativeIdentityError("Python extension loader metadata is unknown")
                row["loader_type"] = "ExtensionFileLoader"
            elif kind == "ctypes_library":
                if not isinstance(value, ctypes.CDLL):
                    raise NativeIdentityError("Loaded ctypes library handle is unavailable")
                location = vars(value).get("_name")
                if isinstance(location, bytes):
                    location = os.fsdecode(location)
            else:
                location = value
            if isinstance(location, (str, os.PathLike)):
                row["reported_path"] = os.fspath(location)
            path = _path(location, roots)
            if kind == "python_extension" and not str(path).endswith(tuple(EXTENSION_SUFFIXES)):
                raise NativeIdentityError("Python source is not a native extension path")
            if kind == "ctypes_library" and not re.search(r"\.so(?:\.\d+)*$", path.name):
                raise NativeIdentityError("ctypes path does not identify a shared library file")
            if kind == "device_code" and path.suffix not in {".co", ".hsaco"}:
                raise NativeIdentityError("Device-code path must identify a .co or .hsaco file")
            bound_root = roots[location["root"]] if isinstance(location, dict) else None
            row.update(_hash_regular(path, max_artifact_bytes, budget, native=kind != "table", bound_root=bound_root))
            row.update(status="identified", reported_path=os.fspath(path))
        except (OSError, ValueError, NativeIdentityError) as error:
            row.update(status="unresolved", reason=str(error))
    return {
        "schema_version": 1, "required": dict(required), "artifacts": artifacts,
        "declared_artifact_set_complete": all(row["status"] == "identified" for row in artifacts.values()),
        "limits": {"max_artifact_bytes": max_artifact_bytes, "max_total_bytes": max_total_bytes},
        "scope": SCOPE,
    }


def verify_native_identity(expected, observed, *, roots=None):
    """Compare a protected capture manifest with fresh observations or raise.

    Expected artifact paths may be replaced with explicit root/path mappings
    for a protected manifest that is portable between declared mount roots.
    Neither an empty declaration nor an incomplete record can pass.
    """
    try:
        _requirements(expected["required"])
        if expected["required"] != observed["required"]:
            raise NativeIdentityError("Required artifact IDs or kinds differ")
        for report in (expected, observed):
            if report["schema_version"] != 1 or report["declared_artifact_set_complete"] is not True:
                raise NativeIdentityError("Declared native artifact set is incomplete")
            if set(report["artifacts"]) != set(expected["required"]):
                raise NativeIdentityError("Required artifact is missing or an extra artifact was supplied")
        for key, kind in expected["required"].items():
            left, right = expected["artifacts"][key], observed["artifacts"][key]
            for row in (left, right):
                if (row["status"] != "identified" or row["kind"] != kind
                        or row["evidence_kind"] != EVIDENCE_KINDS[kind]
                        or row["format_check"] != ("not_checked" if kind == "table" else "elf_magic")
                        or (kind == "python_extension" and row["loader_type"] != "ExtensionFileLoader")
                        or type(row["size_bytes"]) is not int or row["size_bytes"] < 0
                        or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
                    raise NativeIdentityError(f"Invalid artifact evidence: {key}")
            expected_path = _path(left["path"], roots or {})
            if (expected_path != _path(right["path"], {})
                    or left["size_bytes"] != right["size_bytes"]
                    or left["sha256"] != right["sha256"]
                    or (kind == "python_extension" and left["module_name"] != right["module_name"])):
                raise NativeIdentityError(f"Artifact identity differs: {key}")
    except (KeyError, TypeError, ValueError) as error:
        raise NativeIdentityError("Malformed native artifact identity record") from error
