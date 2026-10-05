"""Materialize Git-pinned kernel fixtures on the trusted host, before task execution.

Only data codecs are accepted. No task importer is executed and no network or
credential mount is added to an evaluation container.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

if __package__:
    from ..task_contract import fingerprint, require, strict_json
    from .trusted_native_eval import read_regular
else:
    from task_contract import fingerprint, require, strict_json
    from trusted_native_eval import read_regular

SCHEMA = "trusted-external-fixtures-v1"
DEFAULT_MAX_FILES = 16384
DEFAULT_MAX_BYTES = 64 << 30
BATCH_FILES = 24
BATCH_BYTES = 64 << 20
ROLES = {"case_metadata", "activation", "runtime_control", "oracle_output", "kernel_weight", "kernel_weight_scale"}
CODECS = {"served-tensor-fixture-v1": ".json", "raw-storage-segment-v1": ".bin"}


def relative_path(value):
    require(isinstance(value, str) and bool(value) and "\\" not in value
            and not any(ord(c) < 32 or ord(c) == 127 for c in value), "invalid fixture relative path")
    path = PurePosixPath(value)
    require(path.parts and not path.is_absolute() and str(path) == value
            and all(p not in ("..", ".git", "build", "__pycache__") for p in path.parts),
            "fixture path escapes the immutable package")
    return value


def oci_prefix(value):
    require(isinstance(value, str) and re.fullmatch(r"oci:[A-Za-z0-9_-]+/.+/", value) is not None,
            "fixture OCI prefix must be a pinned oci:remote/bucket/prefix/ path")
    relative_path(value.split(":", 1)[1].removesuffix("/"))
    require(not any(c in value for c in ("?", "#", "%", "@", "://")) and value.count(":") == 1,
            "fixture OCI prefix cannot contain credentials or URL parameters")
    return value


def paths_overlap(left, right):
    left, right = PurePosixPath(left), PurePosixPath(right)
    return left == right or left in right.parents or right in left.parents


def claim_path(name, files, directories):
    parents = {p.as_posix() for p in PurePosixPath(name).parents if p.as_posix() != "."}
    require(name not in files and name not in directories and not parents & files,
            "duplicate or overlapping fixture paths")
    files.add(name)
    directories.update(parents)


def load_fixture_manifest(task, path, cases, protected):
    """Read only the exact Git-extracted manifest; validate before any transfer."""
    path = relative_path(path)
    require(not any(paths_overlap(path, p) for p in protected), "fixture manifest overlaps protected/editable files")
    raw = read_regular(task / path)
    manifest = strict_json(raw)
    require(isinstance(manifest, dict) and manifest.get("schema") == SCHEMA, "unsupported fixture manifest schema")
    require(manifest.get("case_manifest_fingerprint") == fingerprint(cases), "fixture manifest names different cases")
    require(manifest.get("runtime_image") == cases["runtime_image"], "fixture manifest names a different runtime image")
    oci_prefix(manifest.get("oci_prefix"))
    assets = manifest.get("assets")
    require(isinstance(assets, list) and bool(assets), "fixture manifest requires explicit assets")
    destinations, objects, destination_dirs, object_dirs = set(), set(), set(), set()
    for row in assets:
        require(isinstance(row, dict), "invalid fixture asset")
        name, key = relative_path(row.get("path")), relative_path(row.get("object_key"))
        require(PurePosixPath(name).parts[0] == "fixtures" and len(PurePosixPath(name).parts) > 1,
                "fixture destinations must be beneath fixtures/")
        require(not any(paths_overlap(name, p) for p in (*protected, path)),
                "fixture destination overlaps protected/editable files or another asset")
        claim_path(name, destinations, destination_dirs)
        claim_path(key, objects, object_dirs)
        require(type(row.get("bytes")) is int and row["bytes"] >= 0
                and isinstance(row.get("sha256"), str) and re.fullmatch(r"[a-f0-9]{64}", row["sha256"]),
                "fixture asset requires exact bytes and SHA256")
        codec = row.get("codec")
        require(codec in CODECS and PurePosixPath(name).suffix == CODECS[codec], "unsupported fixture data codec")
        require(codec != "served-tensor-fixture-v1" or row["bytes"] <= 64 << 20,
                "fixture JSON exceeds the 64 MiB metadata limit")
        roles = row.get("roles")
        require(isinstance(roles, list) and roles and all(isinstance(r, str) for r in roles)
                and set(roles) <= ROLES and len(set(roles)) == len(roles), "unsupported fixture tensor roles")
        require((roles == ["case_metadata"]) == (codec == "served-tensor-fixture-v1"),
                "fixture metadata and storage roles disagree")
        require(codec == "served-tensor-fixture-v1" or "case_metadata" not in roles,
                "raw storage cannot declare a metadata role")
        # Existing Git files are immutable. Even equal content must not be overwritten.
        destination = task / name
        require(not destination.exists() and not destination.is_symlink(), "fixture destination already exists in Git task")
        for parent in destination.parents:
            if parent == task:
                break
            require(not parent.is_symlink() and (not parent.exists() or parent.is_dir()), "invalid fixture destination parent")
    return {"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "manifest": manifest}


def file_digest(path, expected_size):
    """Stream large raw tensors through a no-symlink file descriptor."""
    path = Path(os.path.abspath(path))
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            require(stat.S_ISREG(before.st_mode) and before.st_size == expected_size, "fixture is missing or has wrong size")
            digest, count = hashlib.sha256(), 0
            for part in iter(lambda: handle.read(8 << 20), b""):
                count += len(part)
                require(count <= expected_size, "fixture grew during verification")
                digest.update(part)
            after = os.fstat(handle.fileno())
            require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                    (after.st_size, after.st_mtime_ns, after.st_ctime_ns) and count == expected_size,
                    "fixture changed during verification")
            return digest.hexdigest()
    finally:
        os.close(directory)


def verify_assets(root, assets, key="path"):
    for row in assets:
        require(file_digest(root / row[key], row["bytes"]) == row["sha256"], "fixture SHA256 mismatch: " + row[key])


def verify_closure(root, assets, cases):
    """Resolve fixture JSON and raw segments as data, without importing task code."""
    rows = {row["path"]: row for row in assets}
    referenced_cases, referenced_blobs = set(), set()
    for case in cases["cases"]:
        for field in ("fixture", "live_fixture"):
            if field not in case:
                continue
            ref = case[field]
            require(isinstance(ref, dict), "invalid case fixture reference")
            name = relative_path(ref.get("path"))
            row = rows.get(name, {})
            require(row.get("codec") == "served-tensor-fixture-v1" and row.get("sha256") == ref.get("sha256"),
                    "case references missing or mismatched fixture")
            referenced_cases.add(name)
    metadata = {r["path"] for r in assets if r["codec"] == "served-tensor-fixture-v1"}
    require(metadata == referenced_cases, "fixture metadata must be referenced by the case manifest")
    for name in sorted(metadata):
        fixture = strict_json(read_regular(root / name))
        require(isinstance(fixture, dict) and fixture.get("schema") == "served-tensor-fixture-v1", "unsupported case fixture schema")
        payload = fixture.get("payload")
        require(isinstance(payload, dict) and bool(payload) and set(payload) <= {"inputs", "outputs"}, "invalid fixture payload")
        for groups in payload.values():
            require(isinstance(groups, dict), "invalid fixture alias groups")
            for group in groups.values():
                require(isinstance(group, dict) and type(group.get("storage_nbytes")) is int
                        and group["storage_nbytes"] >= 0 and isinstance(group.get("segments"), list), "invalid fixture storage")
                spans = []
                for segment in group["segments"]:
                    require(isinstance(segment, dict), "invalid fixture segment")
                    blob = (PurePosixPath(name).parent / relative_path(segment.get("blob"))).as_posix()
                    row = rows.get(blob, {})
                    require(row.get("codec") == "raw-storage-segment-v1" and row.get("sha256") == segment.get("sha256")
                            and row.get("bytes") == segment.get("bytes"), "fixture references missing or mismatched segment")
                    offset, size = segment.get("offset_bytes"), segment.get("bytes")
                    require(type(offset) is int and type(size) is int and 0 <= offset <= offset + size <= group["storage_nbytes"],
                            "fixture segment escapes original storage")
                    spans.append((offset, offset + size))
                    referenced_blobs.add(blob)
                spans.sort()
                require(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "fixture storage segments overlap")
    blobs = {r["path"] for r in assets if r["codec"] == "raw-storage-segment-v1"}
    require(blobs == referenced_blobs, "unreferenced fixture blob")
    return {"case_fixtures": len(metadata), "storage_blobs": len(blobs)}


def batches(assets):
    batch, size = [], 0
    for row in sorted(assets, key=lambda r: r["object_key"]):
        if batch and (len(batch) == BATCH_FILES or size + row["bytes"] > BATCH_BYTES):
            yield batch
            batch, size = [], 0
        batch.append(row)
        size += row["bytes"]
    if batch:
        yield batch


def materialize_fixtures(task, descriptor, cases, *, candidate_workspace, staging,
                         local_mirror=None, allowed_oci_prefix=None, max_files=DEFAULT_MAX_FILES,
                         max_bytes=DEFAULT_MAX_BYTES, timeout=1800):
    manifest, assets = descriptor["manifest"], descriptor["manifest"]["assets"]
    require(type(max_files) is int and max_files > 0 and type(max_bytes) is int and max_bytes > 0,
            "fixture host budgets must be positive integers")
    total = sum(row["bytes"] for row in assets)
    require(len(assets) <= max_files and total <= max_bytes, "fixture assets exceed trusted host budget")
    require(timeout > 0, "fixture transfer timeout must be positive")
    candidate_workspace = Path(candidate_workspace).resolve()
    mirror = None
    if local_mirror is not None:
        mirror = Path(local_mirror).absolute()
        require(mirror.resolve() == mirror and mirror.is_dir(), "fixture local mirror must be a canonical directory")
        require(not mirror.is_relative_to(candidate_workspace) and not candidate_workspace.is_relative_to(mirror),
                "fixture local mirror overlaps candidate workspace")
        verify_assets(mirror, assets, "object_key")
    else:
        require(allowed_oci_prefix is not None and oci_prefix(allowed_oci_prefix) == manifest["oci_prefix"],
                "fixture download requires the host-approved exact OCI prefix")
    if allowed_oci_prefix is not None:
        require(oci_prefix(allowed_oci_prefix) == manifest["oci_prefix"], "fixture OCI prefix differs from host approval")
    # Download transaction plus complete reference and candidate task copies.
    require(shutil.disk_usage(staging).free >= 3 * total + (64 << 20), "insufficient fixture scratch space")
    with tempfile.TemporaryDirectory(prefix="fixtures-", dir=staging) as temporary:
        transfer = Path(temporary)
        payload = transfer / "payload"
        payload.mkdir()
        environment = dict(os.environ, GOMAXPROCS="1")
        transfers = []
        for index, batch in enumerate(batches(assets)):
            listing = transfer / (str(index) + ".files")
            listing.write_text("".join(r["object_key"] + "\n" for r in batch))
            command = ["rclone", "copy", str(mirror) if mirror is not None else manifest["oci_prefix"], str(payload),
                       "--files-from-raw", str(listing), "--transfers", "64000", "--progress", "--buffer-size", "0",
                       "--multi-thread-streams", "0", "--checkers", "2", "--no-traverse"]
            if mirror is not None:
                command.extend(["--config", os.devnull])
            subprocess.run(command, check=True, timeout=timeout, env=environment,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            transfers.append({"files": len(batch), "bytes": sum(r["bytes"] for r in batch)})
        verify_assets(payload, assets, "object_key")
        actual = {p.relative_to(payload).as_posix() for p in payload.rglob("*") if not p.is_dir()}
        require(actual == {r["object_key"] for r in assets}, "fixture download file set differs from manifest")
        if mirror is not None:
            verify_assets(mirror, assets, "object_key")
        # Renames within the private transaction map remote keys to task paths.
        mapped = transfer / "mapped"
        mapped.mkdir()
        for row in assets:
            destination = mapped / row["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            (payload / row["object_key"]).rename(destination)
        closure = verify_closure(mapped, assets, cases)
        for row in assets:
            destination = task / row["path"]
            require(not destination.exists() and not destination.is_symlink(), "fixture destination changed before install")
            destination.parent.mkdir(parents=True, exist_ok=True)
            (mapped / row["path"]).rename(destination)
        verify_assets(task, assets)
    return {"schema": "trusted-fixtures-receipt-v1", "manifest_path": descriptor["path"],
            "manifest_sha256": descriptor["sha256"], "case_manifest_fingerprint": fingerprint(cases),
            "runtime_image": manifest["runtime_image"], "oci_prefix": manifest["oci_prefix"],
            "source": "verified_local_mirror" if mirror is not None else "verified_oci_download",
            "files": len(assets), "bytes": total, "assets": assets, "closure": closure,
            "host_limits": {"files": max_files, "bytes": max_bytes}, "transfer_batches": transfers}
