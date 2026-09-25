"""Validate source-only delivery in a scratch copy before installing any bytes."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess

from src.harness_guard import verify_workspace_harness

from .contract import canonical, digest, regular_file


MAX_BYTES = 64 * 1024 * 1024


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def read_json(path: Path, limit: int = 8 * 1024 * 1024) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Missing, unsafe or oversized Apex JSON artifact")
    data = json.loads(path.read_text(), object_pairs_hook=_unique)
    if not isinstance(data, dict):
        raise ValueError("Apex JSON artifact must be an object")
    canonical(data)  # Reject non-finite values, too.
    return data


def _git(root: Path, *args: str, payload: bytes) -> bytes:
    result = subprocess.run(["git", "-C", str(root), "apply", *args], input=payload,
                            capture_output=True, timeout=30, check=True)
    return result.stdout


def prepare_delivery(result: dict, *, task: dict, original: dict[str, bytes],
                     artifacts: Path, source: Path, harness) -> dict[str, bytes]:
    if (result.get("schema_version") != 1 or result.get("task_id") != task["task_id"]
            or result.get("applied") is not False
            or result.get("external_verification_required") is not True):
        raise ValueError("Apex result does not match the requested unapplied task")
    if result.get("status") == "no_gain":
        if result.get("bundle_path") or result.get("bundle_digest") or result.get("changed_files"):
            raise ValueError("Apex no_gain result must not deliver a candidate")
        return {}
    if result.get("status") != "candidate_ready":
        raise ValueError(f"Apex did not deliver a candidate: {result.get('reason_code', 'unknown')}")
    raw = result.get("bundle_path")
    if not isinstance(raw, str):
        raise ValueError("Apex candidate is missing its bundle")
    root = Path(raw)
    relative = root.relative_to(artifacts).as_posix()
    manifest_path = regular_file(artifacts, relative + "/bundle.json")
    manifest = read_json(manifest_path)
    baseline = {name: digest(data) for name, data in original.items()}
    changed = manifest.get("changed_files")
    if (manifest.get("schema_version") != 1 or manifest.get("task_id") != task["task_id"]
            or manifest.get("delivery") != {"mode": "bundle", "applied": False}
            or manifest.get("baseline", {}).get("file_hashes") != baseline
            or not isinstance(changed, list) or not changed
            or any(not isinstance(name, str) for name in changed)
            or len(changed) != len(set(changed)) or not set(changed) <= baseline.keys()
            or changed != result.get("changed_files")
            or set(manifest.get("candidate_file_hashes", {})) != set(changed)):
        raise ValueError("Apex bundle task, source scope or baseline mismatch")
    entries = manifest.get("patches")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Apex bundle has no patches")
    payloads = []
    declared = {"bundle.json"}
    size = len(canonical(manifest))
    for entry in entries:
        name = entry["path"]
        path = regular_file(root, name)
        size += path.stat().st_size
        if name in declared or size > MAX_BYTES:
            raise ValueError("Apex bundle has duplicate or oversized patches")
        data = path.read_bytes()
        if digest(data) != entry["sha256"]:
            raise ValueError("Apex patch hash mismatch")
        payloads.append(data)
        declared.add(name)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Apex bundle cannot contain links")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != declared or digest(canonical(manifest) + b"".join(payloads)) != result.get("bundle_digest"):
        raise ValueError("Apex bundle file set or digest mismatch")
    if any(regular_file(source, name).read_bytes() != data for name, data in original.items()):
        raise ValueError("Apex input source changed during optimization")
    verify_workspace_harness(replace(harness, root=source), discard_added=False)
    scratch = artifacts / "delivery-check"
    shutil.copytree(source, scratch)
    payload = b"".join(payloads)
    if _git(scratch, "--summary", payload=payload).strip():
        raise ValueError("Apex bundle cannot add, delete, rename or change file modes")
    records = _git(scratch, "--numstat", "-z", payload=payload).decode().split("\0")
    targets = []
    for record in filter(None, records):
        added, removed, name = record.split("\t", 2)
        if not added.isdigit() or not removed.isdigit():
            raise ValueError("Apex bundle cannot contain binary patches")
        targets.append(name)
    if len(targets) != len(set(targets)) or set(targets) != set(changed):
        raise ValueError("Apex patch targets differ from declared candidate files")
    _git(scratch, "--check", "--whitespace=nowarn", payload=payload)
    _git(scratch, "--whitespace=nowarn", payload=payload)
    verify_workspace_harness(replace(harness, root=scratch), discard_added=False)
    candidate = {name: regular_file(scratch, name).read_bytes() for name in changed}
    if {name: digest(data) for name, data in candidate.items()} != manifest["candidate_file_hashes"]:
        raise ValueError("Applied Apex source hash mismatch")
    return candidate
