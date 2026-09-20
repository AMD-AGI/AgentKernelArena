"""Validate explicit JSON evidence for reusing a native verifier phase prefix.

Only original successful compile/correctness phases can be reused. No code or
pickle from the evidence bundle is imported, and performance is always fresh.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

from src.tools.runtime_image_identity import verify_identity


SCHEMA = "aka-native-prefix-resume-v1"
PREFIX = ("compile", "correctness")
MAX_JSON_BYTES = 32 * 1024 * 1024


def checked_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("Evidence paths must be nonempty relative paths")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts:
        raise ValueError("Evidence path escapes its explicitly selected bundle")
    path = root
    for part in parts.parts:
        path /= part
        if path.is_symlink():
            raise ValueError("Evidence paths must not contain symlinks")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes its bundle")
    return path


def read_pinned_json(path: Path, expected_sha256: str, expected_bytes: int | None = None):
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("A complete SHA-256 pin is required for resume evidence")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_JSON_BYTES:
            raise ValueError("Resume evidence must be bounded regular JSON files")
        raw = stream.read(MAX_JSON_BYTES + 1)
        after = os.fstat(stream.fileno())
    if (len(raw) > MAX_JSON_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha256
            or (expected_bytes is not None and len(raw) != expected_bytes)
            or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
        raise ValueError(f"Resume evidence hash/size mismatch or concurrent modification: {path.name}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Resume evidence must be JSON objects")
    return value, raw


def current_gpu_arch() -> str:
    import torch
    if not torch.cuda.is_available():
        raise ValueError("Prefix reuse requires an available current ROCm GPU")
    return str(torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName).split(":", 1)[0]


def validate_prefix(manifest_path: Path, manifest_sha256: str, *, selector: str,
                    source_identity: dict, task_config: dict, workspace: Path,
                    runtime_identity: dict, current_plan: dict) -> dict:
    manifest, raw_manifest = read_pinned_json(manifest_path, manifest_sha256)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported native prefix evidence schema")
    root = manifest_path.parent.resolve()
    old_repository = checked_path(root, manifest.get("repository_root"))
    old_workspace = checked_path(root, manifest.get("workspace"))
    loaded, snapshots, paths = {}, {"manifest": raw_manifest}, {}
    for name in ("task_report", "run_report", "runtime_report"):
        declaration = manifest.get(name) or {}
        if type(declaration.get("bytes")) is not int or declaration["bytes"] < 1:
            raise ValueError(f"Resume evidence needs a byte-size pin for {name}")
        paths[name] = checked_path(root, declaration.get("path"))
        loaded[name], snapshots[name] = read_pinned_json(
            paths[name], declaration.get("sha256"), declaration["bytes"])
    if (paths["task_report"] != old_workspace.parent / (old_workspace.name + ".direct.json")
            or paths["run_report"] != old_workspace.parent / "direct-verification.json"
            or paths["runtime_report"] != old_workspace / "build/runtime_preflight.json"):
        raise ValueError("Resume reports do not belong to the explicitly selected prior workspace")
    task, run, runtime = (loaded[key] for key in ("task_report", "run_report", "runtime_report"))
    if (task.get("schema") != "aka-direct-task-verification-v1"
            or run.get("schema") != "aka-direct-verification-v1"
            or task.get("task") != selector or selector not in run.get("plan", {}).get("tasks", [])
            or checked_path(old_repository, task.get("workspace")) != old_workspace
            or selector not in (run.get("shard") or {}).get("assigned_tasks", [])
            or task.get("resume_provenance") is not None
            or task.get("framework_PASS_claimed") is not False
            or task.get("framework_task_validator") != "NOT_RUN"):
        raise ValueError("Prior direct-verifier task/run identity is inconsistent")
    if task.get("source_identity") != source_identity:
        raise ValueError("Current materialized task source_identity differs from the prior successful prefix")
    # v1 source_identity deliberately excludes .pt payloads. Until a pinned
    # full-input identity is available, reject them instead of treating them as
    # verified or deserializing old tensor data.
    if any(path.is_file() for path in workspace.rglob("*.pt")):
        raise ValueError("Prefix reuse requires fully fingerprinted generated inputs; .pt payloads are not supported")
    metadata = task_config.get("headkernel") or {}
    expected_image = metadata.get("docker")
    expected_config = (metadata.get("runtime") or {}).get("expected_image_id")
    if (not isinstance(expected_image, str)
            or re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", expected_image) is None
            or not isinstance(expected_config, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", expected_config) is None):
        raise ValueError("Prefix reuse requires pinned current manifest and config digests")
    previous = run.get("runtime_identity") or {}
    proof = json.loads(previous.get("AGENT_KERNEL_ARENA_DOCKER_IDENTITY") or "{}")
    prior_identity = verify_identity(expected_image, expected_config, {
        "Id": previous.get("AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID"),
        "RepoDigests": json.loads(previous.get("AGENT_KERNEL_ARENA_DOCKER_REPO_DIGESTS") or "[]"),
        "Descriptor": proof.get("descriptor"),
    })
    if (run.get("plan", {}).get("image") != expected_image
            or run.get("plan", {}).get("target_gpu_model") != current_plan.get("target_gpu_model")
            or current_plan.get("target_gpu_model") != "MI355X"
            or run.get("plan", {}).get("expected_image_id") != expected_config
            or previous.get("AGENT_KERNEL_ARENA_DOCKER_IMAGE") != expected_image
            or previous.get("AGENT_KERNEL_ARENA_DOCKER_CONFIG_DIGEST") != expected_config
            or prior_identity["verified_config_digest"] != expected_config
            or runtime_identity.get("AGENT_KERNEL_ARENA_DOCKER_IMAGE") != expected_image
            or runtime_identity.get("AGENT_KERNEL_ARENA_DOCKER_CONFIG_DIGEST") != expected_config
            or previous.get("AGENT_KERNEL_ARENA_GPU_ARCH") != "gfx950"
            or runtime_identity.get("AGENT_KERNEL_ARENA_GPU_ARCH") != "gfx950"
            or runtime.get("status") != "ok" or runtime.get("phase") != "complete"
            or runtime.get("native_resolution_complete") is not True
            or runtime.get("selected_image") != expected_image
            or runtime.get("verified_config_digest") != expected_config
            or not (metadata.get("runtime") or {}).get("profile")
            or runtime.get("profile") != metadata["runtime"]["profile"]
            or runtime.get("architecture") != "gfx950"):
        raise ValueError("Prior/current runtime image, config digest or completed gfx950 preflight differs")
    for name, value in current_plan.get("required_environment", {}).items():
        if (runtime.get("environment", {}).get(name) != value
                or previous.get(name) != value
                or runtime_identity.get(name) != value):
            raise ValueError(f"Prior/current required runtime environment differs: {name}")
    old_phases = task.get("phases") or []
    if [entry.get("phase") for entry in old_phases] not in [list(PREFIX), [*PREFIX, "performance"]]:
        raise ValueError("A complete original compile/correctness prefix is required")
    reports = {}
    for phase, entry in zip(PREFIX, old_phases):
        commands = task_config.get(phase + "_command")
        executions = entry.get("commands") or []
        if (entry.get("status") != "native_phase_succeeded" or entry.get("native_status") != "ok"
                or "prior_phase" in entry or entry.get("executed_here", True) is not True
                or entry.get("executed_in_this_run", True) is not True
                or not isinstance(commands, list) or not commands
                or [execution.get("command") for execution in executions] != commands
                or any(type(execution.get("returncode")) is not int or execution["returncode"] != 0
                       or execution.get("timed_out") is not False
                       for execution in executions)
                or entry.get("timeout_seconds") != task_config.get(phase + "_timeout", 3600)):
            raise ValueError(f"Prior {phase} phase is failed, partial, reused, or has a different command contract")
        declaration = entry.get("native_report") or {}
        if (declaration.get("path") != f"direct-native-reports/{phase}_report.json"
                or type(declaration.get("bytes")) is not int or declaration["bytes"] < 1):
            raise ValueError("Prior phase must reference its retained native report")
        native, raw = read_pinned_json(checked_path(old_workspace, declaration["path"]),
                                       declaration.get("sha256"), declaration.get("bytes"))
        if native.get("status") != "ok":
            raise ValueError(f"Retained native {phase} report did not pass")
        reports[phase] = {"entry": entry, "raw": raw}
    if current_gpu_arch() != "gfx950":
        raise ValueError("The current physical GPU is not gfx950")
    return {"manifest": manifest, "manifest_sha256": manifest_sha256,
            "snapshots": snapshots, "reports": reports,
            "prior_runtime_identity": previous, "prior_task_status": task.get("status")}


def retain_prefix(validated: dict, workspace: Path) -> tuple[list[dict], dict]:
    directory = workspace / "resume-evidence"
    directory.mkdir()
    for name, raw in validated["snapshots"].items():
        relative = "manifest.json" if name == "manifest" else validated["manifest"][name]["path"]
        snapshot = checked_path(directory, relative)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(raw)
    entries = []
    for phase in PREFIX:
        prior = validated["reports"][phase]
        prior_workspace = checked_path(directory, validated["manifest"]["workspace"])
        path = prior_workspace / "direct-native-reports" / (phase + "_report.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(prior["raw"])
        entries.append({"phase": phase, "status": "native_phase_reused", "native_status": "ok",
                        "executed_in_this_run": False, "executed_here": False, "commands": [],
                        "reused_at_utc": datetime.now(timezone.utc).isoformat(),
                        "prior_phase": copy.deepcopy(prior["entry"]),
                        "native_report": {"path": path.relative_to(workspace).as_posix(),
                                          "sha256": hashlib.sha256(prior["raw"]).hexdigest(),
                                          "bytes": len(prior["raw"])}})
    provenance = {"manifest": "resume-evidence/manifest.json",
                  "manifest_sha256": validated["manifest_sha256"], "reused_phases": list(PREFIX),
                  "origin": validated["manifest"].get("origin", {}),
                  "origin_assertion": "caller-supplied provenance retained under the manifest pin",
                  "prior_runtime_identity": validated["prior_runtime_identity"],
                  "prior_task_status": validated["prior_task_status"],
                  "performance_reused": False}
    return entries, provenance
