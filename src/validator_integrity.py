"""Framework-owned post-validator integrity and portable completion evidence.

The control record lives outside the agent workspace. Its exported receipt is
hash-bound by the completion marker so an archived workspace is self-contained.
This is an integrity check, not a process or filesystem security sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import uuid

import yaml

from .eval_tools.evidence import SubmissionEvidence, capture_submission_evidence
from .harness_guard import HarnessIntegrityError, WorkspaceSnapshot, verify_workspace_harness


EXPORT_NAME = "validation_integrity.json"
REPORT_LIFECYCLE = "framework_post_validator_integrity_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Integrity record is not a regular file: " + str(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Integrity record must be a mapping")
    return value


def _control_directory(workspace: Path) -> Path:
    return workspace.parent / ".validator-integrity" / workspace.name


@dataclass(frozen=True)
class ValidatorIntegrity:
    workspace: Path
    snapshot: WorkspaceSnapshot
    attempt: str
    storage: Path
    before_sha256: str
    before: dict
    sources: SubmissionEvidence | None


def begin_validator_integrity(
    workspace: Path, task_config: dict, snapshot: WorkspaceSnapshot
) -> ValidatorIntegrity:
    """Persist the actual pre-agent snapshot and mark this attempt incomplete."""
    workspace = workspace.absolute()
    control = _control_directory(workspace)
    for path in (control.parent, control):
        if path.is_symlink():
            raise ValueError("Validator integrity directories must not be symlinks")
        path.mkdir(exist_ok=True)
    attempt = uuid.uuid4().hex
    storage = control / attempt
    storage.mkdir()
    policy = task_config.get("harness_protection", {})
    freeze_sources = (
        isinstance(task_config.get("trusted_evaluation"), dict)
        or isinstance(policy, dict) and policy.get("freeze_task_inputs") is True
    )
    # Reuse declared submission paths and their contained repository-aware
    # resolver. Ordinary optimizers and non-opted-in validator sources retain
    # their existing editability; no whole-repository source boundary is guessed.
    sources = capture_submission_evidence(workspace, task_config, storage / "sources") if freeze_sources else None
    before = {
        "schema_version": 1, "attempt": attempt, "observed_unix": time.time(),
        "harness_digests": snapshot.digests, "digest_modes": snapshot.digest_modes,
        "path_states": snapshot.path_states, "source_aliases": snapshot.source_aliases,
        "raw_declared_source_freeze": freeze_sources,
        "source_manifest": sources.manifest if sources is not None else None,
        # The manifest hash also includes schema/workspace metadata. Record
        # the same declared-entry fingerprint used by the post-run check.
        "source_fingerprint": sources.candidate_fingerprint() if sources is not None else None,
    }
    _atomic_json(storage / "before.json", before)
    _atomic_json(control / "active.json", {"schema_version": 1, "attempt": attempt, "status": "PENDING"})
    return ValidatorIntegrity(workspace, snapshot, attempt, storage, _sha256(storage / "before.json"), before, sources)


def integrity_marker_requirement(workspace: Path) -> dict | None:
    """An early launcher report must not complete an unfinished main audit."""
    active = _control_directory(Path(workspace)) / "active.json"
    if not active.exists():
        return None
    record = _read_json(active)
    return {"required": True, "attempt": record["attempt"], "status": "PENDING"}


def finish_validator_integrity(
    context: ValidatorIntegrity, *, task_name: str, launcher_error: str | None = None
) -> bool:
    """Verify observed inputs, retain failures, and bind the actual receipt."""
    from agents.task_validator.report_schema import invalidate_integrity_report

    errors = []
    harness = None
    source_fingerprint = None
    try:
        if _sha256(context.storage / "before.json") != context.before_sha256:
            errors.append("Framework pre-agent integrity record changed")
    except OSError as exc:
        errors.append("Framework pre-agent integrity record is unavailable: " + str(exc))
    try:
        harness = verify_workspace_harness(context.snapshot, reject_added=True)
    except HarnessIntegrityError as exc:
        harness = exc.verification
        errors.append(str(exc))
    try:
        if context.sources is not None:
            context.sources.verify()
            if harness["changed_paths"]:
                errors.append("Declared source scan withheld after protected path topology changed")
            else:
                source_fingerprint = context.sources.candidate_fingerprint()
                if source_fingerprint != context.before["source_fingerprint"]:
                    errors.append("Original declared source bytes, presence or alias identity changed during validation")
    except (OSError, RuntimeError, ValueError) as exc:
        errors.append("Declared source verification failed: " + str(exc))

    workspace = context.workspace
    workspace_safe = not workspace.is_symlink() and workspace.is_dir()
    if not workspace_safe:
        errors.append("Validator workspace path changed or disappeared")
    if workspace_safe and (errors or launcher_error):
        # This path normalizes data only: do not execute a possibly modified
        # task validator/quality helper after integrity has already failed.
        try:
            invalidate_integrity_report(workspace, task_name, "; ".join([*errors, *([launcher_error] if launcher_error else [])]))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            errors.append("Unable to persist failed validator report: " + str(exc))
    report_path = workspace / "validation_report.yaml"
    receipt = {
        "schema_version": 1, "producer": "framework_post_validator_integrity",
        "attempt": context.attempt, "status": "FAIL" if errors else "PASS",
        "observed_unix": time.time(), "before_sha256": context.before_sha256,
        "before": context.before,
        "harness": harness,
        "after_source_fingerprint": source_fingerprint,
        "raw_declared_sources_unchanged": (
            source_fingerprint == context.before["source_fingerprint"]
            if source_fingerprint is not None else None
        ),
        "validation_report_sha256": _sha256(report_path) if workspace_safe and report_path.is_file() and not report_path.is_symlink() else None,
        "launcher_error": launcher_error, "errors": errors,
        "source_scope": "declared_submission_paths for trusted_evaluation/freeze_task_inputs only",
        "hash_scope": "harness digests use the stated raw or masked modes; opted-in declared sources have a separate full-byte fingerprint",
        "retirement_or_global_idle_verified": False,
    }
    result_path = context.storage / "result.json"
    _atomic_json(result_path, receipt)
    if workspace_safe:
        _atomic_json(workspace / EXPORT_NAME, receipt)
        digest = _sha256(result_path)
        marker_path = workspace / ".validation_complete"
        if marker_path.is_file() and not marker_path.is_symlink():
            marker = yaml.safe_load(marker_path.read_text())
            if isinstance(marker, dict) and marker.get("report_sha256") == receipt["validation_report_sha256"]:
                marker["validator_integrity"] = {
                    "required": True, "attempt": context.attempt, "status": receipt["status"],
                    "receipt": EXPORT_NAME, "receipt_sha256": digest,
                }
                temporary = marker_path.with_name(marker_path.name + "." + uuid.uuid4().hex + ".tmp")
                temporary.write_text(yaml.safe_dump(marker, sort_keys=False))
                temporary.replace(marker_path)
        _atomic_json(context.storage.parent / "active.json", {
            "schema_version": 1, "attempt": context.attempt, "status": "COMPLETE",
            "receipt_sha256": digest,
        })
    return not errors and launcher_error is None


def completion_integrity_matches(workspace: Path, marker: dict, report: dict) -> bool:
    """Check a live control record or the hash-bound portable export."""
    descriptor = marker.get("validator_integrity")
    active_path = _control_directory(workspace) / "active.json"
    if descriptor is None:
        # Only reports without a bound lifecycle can use legacy compatibility.
        # Removing marker metadata must not bypass a new archived audit.
        return "validator_integrity" not in report and not active_path.exists()
    try:
        declaration = report.get("validator_integrity")
        if (not isinstance(declaration, dict) or declaration.get("required") is not True
                or declaration.get("lifecycle") != REPORT_LIFECYCLE
                or not isinstance(declaration.get("attempt"), str) or not declaration["attempt"]):
            return False
        if not isinstance(descriptor, dict) or descriptor.get("required") is not True:
            return False
        if descriptor.get("attempt") != declaration["attempt"]:
            return False
        if descriptor.get("status") not in {"PASS", "FAIL"} or descriptor.get("receipt") != EXPORT_NAME:
            return False
        receipt_path = workspace / EXPORT_NAME
        receipt = _read_json(receipt_path)
        digest = _sha256(receipt_path)
        if (descriptor.get("receipt_sha256") != digest
                or receipt.get("attempt") != descriptor.get("attempt")
                or receipt.get("status") != descriptor.get("status")
                or receipt.get("producer") != "framework_post_validator_integrity"
                or receipt.get("validation_report_sha256") != marker.get("report_sha256")):
            return False
        if receipt["status"] == "FAIL" and report.get("overall_status") != "FAIL":
            return False
        if active_path.exists():
            active = _read_json(active_path)
            if (active.get("status") != "COMPLETE" or active.get("attempt") != receipt["attempt"]
                    or active.get("receipt_sha256") != digest):
                return False
        return True
    except (OSError, ValueError, KeyError):
        return False
