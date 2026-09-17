"""Expose existing TaskSession evidence without trusting candidate-owned locators."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import uuid

import yaml


def _read_regular(path: Path) -> bytes:
    if path.absolute() != path.resolve(strict=True) or not path.is_file():
        raise RuntimeError(f"Review evidence must be a regular file without symlinks: {path}")
    return path.read_bytes()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _serialize(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def snapshot_context_files(session) -> dict[Path, bytes]:
    # Called immediately after initial validation, before the optimizer starts.
    return {session.state_directory / name: _read_regular(session.state_directory / name)
            for name in ("agent_context.json", "validation_context.json")}


def snapshot_candidate_actions(session, previous: set[str]) -> str:
    records = {}
    for action in ("compile", "correctness", "performance"):
        executed = session.results.get(("candidate_evaluation", "candidate", action))
        # A later failed evaluation must not advertise earlier successful actions.
        if executed is not None and executed.invocation_id not in previous:
            records[action] = {
                "phase": "candidate_evaluation", "invocation_id": executed.invocation_id,
                "result": executed.result.to_mapping(),
                "commands": [asdict(command) for command in executed.commands],
            }
    return _serialize(records)


@dataclass(frozen=True)
class ReviewEvidence:
    path: Path
    sha256: str
    files: tuple[tuple[Path, bytes], ...]

    def verify(self) -> None:
        for path, expected in self.files:
            if _read_regular(path) != expected:
                raise RuntimeError(f"Modified protected evaluation evidence: {path}")


def expose_review_evidence(session, *, context_files: dict[Path, bytes],
                           evaluated_result: str, candidate_actions: str,
                           result: dict) -> ReviewEvidence:
    """Create a locator from controller memory, never from task_result pointers.

    The raw stdout envelopes were parsed and case-checked by TaskSession. Match
    their on-disk copies to the exact in-memory ExecutedAction, including argv,
    invocation, return code and output. This is transport, not a new scoring gate.
    """
    if (_serialize(result) != evaluated_result or result.get("task_name") != session.spec.task_id):
        raise RuntimeError("Review result does not match the controller's current evaluation")
    session.verify_candidate_harness()
    session.verify_baseline_sources()
    source = session.candidate_source_evidence()
    if source["error"] or source["sources"] != result.get("evaluated_candidate_sources"):
        raise RuntimeError("Review candidate does not match the evaluated source bytes")
    state = session.state_directory
    if state.resolve().is_relative_to(session.workspace.resolve()):
        raise RuntimeError("Review evidence must be outside the candidate workspace")
    files = dict(context_files)
    result_path = session.workspace / "task_result.yaml"
    files[result_path] = _read_regular(result_path)
    if yaml.safe_load(files[result_path]) != result:
        raise RuntimeError("Review report differs from the current evaluation")
    for path, expected in files.items():
        if _read_regular(path) != expected:
            raise RuntimeError(f"Modified protected evaluation evidence: {path}")

    def locator(path):
        return {"path": str(path), "sha256": _sha256(files[path])}

    records = json.loads(candidate_actions)
    actions = {action: {"status": "NO_COMPLETED_ACTION", "record": None}
               for action in ("compile", "correctness", "performance")}
    matches = {action: [] for action in records}
    for path in sorted(state.glob("action-*.json")):
        data = _read_regular(path)
        record = json.loads(data)
        for action, expected in records.items():
            if record.get("invocation_id") == expected["invocation_id"]:
                if record != expected:
                    raise RuntimeError(f"Action evidence differs from executed {action}: {path}")
                matches[action].append(path)
                files[path] = data
    for action, expected in records.items():
        if len(matches[action]) != 1:
            raise RuntimeError(f"Review requires exactly one current {action} action record")
        actions[action] = {"status": expected["result"]["status"],
                           "invocation_id": expected["invocation_id"],
                           "record": locator(matches[action][0])}
    data = (_serialize({
        "version": 1, "task_id": session.spec.task_id,
        "workspace": str(session.workspace), "candidate_sources": source["sources"],
        "result": locator(result_path),
        "contexts": {path.name: locator(path) for path in context_files},
        "candidate_actions": actions,
    }) + "\n").encode()
    path = state / f"review-evidence-{uuid.uuid4().hex}.json"
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(state, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    files[path] = data
    evidence = ReviewEvidence(path, _sha256(data), tuple(files.items()))
    evidence.verify()
    return evidence
