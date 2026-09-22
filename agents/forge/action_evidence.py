"""Append-only command diagnostics outside disposable task build workspaces."""
from __future__ import annotations

import base64
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

from src.task_spec import resolve_task_path
from agents.forge.bundles import candidate_files, protected_paths


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_binding(context, root: Path) -> dict:
    """Bind declared inputs, not transient compiler outputs or a copied build."""
    files = (set(candidate_files(context.spec, root, required=False))
             | {edit.path for edit in context.spec.candidate.editable}
             | protected_paths(context.spec))
    entries = {}
    for name in sorted(files):
        path = resolve_task_path(root, name)
        entries[name] = ({"sha256": digest(path)} if path.is_file() else
                         {"kind": "directory" if path.is_dir() else "missing"})
    binding = {"scope": "declared candidate files and protected task inputs; excludes build outputs",
               "files": entries, "context_path": str(context.path),
               "context_sha256": digest(context.path),
               "baseline_workspace": str(context.baseline_workspace)}
    # TaskSession's external inventory binds the complete original materialized
    # source without rehashing gigabytes of generated JIT/build files per action.
    # Older contexts/test fixtures need not expose that optional inventory.
    initial = context.path.parent / "initial_sources.json"
    if initial.is_file():
        binding["initial_source_inventory"] = {"path": str(initial), "sha256": digest(initial)}
    return binding


def _json_default(value):
    # run_action normally returns text. Preserve bytes if an execution error
    # carries them; decoding with replacement would destroy diagnostic evidence.
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    raise TypeError(f"Unsupported diagnostic value: {type(value).__name__}")


def _write_once(path: Path, data: dict) -> None:
    serialized = json.dumps(data, indent=2, default=_json_default) + "\n"
    # Never replace an earlier invocation/result. A partial write is not a valid
    # result and must not be mistaken for a completed command record.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


class ActionEvidence:
    def __init__(self, plan, context, root, *, evaluation_id, role, action,
                 requested_action, source, spec, engine_root):
        artifact = Path(plan["template"]).resolve(strict=True).parent
        store = artifact / "action-evidence"
        store.mkdir(mode=0o700, exist_ok=True)
        if store.is_symlink():
            raise ValueError("Forge action evidence directory must not be a symlink")
        self.directory = Path(tempfile.mkdtemp(prefix=f"{role}-{action}-", dir=store))
        self.result_path = self.directory / "result.json"
        self.header = {"version": 1, "evaluation_id": evaluation_id,
                       "action_attempt_id": uuid.uuid4().hex, "task_id": context.spec.task_id,
                       "role": role, "action": action, "requested_action": requested_action,
                       "phase": "candidate_evaluation", "workspace": str(root),
                       "engine_root": str(engine_root),
                       "master_engine_root": str(plan["engine_root"]), "source": source,
                       "declared_argv": spec.action(role, action).commands,
                       "timeout_s": spec.action(role, action).timeout_s,
                       "started_utc": datetime.now(timezone.utc).isoformat()}
        _write_once(self.directory / "started.json", dict(self.header, status="STARTED"))

    def finish(self, *, executed=None, error=None) -> Path:
        commands = executed.commands if executed is not None else error.commands
        record = dict(self.header, finished_utc=datetime.now(timezone.utc).isoformat(),
                      status=executed.result.status if executed is not None else "ERROR",
                      run_action_invocation_id=executed.invocation_id if executed is not None else None,
                      commands=[asdict(command) for command in commands],
                      result=executed.result.to_mapping() if executed is not None else None)
        if error is not None:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            cause = error.__cause__
            if cause is not None:
                record["error"]["cause"] = {"type": type(cause).__name__, "message": str(cause)}
        _write_once(self.result_path, record)
        return self.result_path
