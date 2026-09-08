"""JSON-returning CLI tools available inside a generator session."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from .bundle import inspect_bundle
from .config import Config
from .materialize import check_contract, materialize_task
from .validation import check_task, validate_task


def dispatch(action: str, run_dir: Path, task_id: str, mode="source-check", validation_id=None):
    config = Config(**json.loads((run_dir / "config.json").read_text()))
    tasks = inspect_bundle(run_dir / "bundle", config.selections)
    selected = next((t for t in tasks if t.task_id == task_id), None)
    if action == "inspect_bundle":
        return {"ok": True, "tasks": [t.summary() for t in tasks]}
    if selected is None:
        raise ValueError(f"Unknown task: {task_id}")
    draft = run_dir / "drafts" / task_id
    artifacts = run_dir / "tool_runs" / task_id
    if action == "describe_task":
        return {"ok": True, **selected.summary(), "contract": selected.contract(),
                "editable": ["scripts/task_inputs.py"]}
    if action == "materialize_task":
        return {"ok": True, **materialize_task(selected, config, draft)}
    if action == "check_contract":
        return check_contract(selected, config, draft)
    if action in {"check_task", "validate_task"}:
        contract = check_contract(selected, config, draft)
        if not contract["ok"]:
            return contract
        if action == "check_task":
            return check_task(draft, artifacts / uuid.uuid4().hex, config, mode)
        return validate_task(draft, artifacts, config)
    if action == "read_validation":
        if not validation_id or any(c not in "0123456789abcdef" for c in validation_id):
            raise ValueError("Invalid validation ID")
        return json.loads((artifacts / validation_id / "evidence.json").read_text())
    raise ValueError(f"Unknown action: {action}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("action", choices=("inspect_bundle", "describe_task", "materialize_task",
                                          "check_contract", "check_task", "validate_task", "read_validation"))
    parser.add_argument("--mode", default="source-check")
    parser.add_argument("--validation-id")
    args = parser.parse_args(argv)
    try:
        result = dispatch(args.action, args.run_dir.resolve(), args.task_id, args.mode, args.validation_id)
    except Exception as exc:
        result = {"ok": False, "diagnostics": [{"code": getattr(exc, "code", "tool_error"), "message": str(exc)}]}
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
