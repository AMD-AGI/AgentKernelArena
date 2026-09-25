"""Run Arena task-owned actions inside an Apex candidate copy."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_execution import run_action
from src.task_protocol import CaseManifest, RESULT_PREFIX, parse_command_result
from src.task_spec import TaskSpec


def run(job_path: Path, action: str, workspace: Path) -> None:
    job = json.loads(job_path.read_text())
    raw = job["context"]
    spec = TaskSpec.from_mapping(raw["task_config"], task_id=raw["task_id"])
    manifest = CaseManifest.from_result(parse_command_result(
        RESULT_PREFIX + json.dumps(raw["manifest"]), role="task", action="validate-task", returncode=0))
    # Check against the frozen caller input, including symbol-scoped harnesses.
    from dataclasses import replace
    snapshot = snapshot_workspace_harness(Path(job["source"]), task_spec=spec)
    snapshot = replace(snapshot, root=workspace)
    verify_workspace_harness(snapshot, discard_added=False)
    # Build/report outputs must not become undeclared edits in Apex's source
    # projection. Each check uses a fresh copy and rebuilds its prerequisites.
    from agents.apex.launch_agent import copy_workspace
    with tempfile.TemporaryDirectory(prefix="arena-apex-check-") as temporary:
        evaluation = Path(temporary) / "workspace"
        copy_workspace(workspace, evaluation)
        for phase in ("compile", "correctness", "performance"):
            result = run_action(spec, evaluation, role="candidate", action=phase,
                                phase="candidate_evaluation", manifest=manifest)
            verify_workspace_harness(replace(snapshot, root=evaluation), discard_added=False)
            if not result.result.passed:
                raise RuntimeError(f"Arena candidate {phase} failed")
            if phase == action:
                break
    verify_workspace_harness(snapshot, discard_added=False)
    print(RESULT_PREFIX + json.dumps(result.result.to_mapping(), allow_nan=False))


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2], Path.cwd())
