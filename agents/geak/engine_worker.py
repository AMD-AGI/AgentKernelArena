"""Subprocess boundary around GEAK's actual Claude Workflow runtime."""
from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.geak.bridge import Bridge, write_json
from agents.geak_v4.workflow_runner import (
    DEFAULT_SETTINGS, _extract_workflow_return, _read_json, _valid_workflow_return,
    build_prompt, invoke_via_sdk,
)


def run(job_path: Path) -> int:
    bridge = Bridge(job_path)
    handoff = bridge.job["engine"]
    options = bridge.job["options"]
    result_path = bridge.root / "engine_result.json"
    runtime = {"requested_model": options.get("model")}
    error_code = "sdk_workflow_failed"
    try:
        script = Path(handoff["script_path"])
        args = handoff["args"]
        transcript = invoke_via_sdk(
            build_prompt(script, args), workflow_dir=script.parent,
            eval_dir=bridge.eval_dir, model=options.get("model"), effort=options["effort"],
            settings=json.dumps(DEFAULT_SETTINGS), cli_path=options["claude_cli_path"],
            timeout_seconds=bridge.remaining(), done_grace_seconds=min(30, bridge.remaining()),
            done_poll_seconds=0.25, quiet=True,
            runtime_metadata=runtime,
            require_workflow_result=True,
            runtime_metadata_path=bridge.root / "runtime_identity.json",
        )
        error_code = "missing_terminal_workflow_result"
        returned = _read_json(bridge.eval_dir / "workflow_return.json")
        if returned is None:
            returned = _extract_workflow_return(transcript, bridge.eval_dir)
        if not _valid_workflow_return(returned, bridge.eval_dir, require_pinned_patch=True):
            raise RuntimeError("GEAK did not return a valid terminal Workflow result")
        write_json(bridge.eval_dir / "workflow_return.json", returned)
        # Keep bounded structural provenance; don't save model transcripts or free text.
        status = returned["validation_status"]
        error_code = "invalid_terminal_status"
        if status not in {"accepted", "flagged", "author_failed", "no_baseline"}:
            raise RuntimeError("GEAK returned an unknown terminal status")
        if returned.get("budget_used") == 0 and runtime.get("workflow_agent_errors"):
            error_code = "workflow_agent_errors_before_search"
            raise RuntimeError("GEAK could not dispatch a search after child-agent errors")
        write_json(result_path, {"status": status,
                                "mode": args["mode"], "target_language": args["target_language"],
                                "workflow_completed": True, "runtime": runtime,
                                "rounds": returned.get("rounds"), "budget_used": returned.get("budget_used")})
        return 0 if status == "accepted" else 1
    except Exception as exc:
        error_code = next((code for code in ("oauth_session_expired", "oauth_refresh_failed",
                                             "authentication_failed")
                           if code in runtime.get("runtime_error_codes", [])), error_code)
        write_json(result_path, {"status": "FAILED", "error_type": type(exc).__name__,
                                "error_code": error_code, "workflow_completed": False, "runtime": runtime})
        return 1


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1])))
