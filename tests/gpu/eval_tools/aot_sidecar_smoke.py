"""Real-GPU rocJITsu AOT replay smoke for a running sidecar.

The host runner must expose its Unix socket and dedicated artifact mount.  The
two workspaces are expected to contain validated ``capsule.json`` fixtures: a
clean Triton store and a deliberately racy FlyDSL LDS dispatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval_tools.contracts import ExecutionStatus, FindingStatus
from src.eval_tools.factory import create_default_manager, task_artifact_root


def _evaluate(workspace: Path, language: str):
    workspace = workspace.resolve(strict=True)
    capsule = workspace / "capsule.json"
    if not capsule.is_file():
        raise FileNotFoundError(capsule)
    task_config = {
        "task_type": f"{language}2{language}",
        "source_file_path": ["capsule.json"],
        "target_kernel_functions": [language],
        "evaluation_profile": {
            "language": language,
            "artifact_kind": "python_jit",
            "framework": language,
            "adapter": f"{language}_aot",
            "source_available": True,
            "submission_paths": ["capsule.json"],
        },
    }
    config = {
        "evaluation_tools": {
            "policy": "advisory",
            "positive_control": "required",
            "enabled": ["rocjitsu"],
            "tools": {
                "rocjitsu": {
                    "timeout_s": 180,
                    "options": {"capsule": "capsule.json"},
                }
            },
        }
    }
    return create_default_manager().evaluate(
        workspace=workspace,
        task_config=task_config,
        config=config,
        gpu_arch="gfx950",
        artifact_root=task_artifact_root(workspace),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triton-workspace", type=Path, required=True)
    parser.add_argument("--flydsl-workspace", type=Path, required=True)
    args = parser.parse_args()

    triton = _evaluate(args.triton_workspace, "triton")
    flydsl = _evaluate(args.flydsl_workspace, "flydsl")
    triton_result = triton.evaluations[0].result
    flydsl_result = flydsl.evaluations[0].result
    assert triton_result is not None
    assert flydsl_result is not None
    assert triton_result.execution == ExecutionStatus.COMPLETED
    assert triton_result.finding == FindingStatus.CLEAN
    assert flydsl_result.execution == ExecutionStatus.COMPLETED
    assert flydsl_result.finding == FindingStatus.FOUND
    assert any(finding.kind == "lds-race" for finding in flydsl_result.findings)

    print(
        json.dumps(
            {
                "triton": triton.to_dict(),
                "flydsl": flydsl.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
