"""Run safe/seeded-bug candidate cases through a selected real sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval_tools.contracts import ExecutionStatus, FindingStatus
from src.eval_tools.factory import create_default_manager, task_artifact_root


HARNESS = "/workspace/tests/gpu/eval_tools/candidate_harness.py"


def _case(workspace: Path, *, tool: str, language: str, mode: str, bug: bool):
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "case.txt").write_text(f"{tool} {language} bug={bug}\n", encoding="utf-8")
    profile = {
        "language": language,
        "artifact_kind": "python_jit" if language == "triton" else "source_aot",
        "framework": language,
        "instrumentation_control": "compiler_controlled" if language == "triton" else "recompile",
        "adapter": "triton_python_jit" if language == "triton" else "hip_source",
        "source_available": True,
        "submission_paths": ["case.txt"],
    }
    if tool == "hip_fpsan":
        profile["fpsan_ported"] = True
        profile["adapter"] = "hip_fpsan_manual"
    task_config = {
        "task_type": f"{language}2{language}",
        "source_file_path": ["case.txt"],
        "target_kernel_functions": [mode],
        "evaluation_profile": profile,
    }
    command = [
        "/opt/venv/bin/python",
        HARNESS,
        "--mode",
        mode,
        *(["--bug"] if bug else []),
    ]
    config = {
        "evaluation_tools": {
            "enabled": [tool],
            "policy": "advisory",
            "positive_control": "required",
            "tools": {tool: {"timeout_s": 180, "options": {"command": command}}},
        }
    }
    report = create_default_manager().evaluate(
        workspace=workspace,
        task_config=task_config,
        config=config,
        gpu_arch="gfx950",
        artifact_root=task_artifact_root(workspace),
    )
    result = report.evaluations[0].result
    assert result is not None
    assert result.execution == ExecutionStatus.COMPLETED
    assert result.finding == (FindingStatus.FOUND if bug else FindingStatus.CLEAN)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("triton_fpsan", "gpu_asan", "hip_fpsan"), required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    args = parser.parse_args()
    specs = {
        "triton_fpsan": [("triton_fpsan", "triton", "triton_fpsan")],
        "gpu_asan": [
            ("gpu_asan", "hip", "gpu_asan"),
            ("gpu_asan", "triton", "gpu_asan_triton"),
        ],
        "hip_fpsan": [("hip_fpsan", "hip", "hip_fpsan")],
    }[args.suite]
    output = {}
    for tool, language, mode in specs:
        for bug in (False, True):
            key = f"{language}-{'bug' if bug else 'safe'}"
            output[key] = _case(
                args.workspace_root / key,
                tool=tool,
                language=language,
                mode=mode,
                bug=bug,
            ).to_dict()
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
