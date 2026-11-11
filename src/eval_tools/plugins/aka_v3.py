"""Framework-owned adapters for schema-v3 kernel tasks.

A schema-v3 ``task.yaml`` declares no evaluation-tool adapters. The framework
does not need any: every v3 workspace carries the generated runner
(``_aka_eval``), whose ``sanitize LANE`` action runs the role's implementation
on the task's own correctness cases under one lane's environment. A plugin on
this path only chooses the lane, sets the lane's environment, and judges the
process it ran. ``aka_role`` (``candidate`` by default) lets the manager rerun
a gate on the baseline after a candidate finding.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional

from ..contracts import KernelLanguage, ToolContext, ToolInvocation
from ..task_declarations import is_v3_task

SIDECAR_PYTHON = "/opt/venv/bin/python"
ADAPTER = "aka_v3"
REPORT_PREFIX = "ARENA_EVAL_RESULT="
ROLES = ("candidate", "baseline")
V3_LANGUAGES = frozenset({"triton", "gluon", "hip", "flydsl"})
NO_CACHING = {
    "PYTORCH_NO_HIP_MEMORY_CACHING": "1",
    "AMD_PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
    "HSA_DISABLE_FRAGMENT_ALLOCATOR": "1",
}
# Signatures of a device memory fault surfacing through ROCr, HIP or PyTorch.
FAULT_RE = re.compile(
    r"Memory access fault by GPU|HSA_STATUS_ERROR_MEMORY_FAULT|Memory Fault Error|"
    r"illegal memory access|hipErrorIllegalAddress",
    re.I,
)


def is_v3(context: ToolContext) -> bool:
    return is_v3_task(context.task_config)


def role(context: ToolContext) -> str:
    value = str(context.options.get("aka_role", "candidate"))
    if value not in ROLES:
        raise ValueError(f"aka_role must be one of {ROLES}, not {value!r}")
    return value


def _contract(context: ToolContext) -> Mapping[str, Any]:
    return context.task_config.get("contract") or {}


def target_language(context: ToolContext) -> str:
    return str(_contract(context).get("target_language") or context.profile.language.value)


def baseline_language(context: ToolContext) -> str:
    return str(_contract(context).get("baseline_language") or "")


def role_language(context: ToolContext) -> str:
    return baseline_language(context) if role(context) == "baseline" else target_language(context)


def language_enum(value: str) -> KernelLanguage:
    return {item.value: item for item in KernelLanguage}.get(value, KernelLanguage.UNKNOWN)


def lane_invocation(
    tool: str,
    context: ToolContext,
    lane: str,
    *,
    env: Mapping[str, str] | None = None,
    timeout_s: int = 1800,
    metadata: Mapping[str, Any] | None = None,
) -> ToolInvocation:
    """Run ``python -m _aka_eval.evaluate ROLE sanitize LANE`` in the workspace."""

    command = (SIDECAR_PYTHON, "-m", "_aka_eval.evaluate", role(context), "sanitize", lane)
    environment = {
        **dict(context.env),
        # The lane process imports only the workspace runner and the candidate;
        # the sidecar's own framework tree stays off its import path.
        "PYTHONPATH": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        # The sidecar UID need not exist in passwd; PyTorch's getpass needs these.
        "USER": context.env.get("USER", "aka"),
        "LOGNAME": context.env.get("LOGNAME", "aka"),
        "AKA_EVAL_TOOL": tool,
        **dict(env or {}),
    }
    return ToolInvocation(
        tool=tool,
        command=command,
        cwd=context.workspace,
        env=environment,
        timeout_s=timeout_s,
        artifact_dir=context.artifact_dir,
        metadata={"adapter": ADAPTER, "lane": lane, "role": role(context), **dict(metadata or {})},
    )


def lane_report(stdout: str, *, lane: str) -> tuple[Optional[dict], Optional[str]]:
    """The lane's single ``ARENA_EVAL_RESULT`` report, or why it is unusable.

    The report must be the last line: the runner prints it after the role's
    code has run, so output the role prints cannot follow it.
    """

    lines = [line for line in stdout.splitlines() if line.strip()]
    reports = [line for line in lines if line.startswith(REPORT_PREFIX)]
    if not reports:
        return None, "lane_report_missing"
    if len(reports) > 1 or lines[-1] != reports[0]:
        return None, "lane_report_ambiguous"
    try:
        report = json.loads(reports[0][len(REPORT_PREFIX):])
    except json.JSONDecodeError:
        return None, "lane_report_invalid_json"
    if not isinstance(report, dict) or report.get("action") != "sanitize":
        return None, "lane_report_wrong_action"
    if (report.get("metadata") or {}).get("lane") != lane and report.get("status") == "PASS":
        return None, "lane_report_wrong_lane"
    return report, None


def fault_excerpt(text: str) -> Optional[str]:
    match = FAULT_RE.search(text)
    if not match:
        return None
    start = max(0, match.start() - 400)
    return text[start:match.end() + 800]


def artifact_subdir(context: ToolContext, name: str) -> Path:
    path = Path(context.artifact_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    return path
