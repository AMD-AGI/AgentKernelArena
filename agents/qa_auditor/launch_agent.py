# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""QA auditor: deterministic task checks, then a model review, then the report.

The framework has already run the task's initial validation (TaskSession) and
passes its captured evidence through ARENA_VALIDATION_CONTEXT. The auditor runs
its deterministic checks before any model sees the workspace, so a review
cannot influence them, and writes the itemized report the framework verifies.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from agents import register_agent
from .deterministic import run_deterministic
from .report import build_report, write_report
from .review import run_review


def _agent_config() -> dict:
    with Path(__file__).with_name("agent_config.yaml").open() as handle:
        return yaml.safe_load(handle) or {}


@register_agent("qa_auditor")
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str) -> str:
    logger = logging.getLogger(__name__)
    workspace_path = Path(workspace)
    task_id = eval_config.get("_task_id") or "unknown"
    request_id = eval_config.get("_task_validation_request_id")
    errors: list[str] = []
    findings: list[dict] = []
    evidence: dict = {}
    package_sha = None
    target_language = None
    state = None
    try:
        context_path = eval_config.get("_task_validation_context") or os.environ.get("ARENA_VALIDATION_CONTEXT")
        if not context_path:
            raise RuntimeError("The auditor needs the framework's validation context")
        state = Path(context_path).parent
        context = json.loads(Path(context_path).read_text(encoding="utf-8"))
        if context.get("task_id") != task_id:
            raise RuntimeError("Validation context belongs to another task")
        target_language = (context.get("task_config", {}).get("contract") or {}).get("target_language")
        from src.task_materialization import load_materialization_record

        package_sha = load_materialization_record(workspace_path)["task_package_sha256"]
        config = _agent_config()
        review_config = dict(config.get("review") or {})
        run_agent = eval_config.get("agent") or {}
        for key in ("backend", "model", "effort", "timeout_seconds"):
            if key in run_agent:
                review_config[key] = run_agent[key]
        deterministic, evidence = run_deterministic(
            workspace_path, context, state,
            controls_timeout_s=int(run_agent.get("controls_timeout_seconds", 7200)))
        findings = [item.to_mapping() for item in deterministic]
        review = run_review(workspace=workspace_path, state=state, task_id=task_id, context=context,
                            deterministic=findings, controls=evidence.get("controls"),
                            review_config=review_config, eval_config=eval_config, logger=logger)
        findings += [item.to_mapping() for item in review]
    except Exception as exc:
        logger.exception("QA audit failed")
        errors.append(f"{type(exc).__name__}: {exc}")
    report = build_report(task_id=task_id, findings=findings, task_package_sha256=package_sha,
                          target_language=target_language, request_id=request_id, evidence=evidence,
                          framework_errors=errors)
    write_report(workspace_path, report, state=state)
    logger.info("QA audit %s: %s (%d blocking, %d warnings)", task_id, report["status"],
                report["blocking"], report["warnings"])
    return report["status"]
