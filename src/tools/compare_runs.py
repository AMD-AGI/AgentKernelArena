# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Standalone script to compare two AgentKernelArena runs.

Usage:
    python3 src/tools/compare_runs.py run1_path run2_path
    python3 src/tools/compare_runs.py workspace_MI300_cursor/run_20260714_120000_baseline workspace_MI300_cursor/run_20260714_140000_treatment
"""

import json
import argparse
import hashlib
import math
import os
import re
import stat
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

try:
    from src import postprocessing
    from src.score import resolve_speedup_ratio, task_result_scoring
except (ModuleNotFoundError, ImportError):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src import postprocessing
    from src.score import resolve_speedup_ratio, task_result_scoring


_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMPARISON_SCHEMAS = {
    "aka.apex-vs-codex-comparison-contract/v1",
    "aka.apex-vs-codex-comparison-contract/v2",
    "aka.apex-vs-codex-comparison-contract/v3",
}
_CANDIDATE_PERSISTENCE_POLICY = "structured_agent_turn_checkpoint_v2"
_BOUNDARY_QUIESCENCE_POLICY = "sigstop_process_group_snapshot_v1"
_OBJECTIVE_POLICY = "aka.task-package-objective-and-protected-harness/v1"
_PROMPT_POLICY = "aka.shared-objective-backend-native-context-receipted/v1"
_CODEX_IDENTITY_FIELDS = (
    "attempt_timeout_seconds",
    "backend",
    "codex_binary_sha256",
    "codex_version",
    "effort",
    "inner_max_iterations",
    "isolation",
    "max_turns",
    "model",
    "permission_mode",
    "structured_stream_output_limit_bytes",
    "turn_policy",
)


def _read_regular_file_no_follow(path: Path, *, immutable: bool = False) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"unsafe report evidence file: {path}") from error
    try:
        opened = os.fstat(descriptor)
        lexical = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (immutable and opened.st_mode & 0o222)
            or opened.st_dev != lexical.st_dev
            or opened.st_ino != lexical.st_ino
        ):
            raise ValueError(f"unsafe report evidence file: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _formal_manifest_context(
    run_path: Path, manifest: Dict[str, Any], manifest_bytes: bytes
) -> Dict[str, Any]:
    configuration = manifest.get("configuration")
    tasks = configuration.get("tasks") if isinstance(configuration, dict) else None
    comparison = manifest.get("comparison_contract")
    comparison_sha256 = manifest.get("comparison_contract_sha256")
    agent = manifest.get("agent")
    agent_template = agent.get("template") if isinstance(agent, dict) else None
    comparison_codex = comparison.get("codex") if isinstance(comparison, dict) else None
    if (
        not isinstance(tasks, list)
        or not tasks
        or not isinstance(comparison, dict)
        or comparison.get("schema") not in _COMPARISON_SCHEMAS
        or (
            comparison.get("schema")
            == "aka.apex-vs-codex-comparison-contract/v2"
            and comparison.get("candidate_persistence_policy_id")
            != _CANDIDATE_PERSISTENCE_POLICY
        )
        or (
            comparison.get("schema")
            == "aka.apex-vs-codex-comparison-contract/v3"
            and (
                comparison.get("candidate_persistence_policy_id")
                != _CANDIDATE_PERSISTENCE_POLICY
                or comparison.get("boundary_quiescence_policy_id")
                != _BOUNDARY_QUIESCENCE_POLICY
                or not isinstance(comparison_codex, dict)
                or comparison_codex.get("boundary_quiescence_policy_id")
                != _BOUNDARY_QUIESCENCE_POLICY
                or not isinstance(agent, dict)
                or agent.get("boundary_quiescence_policy_id")
                != _BOUNDARY_QUIESCENCE_POLICY
            )
        )
        or (
            comparison.get("schema")
            == "aka.apex-vs-codex-comparison-contract/v1"
            and "candidate_persistence_policy_id" in comparison
        )
        or comparison.get("objective_policy_id") != _OBJECTIVE_POLICY
        or comparison.get("prompt_policy_id") != _PROMPT_POLICY
        or comparison.get("tasks") != tasks
        or not isinstance(comparison_codex, dict)
        or any(
            field not in comparison_codex
            or not isinstance(agent, dict)
            or agent.get(field) != comparison_codex[field]
            for field in _CODEX_IDENTITY_FIELDS
        )
        or any(
            not isinstance(agent, dict) or agent.get(field) != value
            for field, value in comparison_codex.items()
        )
        or not isinstance(comparison_sha256, str)
        or not _SHA256.fullmatch(comparison_sha256)
        or _sha256_bytes(_canonical_json(comparison).encode()) != comparison_sha256
        or agent_template not in {"apex", "codex"}
    ):
        raise ValueError("formal campaign comparison/cohort/agent binding is invalid")

    task_names = []
    task_entries = {}
    for expected_index, task in enumerate(tasks, 1):
        if (
            not isinstance(task, dict)
            or task.get("task_index") != expected_index
            or not isinstance(task.get("task_name"), str)
            or not task["task_name"]
        ):
            raise ValueError("formal campaign cohort is malformed")
        task_names.append(task["task_name"])
        task_entries[task["task_name"]] = task
    if len(task_names) != len(set(task_names)):
        raise ValueError("formal campaign cohort contains duplicates")

    metadata = postprocessing._extract_run_metadata(run_path)
    if metadata.get("agent") != agent_template:
        raise ValueError("formal run path agent differs from campaign manifest agent")
    return {
        "task_names": task_names,
        "task_entries": task_entries,
        "manifest": manifest,
        "campaign_manifest_sha256": _sha256_bytes(manifest_bytes),
        "comparison_contract_sha256": comparison_sha256,
        "ordered_cohort_sha256": _sha256_bytes(_canonical_json(tasks).encode()),
        "agent_template": agent_template,
        "run_metadata": metadata,
    }


def _formal_success_projection(
    run_path: Path, task_name: str, workspace: Path, formal: Dict[str, Any]
) -> tuple[Dict[str, Any], str]:
    try:
        result, lineage = postprocessing._validate_canonical_lineage(
            run_directory=run_path,
            task_name=task_name,
            canonical=workspace,
            formal=formal,
        )
        pass_compilation = result.get("pass_compilation")
        pass_correctness = result.get("pass_correctness")
        summary = result.get("optimization_summary", "") or ""
        if (
            pass_compilation is not True
            or pass_correctness is not True
            or not isinstance(summary, str)
        ):
            raise ValueError("canonical task result has invalid success fields")
        speedup = resolve_speedup_ratio(
            speedup_ratio=result.get("speedup_ratio", 0.0),
            base_execution_time=result.get("base_execution_time", 0.0),
            best_optimized_execution_time=result.get(
                "best_optimized_execution_time", 0.0
            ),
        )
        if not math.isfinite(speedup) or speedup <= 0:
            raise ValueError("canonical task result has invalid speedup")
        score = task_result_scoring(str(workspace))
    except Exception as error:
        raise ValueError(
            f"formal canonical evidence is invalid for {task_name}: {error}"
        ) from error
    return {
        "task_name": task_name,
        "score": score,
        "pass_compilation": True,
        "pass_correctness": True,
        "speedup_ratio": speedup,
    }, lineage["canonical_workspace_manifest_sha256"]


def _formal_failure_projection(
    run_path: Path, task_name: str, formal: Dict[str, Any]
) -> tuple[Dict[str, Any], Dict[str, Any], bool]:
    evidence_parent = run_path / ".campaign_attempts" / task_name.replace("/", "_")
    if evidence_parent.exists():
        postprocessing._require_regular_directory_chain(
            evidence_parent, run_path, "failed task campaign"
        )
    failed_directory = run_path / ".parallel" / "failed"
    if failed_directory.exists():
        postprocessing._require_regular_directory_chain(
            failed_directory, run_path, "failed task descriptors"
        )
    failure = postprocessing._validated_failure_binding(
        run_path, task_name, formal
    )
    task = {
        "task_name": task_name,
        "score": 0.0,
        "pass_compilation": False,
        "pass_correctness": False,
        "speedup_ratio": 0.0,
    }
    report_entry = {
        "task_name": task_name,
        "reason_codes": failure["reason_codes"],
        "campaign_evidence_path": failure["campaign_evidence_path"],
        "campaign_evidence_sha256": failure["campaign_evidence_sha256"],
    }
    return task, report_entry, failure["terminal_binding_verified"] is True


def _recompute_formal_report(
    run_path: Path, formal: Dict[str, Any]
) -> Dict[str, Any]:
    task_names = formal["task_names"]
    workspace_map = postprocessing._formal_workspace_map(run_path, task_names)
    task_details = []
    failed_tasks = []
    canonical_manifests = {}
    terminal_task_count = 0

    for task_name in task_names:
        workspace = workspace_map.get(task_name)
        if workspace is not None:
            detail, manifest_sha256 = _formal_success_projection(
                run_path, task_name, workspace, formal
            )
            canonical_manifests[task_name] = manifest_sha256
            terminal_task_count += 1
        else:
            detail, failed, terminal = _formal_failure_projection(
                run_path, task_name, formal
            )
            failed_tasks.append(failed)
            terminal_task_count += int(terminal)
        task_details.append(detail)

    total_tasks = len(task_names)
    total_score = sum(task["score"] for task in task_details)
    compilation_count = sum(task["pass_compilation"] for task in task_details)
    correctness_count = sum(task["pass_correctness"] for task in task_details)
    speedups = [
        task["speedup_ratio"]
        for task in task_details
        if task["pass_compilation"]
        and task["pass_correctness"]
        and task["speedup_ratio"] > 0
    ]
    speedup_gt_1_count = sum(value > 1.0 for value in speedups)
    reason_counts: Dict[str, int] = defaultdict(int)
    for failure in failed_tasks:
        for reason in failure["reason_codes"]:
            reason_counts[reason] += 1
    completed = bool(
        terminal_task_count == total_tasks
        and postprocessing._formal_queue_has_no_unfinished_work(run_path)
    )
    overall = {
        "total_tasks": total_tasks,
        "total_score": total_score,
        "average_score": total_score / total_tasks,
        "compilation_pass_count": compilation_count,
        "compilation_pass_rate": compilation_count / total_tasks * 100,
        "correctness_pass_count": correctness_count,
        "correctness_pass_rate": correctness_count / total_tasks * 100,
        "speedup_gt_1_count": speedup_gt_1_count,
        "speedup_gt_1_rate": speedup_gt_1_count / total_tasks * 100,
        **postprocessing._compute_speedup_stats(speedups),
        "valid_speedup_count": len(speedups),
        "speedup_population": "canonical_compilation_and_correctness_successes_only",
        "formal_campaign": True,
        "canonical_success_count": len(canonical_manifests),
        "failed_task_count": len(failed_tasks),
        "failure_reason_counts": dict(sorted(reason_counts.items())),
        "campaign_manifest_sha256": formal["campaign_manifest_sha256"],
        "comparison_contract_sha256": formal["comparison_contract_sha256"],
        "ordered_cohort_sha256": formal["ordered_cohort_sha256"],
        "terminal_task_count": terminal_task_count,
        "formal_completion_verified": completed,
    }
    evidence = {
        "schema": "aka.formal-report-evidence/v1",
        "campaign_manifest_sha256": formal["campaign_manifest_sha256"],
        "comparison_contract_sha256": formal["comparison_contract_sha256"],
        "ordered_cohort_sha256": formal["ordered_cohort_sha256"],
        "completion_verified": completed,
        "terminal_task_count": terminal_task_count,
        "canonical_workspace_manifests": canonical_manifests,
    }
    metadata = formal["run_metadata"]
    return {
        "run_timestamp": metadata["timestamp"],
        "agent": formal["agent_template"],
        "target_gpu": metadata["target_gpu"],
        "overall": overall,
        "task_types": postprocessing._aggregate_by_task_type(task_details),
        "formal_evidence": evidence,
        "failed_tasks": failed_tasks,
    }


def _formal_report_contract(run_path: Path, data: Dict[str, Any]) -> Dict[str, str]:
    run_metadata = run_path.lstat()
    if stat.S_ISLNK(run_metadata.st_mode) or not stat.S_ISDIR(run_metadata.st_mode):
        raise ValueError(f"formal run directory is unsafe: {run_path}")
    reports = run_path / "reports"
    reports_metadata = reports.lstat()
    if stat.S_ISLNK(reports_metadata.st_mode) or not stat.S_ISDIR(reports_metadata.st_mode):
        raise ValueError(f"formal reports directory is unsafe: {reports}")
    if reports.resolve(strict=True).parent != run_path.resolve(strict=True):
        raise ValueError("formal reports directory escapes its run")

    manifest_path = run_path / "campaign_manifest.yaml"
    manifest_bytes = _read_regular_file_no_follow(manifest_path, immutable=True)
    manifest = yaml.safe_load(manifest_bytes.decode("utf-8")) or {}
    if not isinstance(manifest, dict) or manifest.get("schema") != "aka.matched-campaign/v1":
        raise ValueError("formal campaign manifest schema is invalid")
    formal = _formal_manifest_context(run_path, manifest, manifest_bytes)
    expected = _recompute_formal_report(run_path, formal)
    if _canonical_json(data) != _canonical_json(expected):
        raise ValueError("formal report does not match recomputed sealed evidence")
    if expected["overall"]["formal_completion_verified"] is not True:
        raise ValueError("formal campaign sealed evidence is not terminal")
    return {
        "campaign_manifest_sha256": formal["campaign_manifest_sha256"],
        "comparison_contract_sha256": formal["comparison_contract_sha256"],
        "ordered_cohort_sha256": formal["ordered_cohort_sha256"],
        "agent_template": formal["agent_template"],
        "resolved_run_path": str(run_path.resolve(strict=True)),
    }


def load_run_data(run_path: Path) -> Dict[str, Any]:
    """
    Load task_type_breakdown.json from a run directory.
    
    Args:
        run_path: Path to run directory (e.g., workspace_MI300_cursor/run_20260714_120000_baseline)
    
    Returns:
        Dictionary containing run data from JSON file
    
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON file is invalid
    """
    json_path = run_path / "reports" / "task_type_breakdown.json"
    
    raw = _read_regular_file_no_follow(json_path)
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"run report is not a JSON object: {json_path}")
    manifest_path = run_path / "campaign_manifest.yaml"
    try:
        manifest_path.lstat()
    except FileNotFoundError:
        manifest_present = False
    else:
        manifest_present = True
    report_declares_formal = data.get("overall", {}).get("formal_campaign") is True
    if manifest_present != report_declares_formal:
        raise ValueError("run report formal status differs from campaign manifest")
    if report_declares_formal:
        # Re-open under the immutable policy so a writable or swapped report
        # cannot opt itself into formal comparison.
        immutable_raw = _read_regular_file_no_follow(json_path, immutable=True)
        if immutable_raw != raw:
            raise ValueError("formal report changed while being loaded")
        data["_formal_contract"] = _formal_report_contract(run_path, data)
        if _read_regular_file_no_follow(json_path, immutable=True) != immutable_raw:
            raise ValueError("formal report changed during evidence validation")
    return data


def _validate_comparable_formal_runs(
    run1_data: Dict[str, Any], run2_data: Dict[str, Any]
) -> None:
    formal1 = run1_data.get("overall", {}).get("formal_campaign") is True
    formal2 = run2_data.get("overall", {}).get("formal_campaign") is True
    if not (formal1 or formal2):
        return
    if not (formal1 and formal2):
        raise ValueError("formal comparison requires two formal completed runs")
    contract1 = run1_data.get("_formal_contract")
    contract2 = run2_data.get("_formal_contract")
    if not isinstance(contract1, dict) or not isinstance(contract2, dict):
        raise ValueError("formal comparison lacks validated run contracts")
    if (
        contract1.get("comparison_contract_sha256")
        != contract2.get("comparison_contract_sha256")
    ):
        raise ValueError("formal run comparison contracts differ")
    if contract1.get("ordered_cohort_sha256") != contract2.get("ordered_cohort_sha256"):
        raise ValueError("formal run ordered cohorts differ")
    if contract1.get("resolved_run_path") == contract2.get("resolved_run_path"):
        raise ValueError("formal comparison cannot compare a run with itself")
    arms = {contract1.get("agent_template"), contract2.get("agent_template")}
    if arms != {"apex", "codex"}:
        raise ValueError("formal comparison requires exactly one apex and one codex arm")


def format_difference(value1: float, value2: float, is_percentage: bool = False) -> str:
    """
    Format the difference between two values.
    
    Args:
        value1: First value (baseline)
        value2: Second value (comparison)
        is_percentage: If True, format as percentage change
    
    Returns:
        Formatted string showing difference
    """
    diff = value2 - value1
    if is_percentage:
        if value1 == 0:
            return f"{diff:+.1f}pp" if diff != 0 else "0.0pp"
        pct_change = (diff / value1 * 100) if value1 != 0 else 0
        return f"{diff:+.1f}pp ({pct_change:+.1f}%)"
    else:
        pct_change = (diff / value1 * 100) if value1 != 0 else 0
        return f"{diff:+.3f} ({pct_change:+.1f}%)"


def compare_overall(run1_data: Dict[str, Any], run2_data: Dict[str, Any]) -> list:
    """
    Compare overall statistics between two runs.
    
    Returns:
        List of formatted comparison lines
    """
    overall1 = run1_data.get('overall', {})
    overall2 = run2_data.get('overall', {})
    formal = (
        overall1.get("formal_campaign") is True
        and overall2.get("formal_campaign") is True
    )
    speedup_population = "Canonical-success-only " if formal else ""
    
    lines = [
        "=" * 80,
        "OVERALL STATISTICS COMPARISON",
        "=" * 80,
        f"Run 1: {run1_data.get('run_timestamp', 'unknown')} ({run1_data.get('agent', 'unknown')})",
        f"Run 2: {run2_data.get('run_timestamp', 'unknown')} ({run2_data.get('agent', 'unknown')})",
        "=" * 80,
        "",
        f"{'Metric':<40} {'Run 1':<15} {'Run 2':<15} {'Difference':<20}",
        "-" * 80,
    ]
    
    metrics = []
    if formal:
        metrics.extend([
            ('Canonical Success Tasks', 'canonical_success_count', False),
            ('Failed Tasks', 'failed_task_count', False),
            ('Canonical-success-only Speedup Count', 'valid_speedup_count', False),
        ])
    metrics.extend([
        ('Total Tasks', 'total_tasks', False),
        ('Total Score', 'total_score', False),
        ('Average Score', 'average_score', False),
        ('Compilation Pass Rate', 'compilation_pass_rate', True),
        ('Correctness Pass Rate', 'correctness_pass_rate', True),
        ('Speedup > 1.0 Rate', 'speedup_gt_1_rate', True),
        (f'{speedup_population}Average Speedup', 'average_speedup', False),
        (f'{speedup_population}Median Speedup', 'median_speedup', False),
        (f'{speedup_population}Std Dev Speedup', 'std_dev_speedup', False),
        (f'{speedup_population}P25 Speedup', 'p25_speedup', False),
        (f'{speedup_population}P75 Speedup', 'p75_speedup', False),
        (f'{speedup_population}P90 Speedup', 'p90_speedup', False),
    ])
    
    for label, key, is_percentage in metrics:
        val1 = overall1.get(key, 0.0)
        val2 = overall2.get(key, 0.0)
        
        if is_percentage:
            fmt1 = f"{val1:.1f}%"
            fmt2 = f"{val2:.1f}%"
        elif key in {
            'total_tasks',
            'canonical_success_count',
            'failed_task_count',
            'valid_speedup_count',
        }:
            fmt1 = f"{int(val1)}"
            fmt2 = f"{int(val2)}"
        elif key == 'total_score':
            fmt1 = f"{val1:.2f}"
            fmt2 = f"{val2:.2f}"
        else:
            fmt1 = f"{val1:.3f}"
            fmt2 = f"{val2:.3f}"
        
        diff_str = format_difference(val1, val2, is_percentage)
        
        # Determine if improvement (green) or regression (red) - for display purposes
        if key in ['average_score', 'compilation_pass_rate', 'correctness_pass_rate',
                   'canonical_success_count', 'valid_speedup_count',
                   'speedup_gt_1_rate', 'average_speedup', 'median_speedup', 
                   'p25_speedup', 'p75_speedup', 'p90_speedup']:
            if val2 > val1:
                indicator = "↑"
            elif val2 < val1:
                indicator = "↓"
            else:
                indicator = "="
        elif key == 'std_dev_speedup':
            # Lower std dev is better (more consistent), so reverse the logic
            if val2 < val1:
                indicator = "↑"
            elif val2 > val1:
                indicator = "↓"
            else:
                indicator = "="
        elif key == 'failed_task_count':
            if val2 < val1:
                indicator = "↑"
            elif val2 > val1:
                indicator = "↓"
            else:
                indicator = "="
        else:
            indicator = ""
        
        lines.append(f"{label:<40} {fmt1:<15} {fmt2:<15} {diff_str:<20} {indicator}")
    
    lines.append("")
    return lines


def compare_task_types(run1_data: Dict[str, Any], run2_data: Dict[str, Any]) -> list:
    """
    Compare task type breakdowns between two runs.
    
    Returns:
        List of formatted comparison lines
    """
    types1 = run1_data.get('task_types', {})
    types2 = run2_data.get('task_types', {})
    formal = (
        run1_data.get("overall", {}).get("formal_campaign") is True
        and run2_data.get("overall", {}).get("formal_campaign") is True
    )
    speedup_population = "Canonical-success-only " if formal else ""
    
    # Get all unique task types from both runs
    all_types = set(types1.keys()) | set(types2.keys())
    
    if not all_types:
        return ["No task type data available for comparison."]
    
    lines = [
        "=" * 80,
        "TASK TYPE BREAKDOWN COMPARISON",
        "=" * 80,
        "",
    ]
    
    for task_type in sorted(all_types):
        stats1 = types1.get(task_type, {})
        stats2 = types2.get(task_type, {})
        
        count1 = stats1.get('count', 0)
        count2 = stats2.get('count', 0)
        
        lines.append(f"{task_type.upper()} ({count1} tasks → {count2} tasks):")
        lines.append("-" * 80)
        
        if count1 == 0 and count2 == 0:
            lines.append("  No tasks in either run")
            lines.append("")
            continue
        
        # Compare key metrics
        metrics = [
            (f'{speedup_population}Average Speedup', 'average_speedup', False),
            (f'{speedup_population}Median Speedup', 'median_speedup', False),
            (f'{speedup_population}Std Dev Speedup', 'std_dev_speedup', False),
            (f'{speedup_population}P25 Speedup', 'p25_speedup', False),
            (f'{speedup_population}P75 Speedup', 'p75_speedup', False),
            (f'{speedup_population}P90 Speedup', 'p90_speedup', False),
            ('Compilation Pass Rate', 'compilation_pass_rate', True),
            ('Correctness Pass Rate', 'correctness_pass_rate', True),
            ('Speedup > 1.0 Rate', 'speedup_gt_1_rate', True),
            ('Average Score', 'average_score', False),
        ]
        
        for label, key, is_percentage in metrics:
            val1 = stats1.get(key, 0.0)
            val2 = stats2.get(key, 0.0)

            # Format both values first, then override with N/A as needed
            if is_percentage:
                fmt1 = f"{val1:.1f}%"
                fmt2 = f"{val2:.1f}%"
            elif key == 'average_score':
                fmt1 = f"{val1:.2f}"
                fmt2 = f"{val2:.2f}"
            else:
                fmt1 = f"{val1:.3f}"
                fmt2 = f"{val2:.3f}"

            if count1 == 0:
                fmt1 = "N/A"
                diff_str = "N/A (new)"
            elif count2 == 0:
                fmt2 = "N/A"
                diff_str = "N/A (removed)"
            else:
                diff_str = format_difference(val1, val2, is_percentage)
            
            if count1 > 0 and count2 > 0:
                # For std_dev_speedup, lower is better (more consistent)
                if key == 'std_dev_speedup':
                    if val2 < val1:
                        indicator = "↑ (improved)"
                    elif val2 > val1:
                        indicator = "↓ (regressed)"
                    else:
                        indicator = "= (same)"
                # For percentiles and other speedup metrics, higher is better
                elif key in ['p25_speedup', 'p75_speedup', 'p90_speedup', 'average_speedup', 'median_speedup']:
                    if val2 > val1:
                        indicator = "↑ (improved)"
                    elif val2 < val1:
                        indicator = "↓ (regressed)"
                    else:
                        indicator = "= (same)"
                else:
                    if val2 > val1:
                        indicator = "↑ (improved)"
                    elif val2 < val1:
                        indicator = "↓ (regressed)"
                    else:
                        indicator = "= (same)"
            else:
                indicator = ""
            
            if count1 == 0:
                lines.append(f"  {label:<35} {'N/A':<15} {fmt2:<15} {diff_str:<20} {indicator}")
            elif count2 == 0:
                lines.append(f"  {label:<35} {fmt1:<15} {'N/A':<15} {diff_str:<20} {indicator}")
            else:
                lines.append(f"  {label:<35} {fmt1:<15} {fmt2:<15} {diff_str:<20} {indicator}")
        
        lines.append("")
    
    return lines


def generate_comparison_report(run1_path: Path, run2_path: Path, output_path: Optional[Path] = None) -> str:
    """
    Generate a comparison report between two runs.
    
    Args:
        run1_path: Path to first run directory
        run2_path: Path to second run directory
        output_path: Optional path to save report (if None, auto-generates in comparisons/ directory)
    
    Returns:
        Comparison report as string
    """
    run1_data = load_run_data(run1_path)
    run2_data = load_run_data(run2_path)
    _validate_comparable_formal_runs(run1_data, run2_data)
    
    # Generate comparison report
    lines = [
        "=" * 80,
        "AgentKernelArena Run Comparison Report",
        "=" * 80,
        "",
    ]
    
    lines.extend(compare_overall(run1_data, run2_data))
    lines.extend(compare_task_types(run1_data, run2_data))
    
    lines.extend([
        "=" * 80,
        "Legend:",
        "  ↑ = Improvement (higher is better)",
        "  ↓ = Regression (lower is worse)",
        "  = = No change",
        "  pp = percentage points",
        "=" * 80,
    ])
    
    report = "\n".join(lines)
    
    # Determine output path
    if output_path is None:
        # Auto-generate path in comparisons/ directory at project root
        # Extract run directory names (e.g., "run_20260714_120000_baseline" from full path)
        run1_name = run1_path.name
        run2_name = run2_path.name
        
        # Keep generated comparisons at the project root even though this CLI
        # lives under src/tools/.
        project_root = Path(__file__).resolve().parents[2]
        
        comparisons_dir = project_root / "comparisons"
        comparisons_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename: comparison_report_{run1}_{run2}.txt
        filename = f"comparison_report_{run1_name}_{run2_name}.txt"
        output_path = comparisons_dir / filename
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"Comparison report written to: {output_path}")
    
    return report


def main():
    """Main entry point for comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare two AgentKernelArena runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two runs
  python3 src/tools/compare_runs.py workspace_MI300_cursor/run_20260714_120000_baseline workspace_MI300_cursor/run_20260714_140000_treatment
  
  # Compare and save to file
  python3 src/tools/compare_runs.py run1 run2 --output comparison_report.txt
        """
    )
    
    parser.add_argument(
        'run1',
        type=str,
        help='Path to baseline/first run directory (e.g., workspace_MI300_cursor/run_20260714_120000_baseline)'
    )
    
    parser.add_argument(
        'run2',
        type=str,
        help='Path to treatment/second run directory (e.g., workspace_MI300_cursor/run_20260714_140000_treatment)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Optional output file path for comparison report (if not specified, auto-generates in comparisons/ directory)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    run1_path = Path(args.run1).resolve()
    run2_path = Path(args.run2).resolve()
    
    # Validate paths exist
    if not run1_path.exists():
        print(f"Error: Run 1 directory does not exist: {run1_path}", file=sys.stderr)
        sys.exit(1)
    
    if not run2_path.exists():
        print(f"Error: Run 2 directory does not exist: {run2_path}", file=sys.stderr)
        sys.exit(1)
    
    # Generate and print comparison report
    output_path = Path(args.output).resolve() if args.output else None
    try:
        report = generate_comparison_report(run1_path, run2_path, output_path)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    
    # Print to stdout
    print(report)


if __name__ == "__main__":
    main()
