# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import yaml
import logging
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
import statistics
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
try:
    from src.campaign import (
        CampaignError,
        _campaign_failure_reasons,
        _evaluation_eligibility_errors,
        _select_attempt,
        _run_config_contract,
        campaign_task_path_component,
    )
    from src.score import resolve_speedup_ratio, task_result_scoring
    from src.preprocessing import get_task_workspace_path
except ModuleNotFoundError:
    # Allow direct execution: `python src/postprocessing.py`
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.campaign import (
        CampaignError,
        _campaign_failure_reasons,
        _evaluation_eligibility_errors,
        _select_attempt,
        _run_config_contract,
        campaign_task_path_component,
    )
    from src.score import resolve_speedup_ratio, task_result_scoring
    from src.preprocessing import get_task_workspace_path


_FORMAL_PRIMARY_FAILURE_REASONS = frozenset(
    {
        "descriptor_gpu_affinity_mismatch",
        "formal_task_not_canonical",
    }
)

_ATTEMPT_CAMPAIGN_BINDING_SCHEMA = "aka.attempt-campaign-binding/v1"
_ATTEMPT_CAMPAIGN_BINDING_KEYS = frozenset(
    {
        "schema",
        "formal_execution_sha256",
        "campaign_manifest_path",
        "campaign_manifest_sha256",
        "comparison_contract_sha256",
        "backend_runtime_closure_sha256",
        "task_package_manifest_sha256",
        "task_config_sha256",
        "task_name",
        "task_index",
        "total_tasks",
        "attempt_index",
        "attempt_count",
        "assigned_host_gpu_id",
    }
)
_FORMAL_RECEIPT_SCHEMAS = {
    "apex": "agentkernelarena.apex-attempt-receipt/v5",
    "codex": "agentkernelarena.codex-attempt-receipt/v4",
}


def _build_general_report_lines(
    aggregate_result: Dict[str, Any], 
    run_metadata: Optional[Dict[str, str]] = None,
    task_type_breakdown: Optional[Dict[str, Dict[str, Any]]] = None
) -> List[str]:
    """Build report lines shared by logger output and fallback txt output."""
    formal_campaign = aggregate_result.get("formal_campaign") is True
    speedup_population = (
        "Canonical-success" if formal_campaign else "Correct-and-compiled"
    )
    lines = [
        "=" * 80,
        "AgentKernelArena Task Results Report",
        "=" * 80,
    ]
    
    # Add run metadata if available
    if run_metadata:
        lines.append(f"Run: {run_metadata.get('timestamp', 'unknown')}")
        lines.append(f"Agent: {run_metadata.get('agent', 'unknown')}")
        lines.append(f"Target GPU: {run_metadata.get('target_gpu', 'unknown')}")
        lines.append("=" * 80)
    
    lines.extend([
        "OVERALL STATISTICS:",
        f"  Total Tasks:           {aggregate_result['total_tasks']}",
        f"  Total Score:           {aggregate_result['total_score']:.2f}",
        f"  Average Score:         {aggregate_result['average_score']:.2f}",
        "Compilation:",
        f"  Pass Count:            {aggregate_result['compilation_pass_count']}/{aggregate_result['total_tasks']}",
        f"  Pass Rate:             {aggregate_result['compilation_pass_rate']:.1f}%",
        "Correctness:",
        f"  Pass Count:            {aggregate_result['correctness_pass_count']}/{aggregate_result['total_tasks']}",
        f"  Pass Rate:             {aggregate_result['correctness_pass_rate']:.1f}%",
        "Performance:",
        f"  Speedup > 1.0 Count:   {aggregate_result['speedup_gt_1_count']}/{aggregate_result['total_tasks']}",
        f"  Speedup > 1.0 Rate:    {aggregate_result['speedup_gt_1_rate']:.1f}%",
        f"  {speedup_population} Average Speedup: {aggregate_result['average_speedup']:.2f}x",
        f"  {speedup_population} Median Speedup:  {aggregate_result.get('median_speedup', 0.0):.2f}x",
        f"  {speedup_population} Std Dev Speedup: {aggregate_result.get('std_dev_speedup', 0.0):.2f}x",
        f"  {speedup_population} Observed P25/P75/P90: {aggregate_result.get('p25_speedup', 0.0):.2f}x / {aggregate_result.get('p75_speedup', 0.0):.2f}x / {aggregate_result.get('p90_speedup', 0.0):.2f}x",
        f"  {speedup_population} Speedup Count:   {aggregate_result['valid_speedup_count']}",
    ])

    if aggregate_result.get("formal_campaign"):
        lines.extend([
            "Formal Campaign Cohort:",
            f"  Campaign Manifest:    {aggregate_result['campaign_manifest_sha256']}",
            f"  Comparison Contract:  {aggregate_result['comparison_contract_sha256']}",
            f"  Ordered Cohort:        {aggregate_result['ordered_cohort_sha256']}",
            f"  Manifest Tasks:        {aggregate_result['total_tasks']}",
            f"  Canonical Successes:   {aggregate_result['canonical_success_count']}/{aggregate_result['total_tasks']}",
            f"  Failed Tasks:          {aggregate_result['failed_task_count']}/{aggregate_result['total_tasks']}",
            f"  Terminal Evidence:     {aggregate_result['terminal_task_count']}/{aggregate_result['total_tasks']}",
            f"  Completion Verified:   {aggregate_result['formal_completion_verified']}",
        ])
        failure_counts = aggregate_result.get("failure_reason_counts", {})
        if failure_counts:
            lines.append("  Failure Reasons:")
            for reason, count in sorted(failure_counts.items()):
                lines.append(f"    {reason}: {count}")
    
    # Add task type breakdowns if available
    if task_type_breakdown:
        lines.append("")
        lines.append("TASK TYPE BREAKDOWN:")
        lines.append("")
        
        # Sort task types for consistent output
        sorted_types = sorted(task_type_breakdown.keys())
        for task_type in sorted_types:
            stats = task_type_breakdown[task_type]
            lines.append(f"  {task_type} ({stats['count']} tasks):")
            lines.append(f"    {speedup_population} Average Speedup: {stats['average_speedup']:.2f}x")
            lines.append(f"    {speedup_population} Median Speedup: {stats.get('median_speedup', 0.0):.2f}x")
            lines.append(f"    {speedup_population} Std Dev Speedup: {stats.get('std_dev_speedup', 0.0):.2f}x")
            lines.append(f"    {speedup_population} Observed P25/P75/P90: {stats.get('p25_speedup', 0.0):.2f}x / {stats.get('p75_speedup', 0.0):.2f}x / {stats.get('p90_speedup', 0.0):.2f}x")
            lines.append(f"    Compilation Pass:     {stats['compilation_pass_count']}/{stats['count']}")
            lines.append(f"    Compilation Pass Rate: {stats['compilation_pass_rate']:.1f}%")
            lines.append(f"    Correctness Pass:     {stats['correctness_pass_count']}/{stats['count']}")
            lines.append(f"    Correctness Pass Rate: {stats['correctness_pass_rate']:.1f}%")
            lines.append(f"    Speedup > 1.0:        {stats['speedup_gt_1_count']}/{stats['count']} ({stats['speedup_gt_1_rate']:.1f}%)")
            lines.append(f"    Average Score:        {stats['average_score']:.2f}")
            lines.append("")
    
    # Add total performance summary
    lines.extend([
        "TOTAL PERFORMANCE SUMMARY:",
        f"  {speedup_population} Average Speedup: {aggregate_result['average_speedup']:.2f}x",
        f"  Tasks with Speedup > 1.0: {aggregate_result['speedup_gt_1_count']}/{aggregate_result['total_tasks']} ({aggregate_result['speedup_gt_1_rate']:.1f}%)",
    ])
    
    # Find best and worst speedups
    speedups = []
    for task in aggregate_result.get('task_details', []):
        if task.get('pass_compilation') and task.get('pass_correctness'):
            speedup = task.get('speedup_ratio', 0.0)
            if speedup > 0:
                speedups.append((speedup, task.get('task_name', '')))
    
    if speedups:
        best_speedup, best_task = max(speedups, key=lambda x: x[0])
        worst_speedup, worst_task = min(speedups, key=lambda x: x[0])
        lines.append(f"  Best Speedup:            {best_speedup:.2f}x (task: {best_task})")
        lines.append(f"  Worst Speedup:           {worst_speedup:.2f}x (task: {worst_task})")
    
    lines.extend([
        "",
        "TASK DETAILS:",
        "-" * 80,
    ])

    for task in aggregate_result["task_details"]:
        status = (
            "FAILED"
            if task.get("campaign_status") == "failed"
            else (
                "PASS"
                if task["pass_correctness"]
                else ("PARTIAL" if task["pass_compilation"] else "FAIL")
            )
        )
        lines.append(
            f"{status:<8} {task['task_name']:<40} Score: {task['score']:>6.1f}  Speedup: {task['speedup_ratio']:.2f}x"
        )
        if task["error"]:
            lines.append(f"         Error: {task['error']}")
        if task.get("campaign_evidence_path"):
            lines.append(
                f"         Campaign evidence: {task['campaign_evidence_path']} "
                f"(sha256={task.get('campaign_evidence_sha256')})"
            )

    lines.append("=" * 80)
    return lines


def _get_run_directory(workspace_paths: List[str]) -> Path:
    """
    Extract run directory from workspace paths.
    
    Workspace paths are task directories like:
    workspace_MI300_cursor/run_20250115_143022/task_hip2hip_silu_20250115_143022/
    
    Returns the run directory: workspace_MI300_cursor/run_20250115_143022/
    """
    if not workspace_paths:
        raise ValueError("Cannot determine run directory: empty workspace_paths")
    
    # First workspace path is a task directory, its parent is the run directory
    first_workspace = Path(workspace_paths[0]).resolve()
    run_directory = first_workspace.parent
    
    # Validate that this looks like a run directory (contains task directories)
    if not run_directory.exists():
        raise ValueError(f"Run directory does not exist: {run_directory}")
    
    return run_directory


def _extract_run_metadata(run_directory: Path) -> Dict[str, str]:
    """
    Extract metadata from run directory structure.
    
    Returns dict with: timestamp, agent, target_gpu
    """
    # Extract timestamp from run directory name: run_20250115_143022 -> 20250115_143022
    run_dir_name = run_directory.name
    if run_dir_name.startswith("run_"):
        timestamp = run_dir_name[4:]  # Remove "run_" prefix
    else:
        timestamp = "unknown"
    
    # Extract agent and GPU from workspace directory name: workspace_MI300_cursor -> MI300, cursor
    workspace_dir = run_directory.parent
    workspace_name = workspace_dir.name
    parts = workspace_name.split("_")
    
    # Pattern: workspace_{GPU}_{agent}
    if len(parts) >= 3 and parts[0] == "workspace":
        target_gpu = parts[1]
        agent = "_".join(parts[2:])  # In case agent name has underscores
    else:
        target_gpu = "unknown"
        agent = "unknown"
    
    return {
        "timestamp": timestamp,
        "agent": agent,
        "target_gpu": target_gpu
    }


def _compute_speedup_stats(speedup_values: List[float]) -> Dict[str, float]:
    """
    Compute speedup statistics with bounded observed P25/P75/P90 quantiles.

    Args:
        speedup_values: List of valid speedup ratios (> 0).

    Returns:
        Dict with keys: average_speedup, median_speedup, std_dev_speedup,
        p25_speedup, p75_speedup, p90_speedup.
    """
    if not speedup_values:
        return {
            'average_speedup': 0.0,
            'median_speedup': 0.0,
            'std_dev_speedup': 0.0,
            'p25_speedup': 0.0,
            'p75_speedup': 0.0,
            'p90_speedup': 0.0,
        }

    average = sum(speedup_values) / len(speedup_values)

    try:
        median = statistics.median(speedup_values)
    except statistics.StatisticsError:
        median = 0.0

    try:
        std_dev = statistics.stdev(speedup_values) if len(speedup_values) > 1 else 0.0
    except statistics.StatisticsError:
        std_dev = 0.0

    ordered = sorted(speedup_values)

    def observed_quantile(fraction: float) -> float:
        # Nearest-rank selects an observed sample. Unlike the default exclusive
        # estimator in statistics.quantiles(), it cannot extrapolate beyond the
        # observed min/max when a formal campaign has only a few valid tasks.
        index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
        return ordered[index]

    p25 = observed_quantile(0.25)
    p75 = observed_quantile(0.75)
    p90 = observed_quantile(0.90)

    return {
        'average_speedup': average,
        'median_speedup': median,
        'std_dev_speedup': std_dev,
        'p25_speedup': p25,
        'p75_speedup': p75,
        'p90_speedup': p90,
    }


def _extract_task_type(task_name: str) -> str:
    """
    Extract task type from task name.
    
    Task names are like: hip2hip/silu, triton2triton/vllm/xxx, etc.
    Returns the first part before the first slash, or empty string if no slash.
    """
    if isinstance(task_name, str) and "/" in task_name:
        return task_name.split("/", 1)[0]
    return ""


def _aggregate_by_task_type(task_details: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate statistics by task type.
    
    Returns a dictionary mapping task_type -> statistics dict.
    """
    type_stats = defaultdict(lambda: {
        'count': 0,
        'total_score': 0.0,
        'compilation_pass_count': 0,
        'correctness_pass_count': 0,
        'speedup_gt_1_count': 0,
        'speedup_values': [],
        'task_names': []
    })
    
    for task in task_details:
        task_type = _extract_task_type(task.get('task_name', ''))
        if not task_type:
            task_type = 'unknown'
        
        stats = type_stats[task_type]
        stats['count'] += 1
        stats['total_score'] += task.get('score', 0.0)
        stats['task_names'].append(task.get('task_name', ''))
        
        if task.get('pass_compilation', False):
            stats['compilation_pass_count'] += 1
        
        if task.get('pass_correctness', False):
            stats['correctness_pass_count'] += 1
        
        # Check speedup (only if both compilation and correctness passed)
        if task.get('pass_compilation', False) and task.get('pass_correctness', False):
            speedup = task.get('speedup_ratio', 0.0)
            if speedup > 1.0:
                stats['speedup_gt_1_count'] += 1
            if speedup > 0:  # Only include valid speedups
                stats['speedup_values'].append(speedup)
    
    # Calculate derived statistics for each task type
    result = {}
    for task_type, stats in type_stats.items():
        count = stats['count']
        speedup_values = stats['speedup_values']
        speed_stats = _compute_speedup_stats(speedup_values)

        result[task_type] = {
            'count': count,
            'total_score': stats['total_score'],
            'average_score': stats['total_score'] / count if count > 0 else 0.0,
            'compilation_pass_count': stats['compilation_pass_count'],
            'compilation_pass_rate': (stats['compilation_pass_count'] / count * 100) if count > 0 else 0.0,
            'correctness_pass_count': stats['correctness_pass_count'],
            'correctness_pass_rate': (stats['correctness_pass_count'] / count * 100) if count > 0 else 0.0,
            'speedup_gt_1_count': stats['speedup_gt_1_count'],
            'speedup_gt_1_rate': (stats['speedup_gt_1_count'] / count * 100) if count > 0 else 0.0,
            **speed_stats,
            'valid_speedup_count': len(speedup_values)
        }
    
    return result


def _ensure_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    """Return a usable logger when caller passes None."""
    if logger is not None:
        return logger

    fallback_logger = logging.getLogger("postprocessing_fallback")
    if not fallback_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        fallback_logger.addHandler(handler)
    fallback_logger.setLevel(logging.INFO)
    fallback_logger.propagate = False
    return fallback_logger


def _normalize_workspace_paths(workspace_paths: Union[str, List[str]]) -> List[str]:
    """
    Accept list of task workspace paths or a workspace root directory.
    """
    if isinstance(workspace_paths, str):
        root = Path(workspace_paths).resolve()
        if root.is_dir():
            subdirs = sorted([str(p) for p in root.iterdir() if p.is_dir()])
            return subdirs
        return [workspace_paths]
    return workspace_paths



def general_log_report(
    aggregate_result: Dict[str, Any], 
    logger: logging.Logger, 
    run_metadata: Optional[Dict[str, str]] = None,
    task_type_breakdown: Optional[Dict[str, Dict[str, Any]]] = None
) -> None:
    """
    Log a formatted report using the provided logger.

    Args:
        aggregate_result: Report dictionary from post_processing()
        logger: Logger instance to use for output
        run_metadata: Optional dict with timestamp, agent, target_gpu
        task_type_breakdown: Optional dict with task type statistics
    """
    for line in _build_general_report_lines(aggregate_result, run_metadata, task_type_breakdown):
        logger.info(line)


def _collect_all_tasks_from_run(run_directory: Path) -> List[str]:
    """
    Collect all task directories from a run directory that have task_result.yaml.
    
    Args:
        run_directory: Run-level directory (e.g., workspace_MI300_cursor/run_20250115_143022/)
    
    Returns:
        List of task directory paths (as strings) that have task_result.yaml
    """
    task_paths = []
    if not run_directory.exists():
        return task_paths
    
    for item in run_directory.iterdir():
        if item.is_dir():
            result_file = item / "task_result.yaml"
            if result_file.exists():
                task_paths.append(str(item))
    
    return sorted(task_paths)


def _safe_read_only_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return (
        path.is_file()
        and not path.is_symlink()
        and metadata.st_nlink == 1
        and not metadata.st_mode & 0o222
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_formal_cohort(run_directory: Path) -> Dict[str, Any] | None:
    manifest_path = run_directory / "campaign_manifest.yaml"
    if not manifest_path.exists():
        return None
    if not _safe_read_only_file(manifest_path):
        raise ValueError(f"formal campaign manifest is unsafe or mutable: {manifest_path}")
    manifest_bytes = _read_immutable_file_no_follow(manifest_path)
    manifest = yaml.safe_load(manifest_bytes) or {}
    if not isinstance(manifest, dict) or manifest.get("schema") != "aka.matched-campaign/v1":
        raise ValueError("formal campaign manifest has an unsupported schema")
    configuration = manifest.get("configuration")
    raw_tasks = configuration.get("tasks") if isinstance(configuration, dict) else None
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("formal campaign manifest has no task cohort")
    comparison = manifest.get("comparison_contract")
    comparison_digest = manifest.get("comparison_contract_sha256")
    agent = manifest.get("agent")
    try:
        run_config_path = Path(str(configuration.get("run_config_path") or ""))
        expected_run_config = _run_config_contract(
            run_config_path,
            agent_name=str(agent.get("template") or "")
            if isinstance(agent, dict)
            else "",
        )
        run_config_valid = bool(
            configuration.get("run_config_sha256")
            == _sha256_file(run_config_path)
            and configuration.get("run_config_contract") == expected_run_config
            and isinstance(comparison, dict)
            and comparison.get("run_config") == expected_run_config
        )
    except (CampaignError, OSError, TypeError, ValueError):
        run_config_valid = False
    if (
        not isinstance(comparison, dict)
        or not isinstance(comparison_digest, str)
        or hashlib.sha256(
            json.dumps(
                comparison, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        != comparison_digest
        or comparison.get("tasks") != raw_tasks
        or not run_config_valid
    ):
        raise ValueError("formal campaign manifest task cohort is not digest-bound")
    task_names: List[str] = []
    task_entries: Dict[str, Dict[str, Any]] = {}
    for expected_index, task in enumerate(raw_tasks, 1):
        if (
            not isinstance(task, dict)
            or task.get("task_index") != expected_index
            or not isinstance(task.get("task_name"), str)
            or not task["task_name"]
        ):
            raise ValueError("formal campaign manifest task cohort is malformed")
        task_names.append(task["task_name"])
        task_entries[task["task_name"]] = task
    if len(task_names) != len(set(task_names)):
        raise ValueError("formal campaign manifest task cohort contains duplicates")
    return {
        "task_names": task_names,
        "task_entries": task_entries,
        "manifest": manifest,
        "campaign_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "comparison_contract_sha256": comparison_digest,
        "ordered_cohort_sha256": hashlib.sha256(
            json.dumps(raw_tasks, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }


def _formal_workspace_map(
    run_directory: Path, task_names: List[str]
) -> Dict[str, Path]:
    match = re.match(r"^run_(\d{8}_\d{6})(?:_|$)", run_directory.name)
    if match is None:
        raise ValueError("formal run directory has no canonical timestamp")
    timestamp = match.group(1)
    expected = {
        task_name: get_task_workspace_path(run_directory, task_name, timestamp)
        for task_name in task_names
    }
    expected_paths = [path.resolve() for path in expected.values()]
    if len(expected_paths) != len(set(expected_paths)):
        raise ValueError("formal campaign task names collide on canonical workspace paths")
    by_task: Dict[str, Path] = {}
    for task_name, workspace in expected.items():
        if not (workspace.is_dir() and not workspace.is_symlink()):
            continue
        result_path = workspace / "task_result.yaml"
        if not result_path.exists():
            continue
        by_task[task_name] = workspace
    return by_task


def _require_regular_directory_chain(path: Path, root: Path, label: str) -> None:
    """Reject symlinked/non-directory components between a run and evidence root."""
    root_resolved = root.resolve(strict=True)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the run directory") from error
    current = root
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise ValueError(f"{label} is missing: {current}") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} has an unsafe directory component: {current}")
    try:
        path.resolve(strict=True).relative_to(root_resolved)
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} escapes the resolved run directory") from error


def _regular_tree_manifest(root: Path) -> Dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"canonical lineage workspace is unsafe: {root}")
    manifest: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if path.is_symlink():
            raise ValueError(f"canonical lineage contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file() or metadata.st_nlink != 1:
            raise ValueError(f"canonical lineage contains an unsafe file: {path}")
        manifest[relative] = _sha256_file(path)
    if not manifest:
        raise ValueError(f"canonical lineage workspace is empty: {root}")
    return manifest


def _canonical_json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _json_object_without_duplicate_keys(
    pairs: List[tuple[str, Any]],
) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _formal_task_identity(
    *,
    run_directory: Path,
    task_name: str,
    formal: Dict[str, Any],
) -> Dict[str, Any]:
    """Project the manifest fields that every attempt must independently bind."""
    manifest = formal.get("manifest")
    task = formal.get("task_entries", {}).get(task_name)
    policy = manifest.get("policy") if isinstance(manifest, dict) else None
    measurement = manifest.get("measurement") if isinstance(manifest, dict) else None
    comparison = (
        manifest.get("comparison_contract") if isinstance(manifest, dict) else None
    )
    codex = comparison.get("codex") if isinstance(comparison, dict) else None
    agent = manifest.get("agent") if isinstance(manifest, dict) else None
    runtime = manifest.get("runtime") if isinstance(manifest, dict) else None
    gpu = runtime.get("gpu") if isinstance(runtime, dict) else None
    mappings = gpu.get("task_mapping") if isinstance(gpu, dict) else None
    exclusivity = gpu.get("exclusivity") if isinstance(gpu, dict) else None
    if (
        not isinstance(manifest, dict)
        or not isinstance(task, dict)
        or not isinstance(policy, dict)
        or policy.get("attempts") != 3
        or not isinstance(measurement, dict)
        or not isinstance(comparison, dict)
        or not isinstance(codex, dict)
        or not isinstance(agent, dict)
        or agent.get("backend_runtime_closure_sha256")
        != codex.get("backend_runtime_closure_sha256")
        or agent.get("template") not in _FORMAL_RECEIPT_SCHEMAS
        or agent.get("session_receipt_schema")
        != _FORMAL_RECEIPT_SCHEMAS[agent["template"]]
        or not isinstance(mappings, list)
        or not isinstance(exclusivity, dict)
        or exclusivity.get("exclusivity_verified") is not True
    ):
        raise ValueError("formal manifest lacks a complete attempt identity")
    try:
        attempt_components = [
            campaign_task_path_component(name) for name in formal["task_names"]
        ]
        attempt_component = campaign_task_path_component(task_name)
    except (CampaignError, TypeError, UnicodeError) as error:
        raise ValueError("formal manifest has an unsafe task name") from error
    if len(attempt_components) != len(set(attempt_components)):
        raise ValueError("formal manifest task names collide on attempt roots")
    task_index = task.get("task_index")
    if type(task_index) is not int or task_index < 1:
        raise ValueError("formal manifest task index is invalid")
    matches = [
        mapping
        for mapping in mappings
        if isinstance(mapping, dict)
        and mapping.get("task_name") == task_name
        and mapping.get("task_index") == task_index
    ]
    if len(matches) != 1:
        raise ValueError("formal manifest task/GPU mapping is ambiguous")
    assigned_gpu = matches[0].get("assigned_host_gpu_id")
    manifest_path = run_directory / "campaign_manifest.yaml"
    if (
        not isinstance(assigned_gpu, str)
        or re.fullmatch(r"0|[1-9][0-9]*", assigned_gpu) is None
        or not _safe_read_only_file(manifest_path)
    ):
        raise ValueError("formal manifest task/GPU binding is incomplete")
    resolved_manifest = manifest_path.resolve(strict=True)
    return {
        "manifest": manifest,
        "task": task,
        "policy": policy,
        "measurement": measurement,
        "agent_template": agent["template"],
        "receipt_schema": agent["session_receipt_schema"],
        "assigned_host_gpu_id": assigned_gpu,
        "attempt_component": attempt_component,
        "gpu": gpu,
        "manifest_path": resolved_manifest,
        "backend_runtime_closure_sha256": codex.get(
            "backend_runtime_closure_sha256"
        ),
    }


def _expected_attempt_campaign_binding(
    *,
    identity: Dict[str, Any],
    formal: Dict[str, Any],
    task_name: str,
    attempt: int,
) -> Dict[str, Any]:
    task = identity["task"]
    binding = {
        "schema": _ATTEMPT_CAMPAIGN_BINDING_SCHEMA,
        "formal_execution_sha256": identity["manifest"].get(
            "formal_execution_sha256"
        ),
        "campaign_manifest_path": str(identity["manifest_path"]),
        "campaign_manifest_sha256": formal["campaign_manifest_sha256"],
        "comparison_contract_sha256": formal[
            "comparison_contract_sha256"
        ],
        "backend_runtime_closure_sha256": identity[
            "backend_runtime_closure_sha256"
        ],
        "task_package_manifest_sha256": task.get("package_manifest_sha256"),
        "task_config_sha256": task.get("config_sha256"),
        "task_name": task_name,
        "task_index": task.get("task_index"),
        "total_tasks": len(formal["task_names"]),
        "attempt_index": attempt,
        "attempt_count": 3,
        "assigned_host_gpu_id": identity["assigned_host_gpu_id"],
    }
    if set(binding) != _ATTEMPT_CAMPAIGN_BINDING_KEYS or any(
        value is None for value in binding.values()
    ):
        raise ValueError("formal attempt campaign binding is incomplete")
    return binding


def _receipt_source_delta_files(receipt: Dict[str, Any]) -> List[str] | None:
    integrity = receipt.get("workspace_integrity")
    final_changes = (
        integrity.get("final_changes") if isinstance(integrity, dict) else None
    )
    changed_files = (
        final_changes.get("changed_files")
        if isinstance(final_changes, dict)
        else None
    )
    if isinstance(changed_files, list) and all(
        isinstance(path, str) and path for path in changed_files
    ):
        return changed_files
    return None


def _static_session_receipt_binding(
    receipt: Dict[str, Any], *, agent_template: str
) -> Dict[str, Any]:
    """Rebuild receipt fields whose truth is directly recoverable from its bytes."""
    invocation = receipt.get("invocation")
    binding: Dict[str, Any] = {
        "schema": receipt.get("schema"),
        "comparison_contract_sha256": receipt.get(
            "comparison_contract_sha256"
        ),
        "terminal_status": receipt.get("terminal_status"),
        "codex": receipt.get("codex"),
        "invocation_sha256": (
            _canonical_json_digest(invocation)
            if isinstance(invocation, dict)
            else None
        ),
        "attempt_process_cleanup": receipt.get("attempt_process_cleanup"),
        "budgets": receipt.get("budgets"),
        "turn_budget": receipt.get("turn_budget"),
        "workspace_integrity": receipt.get("workspace_integrity"),
        "gpu": receipt.get("gpu"),
        "lineage": receipt.get("lineage"),
        "source_delta_files": (
            _receipt_source_delta_files(receipt)
            if agent_template == "codex"
            else None
        ),
        "campaign_binding": receipt.get("campaign_binding"),
    }
    if agent_template == "apex":
        lineage = receipt.get("lineage")
        prompt_event = (
            lineage.get("prompt_event") if isinstance(lineage, dict) else None
        )
        binding["lineage_verified"] = isinstance(lineage, dict)
        binding["event_bound_prompt"] = (
            {
                "binding": prompt_event.get("binding"),
                "event_id": prompt_event.get("event_id"),
                "sha256": prompt_event.get("sha256"),
                "size_bytes": prompt_event.get("size_bytes"),
                "stdin_transport_attested": prompt_event.get(
                    "stdin_transport_attested"
                ),
            }
            if isinstance(prompt_event, dict)
            else None
        )
    return binding


def _receipt_gpu_binding_valid(
    receipt: Dict[str, Any], *, identity: Dict[str, Any]
) -> bool:
    observed = receipt.get("gpu")
    expected = identity["gpu"]
    exclusivity = expected.get("exclusivity")
    devices = expected.get("devices")
    selected = [
        device
        for device in devices
        if isinstance(device, dict)
        and device.get("host_device_id")
        == identity["assigned_host_gpu_id"]
    ] if isinstance(devices, list) else []
    device = selected[0] if len(selected) == 1 else None
    runtime_identity = (
        observed.get("runtime_identity") if isinstance(observed, dict) else None
    )
    rocm_identity = (
        runtime_identity.get("rocm_smi_identity")
        if isinstance(runtime_identity, dict)
        else None
    )
    torch_identity = (
        runtime_identity.get("torch")
        if isinstance(runtime_identity, dict)
        else None
    )
    return bool(
        isinstance(observed, dict)
        and isinstance(exclusivity, dict)
        and isinstance(device, dict)
        and observed.get("policy")
        == "physical_device_boundary_with_host_exclusivity_v1"
        and observed.get("plan_sha256")
        == expected.get("gpu_boundary_plan_sha256")
        and isinstance(observed.get("boundary_receipt_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", observed["boundary_receipt_sha256"])
        is not None
        and observed.get("exclusivity_receipt_sha256")
        == exclusivity.get("sha256")
        and observed.get("exclusivity_verified") is True
        and observed.get("host_gpu_id") == identity["assigned_host_gpu_id"]
        and observed.get("unique_id") == device.get("unique_id")
        and observed.get("allowed_render_nodes") == device.get("render_nodes")
        and isinstance(runtime_identity, dict)
        and runtime_identity.get("visible_physical_gpu_count") == 1
        and isinstance(rocm_identity, dict)
        and rocm_identity.get("unique_id") == device.get("unique_id")
        and isinstance(torch_identity, dict)
        and torch_identity.get("device_count") == 1
    )


def _load_bound_attempt_receipt(
    *,
    run_directory: Path,
    attempt_root: Path,
    task_name: str,
    attempt: int,
    record: Dict[str, Any],
    formal: Dict[str, Any],
    identity: Dict[str, Any],
    require_complete: bool,
) -> Dict[str, Any]:
    receipt_path = attempt_root / f"attempt_{attempt:02d}" / "session_receipt.json"
    expected_relative = str(receipt_path.relative_to(run_directory))
    if record.get("session_receipt") != expected_relative:
        raise ValueError(f"attempt {attempt} receipt path is not canonical")
    _require_regular_directory_chain(
        receipt_path.parent, run_directory, f"attempt {attempt} receipt parent"
    )
    if not _safe_read_only_file(receipt_path):
        raise ValueError(f"attempt {attempt} receipt is unsafe or mutable")
    try:
        receipt_bytes = _read_immutable_file_no_follow(receipt_path)
        receipt = json.loads(
            receipt_bytes,
            object_pairs_hook=_json_object_without_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"attempt {attempt} receipt is unreadable") from error
    if not isinstance(receipt, dict):
        raise ValueError(f"attempt {attempt} receipt is not a JSON object")
    if (
        receipt.get("schema") != identity["receipt_schema"]
        or record.get("session_receipt_sha256")
        != hashlib.sha256(receipt_bytes).hexdigest()
    ):
        raise ValueError(f"attempt {attempt} receipt schema or digest differs")
    if not _receipt_gpu_binding_valid(receipt, identity=identity):
        raise ValueError(f"attempt {attempt} receipt GPU binding differs")
    expected_campaign_binding = _expected_attempt_campaign_binding(
        identity=identity,
        formal=formal,
        task_name=task_name,
        attempt=attempt,
    )
    observed_campaign_binding = receipt.get("campaign_binding")
    if (
        not isinstance(observed_campaign_binding, dict)
        or set(observed_campaign_binding) != _ATTEMPT_CAMPAIGN_BINDING_KEYS
        or observed_campaign_binding != expected_campaign_binding
    ):
        raise ValueError(f"attempt {attempt} receipt campaign binding differs")
    observed_binding = record.get("session_receipt_binding")
    expected_binding = _static_session_receipt_binding(
        receipt, agent_template=identity["agent_template"]
    )
    if (
        identity["agent_template"] == "apex"
        and not require_complete
        and isinstance(observed_binding, dict)
        and observed_binding.get("lineage_verified") is False
    ):
        # The producer's full Apex artifact validator may conservatively clear
        # these two derived fields on a failed session. All directly copied
        # fields, including campaign_binding, remain independently checkable.
        expected_binding["lineage_verified"] = False
        expected_binding["event_bound_prompt"] = None
    if (
        not isinstance(observed_binding, dict)
        or observed_binding != expected_binding
        or record.get("session_receipt_binding_sha256")
        != _canonical_json_digest(expected_binding)
    ):
        raise ValueError(f"attempt {attempt} static receipt binding differs")
    if record.get("session_succeeded") is not (
        receipt.get("session_succeeded") is True
    ):
        raise ValueError(f"attempt {attempt} receipt success projection differs")
    return receipt


def _float_matches(observed: Any, expected: float) -> bool:
    if isinstance(observed, bool):
        return False
    try:
        parsed = float(observed)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and math.isclose(
        parsed, expected, rel_tol=1e-12, abs_tol=1e-12
    )


def _revalidate_attempt_record(
    *,
    run_directory: Path,
    attempt_root: Path,
    task_name: str,
    timestamp: str,
    attempt: int,
    record: Dict[str, Any],
    formal: Dict[str, Any],
    identity: Dict[str, Any],
    require_complete: bool,
) -> Dict[str, Any]:
    if record.get("session") != f"fresh-{attempt:02d}":
        raise ValueError(f"attempt {attempt} session identity differs")
    relative_workspace = record.get("workspace")
    if relative_workspace is None:
        if require_complete:
            raise ValueError(f"attempt {attempt} has no evaluator workspace")
        if "session_receipt" in record:
            _load_bound_attempt_receipt(
                run_directory=run_directory,
                attempt_root=attempt_root,
                task_name=task_name,
                attempt=attempt,
                record=record,
                formal=formal,
                identity=identity,
                require_complete=False,
            )
        if (
            record.get("central_evaluator_report") is not None
            or record.get("selection_eligible") is not False
            or not _float_matches(record.get("measured_rate_per_ms"), 0.0)
            or record.get("attempt_completed") is not False
            or not isinstance(record.get("eligibility_errors"), list)
            or not record["eligibility_errors"]
        ):
            raise ValueError(f"attempt {attempt} incomplete record is inconsistent")
        return {
            **record,
            "central_evaluator_report": None,
            "selection_eligible": False,
            "measured_rate_per_ms": 0.0,
        }
    if not isinstance(relative_workspace, str) or Path(relative_workspace).is_absolute():
        raise ValueError(f"attempt {attempt} workspace path is invalid")
    expected_workspace = get_task_workspace_path(
        attempt_root / f"attempt_{attempt:02d}", task_name, timestamp
    )
    workspace = run_directory / relative_workspace
    if workspace != expected_workspace:
        raise ValueError(f"attempt {attempt} workspace path is not canonical")
    _require_regular_directory_chain(
        workspace, run_directory, f"attempt {attempt} workspace"
    )
    receipt = _load_bound_attempt_receipt(
        run_directory=run_directory,
        attempt_root=attempt_root,
        task_name=task_name,
        attempt=attempt,
        record=record,
        formal=formal,
        identity=identity,
        require_complete=require_complete,
    )
    manifest = _regular_tree_manifest(workspace)
    manifest_sha256 = _canonical_json_digest(manifest)
    report_path = workspace / "task_result.yaml"
    expected_report = str(report_path.relative_to(run_directory))
    if (
        record.get("workspace_manifest_sha256") != manifest_sha256
        or record.get("central_evaluator_report") != expected_report
        or record.get("central_evaluator_report_sha256")
        != manifest.get("task_result.yaml")
    ):
        raise ValueError(f"attempt {attempt} evaluator lineage differs")
    try:
        report = yaml.safe_load(report_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"attempt {attempt} evaluator result is unreadable") from error
    if not isinstance(report, dict) or report.get("task_name") != task_name:
        raise ValueError(f"attempt {attempt} evaluator task identity differs")

    errors = _evaluation_eligibility_errors(workspace, report)
    source_delta_files = _receipt_source_delta_files(receipt)
    no_candidate = (
        identity["agent_template"] == "apex"
        and receipt.get("terminal_status") == "no_gain"
    ) or (
        identity["agent_template"] == "codex" and source_delta_files == []
    )
    evaluation_mode = report.get("evaluation_mode")
    score_eligible = report.get("agent_session_score_eligible")
    session_succeeded = report.get("agent_session_succeeded")
    terminal_status = report.get("agent_session_terminal_status")
    if evaluation_mode is not None or score_eligible is not None:
        if no_candidate:
            if evaluation_mode != "no_candidate_baseline_replay_v1":
                errors.append("no_candidate_evaluation_mode_mismatch")
            if score_eligible is not False:
                errors.append("no_candidate_score_eligibility_mismatch")
            if session_succeeded is not True:
                errors.append("no_candidate_session_success_mismatch")
            if terminal_status != "no_gain":
                errors.append("no_candidate_terminal_status_mismatch")
        else:
            if evaluation_mode != "candidate_scoring_v1":
                errors.append("diagnostic_evaluation_not_scoreable")
            if score_eligible is not True:
                errors.append("agent_session_not_score_eligible")
    if identity["agent_template"] == "apex" and (
        terminal_status is not None
        and terminal_status != receipt.get("terminal_status")
    ):
        errors.append("apex_report_terminal_status_mismatch")
    if record.get("attempt_completed") is not True:
        errors.append("agent_session_or_attempt_failed")
    if receipt.get("session_succeeded") is not True:
        errors.append("agent_session_receipt_not_successful")
    errors = sorted(set(errors))
    declared_errors = record.get("eligibility_errors")
    if (
        not isinstance(declared_errors, list)
        or declared_errors != sorted(set(declared_errors))
        or (
            require_complete
            and declared_errors != errors
        )
        or (
            not require_complete
            and not set(errors).issubset(declared_errors)
        )
    ):
        raise ValueError(f"attempt {attempt} eligibility errors differ")
    optimized = report.get("best_optimized_execution_time")
    try:
        optimized_ms = 0.0 if isinstance(optimized, bool) else float(optimized)
    except (TypeError, ValueError):
        optimized_ms = 0.0
    if not math.isfinite(optimized_ms) or optimized_ms <= 0:
        optimized_ms = 0.0
    try:
        raw_speedup = report.get("speedup_ratio") or 0.0
        speedup = 0.0 if isinstance(raw_speedup, bool) else float(raw_speedup)
    except (TypeError, ValueError):
        speedup = 0.0
    effective_errors = errors if require_complete else declared_errors
    eligible = not effective_errors and not no_candidate
    measured_rate = 1.0 / optimized_ms if eligible else 0.0
    exact_fields = {
        "pass_compilation": report.get("pass_compilation") is True,
        "pass_correctness": report.get("pass_correctness") is True,
        "benchmark_method_consistent": (
            report.get("benchmark_method_consistent") is True
        ),
        "evaluation_mode": evaluation_mode,
        "agent_session_score_eligible": score_eligible,
        "agent_session_terminal_status": terminal_status,
        "selection_eligible": eligible,
    }
    if any(record.get(key) != value for key, value in exact_fields.items()):
        raise ValueError(f"attempt {attempt} evaluator projection differs")
    if (
        not _float_matches(record.get("optimized_execution_time_ms"), optimized_ms)
        or not _float_matches(record.get("speedup_ratio"), speedup)
        or not _float_matches(record.get("measured_rate_per_ms"), measured_rate)
    ):
        raise ValueError(f"attempt {attempt} measured result projection differs")
    return {
        **record,
        "selection_eligible": eligible,
        "measured_rate_per_ms": measured_rate,
    }


def _revalidate_task_campaign(
    *,
    run_directory: Path,
    task_name: str,
    formal: Dict[str, Any],
    require_complete: bool,
) -> Dict[str, Any]:
    match = re.match(r"^run_(\d{8}_\d{6})(?:_|$)", run_directory.name)
    if match is None:
        raise ValueError("formal run directory has no canonical timestamp")
    identity = _formal_task_identity(
        run_directory=run_directory, task_name=task_name, formal=formal
    )
    attempt_root = (
        run_directory / ".campaign_attempts" / identity["attempt_component"]
    )
    _require_regular_directory_chain(
        attempt_root, run_directory, "formal task attempt root"
    )
    task_campaign_path = attempt_root / "task_campaign.yaml"
    if not _safe_read_only_file(task_campaign_path):
        raise ValueError("task campaign lineage is unsafe or mutable")
    try:
        task_campaign_bytes = _read_immutable_file_no_follow(task_campaign_path)
        task_campaign = yaml.safe_load(task_campaign_bytes) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError("task campaign lineage is unreadable") from error
    task = identity["task"]
    expected_top_level = {
        "schema": "aka.matched-task-attempts/v1",
        "formal_execution_sha256": identity["manifest"].get(
            "formal_execution_sha256"
        ),
        "task_name": task_name,
        "assigned_host_gpu_id": identity["assigned_host_gpu_id"],
        "task_index": task.get("task_index"),
        "total_tasks": len(formal["task_names"]),
        "task_config_path": task.get("config_path"),
        "task_config_sha256": task.get("config_sha256"),
        "task_package_manifest_sha256": task.get("package_manifest_sha256"),
        "gpu_exclusivity_verified": True,
        "campaign_manifest_sha256": formal["campaign_manifest_sha256"],
        "comparison_contract_sha256": formal[
            "comparison_contract_sha256"
        ],
        "policy": identity["policy"],
        "measurement_contract": identity["measurement"].get("contract"),
        "is_apex_canonical_300_sample_grade": identity["measurement"].get(
            "is_apex_canonical_300_sample_grade"
        ),
    }
    if not isinstance(task_campaign, dict) or any(
        task_campaign.get(key) != value for key, value in expected_top_level.items()
    ):
        raise ValueError("task campaign top-level manifest binding differs")
    attempts = task_campaign.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("task campaign has no attempt records")
    if any(not isinstance(record, dict) for record in attempts):
        raise ValueError("task campaign contains a malformed attempt record")
    indices = [record.get("attempt") for record in attempts]
    expected_indices = list(range(1, len(attempts) + 1))
    if (
        any(type(index) is not int for index in indices)
        or indices != expected_indices
        or len(attempts) > 3
    ):
        raise ValueError("task campaign attempt identities are missing or duplicated")
    if require_complete and indices != [1, 2, 3]:
        raise ValueError("completed formal task requires attempts exactly 1..3")
    refreshed = [
        _revalidate_attempt_record(
            run_directory=run_directory,
            attempt_root=attempt_root,
            task_name=task_name,
            timestamp=match.group(1),
            attempt=index,
            record=record,
            formal=formal,
            identity=identity,
            require_complete=require_complete,
        )
        for index, record in enumerate(attempts, 1)
    ]
    selected = _select_attempt(refreshed)
    selected_attempt = selected.get("attempt") if selected is not None else None
    declared_selected = task_campaign.get("selected_attempt")
    if declared_selected is not None and type(declared_selected) is not int:
        raise ValueError("task campaign selected attempt identity is invalid")
    all_evaluated = len(refreshed) == 3 and all(
        record.get("central_evaluator_report") is not None for record in refreshed
    )
    all_sessions = len(refreshed) == 3 and all(
        record.get("attempt_completed") is True for record in refreshed
    )
    if (
        declared_selected != selected_attempt
        or task_campaign.get("all_attempts_centrally_evaluated") is not all_evaluated
        or task_campaign.get("all_agent_sessions_succeeded") is not all_sessions
        or task_campaign.get("failure_reasons")
        != _campaign_failure_reasons(task_campaign)
    ):
        raise ValueError("task campaign derived selection or completion differs")
    if require_complete and (
        task_campaign.get("campaign_manifest_unchanged") is not True
        or task_campaign.get("within_evaluator_allowance") is not True
        or task_campaign.get("within_task_timeout") is not True
        or task_campaign.get("failure_reasons") != []
        or selected is None
        or selected.get("selection_eligible") is not True
    ):
        raise ValueError("task campaign does not prove a completed task")
    return {
        "task_campaign": task_campaign,
        "task_campaign_path": task_campaign_path,
        "task_campaign_sha256": hashlib.sha256(task_campaign_bytes).hexdigest(),
        "selected": selected,
        "attempt_root": attempt_root,
        "timestamp": match.group(1),
    }


def _validate_canonical_lineage(
    *,
    run_directory: Path,
    task_name: str,
    canonical: Path,
    formal: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, str]]:
    match = re.match(r"^run_(\d{8}_\d{6})(?:_|$)", run_directory.name)
    if match is None:
        raise ValueError("formal run directory has no canonical timestamp")
    expected_canonical = get_task_workspace_path(
        run_directory, task_name, match.group(1)
    )
    if canonical != expected_canonical:
        raise ValueError("canonical task workspace path differs from manifest identity")
    _require_regular_directory_chain(canonical, run_directory, "canonical workspace")
    result_path = canonical / "task_result.yaml"
    if not _safe_read_only_file(result_path):
        raise ValueError(f"canonical task result is unsafe or mutable: {result_path}")
    result = yaml.safe_load(result_path.read_text(encoding="utf-8")) or {}
    if not isinstance(result, dict) or result.get("task_name") != task_name:
        raise ValueError("canonical task result identity differs from manifest")
    campaign_evidence = result.get("campaign_evidence")
    if not isinstance(campaign_evidence, dict):
        raise ValueError("canonical task result lacks campaign lineage")

    validation = _revalidate_task_campaign(
        run_directory=run_directory,
        task_name=task_name,
        formal=formal,
        require_complete=True,
    )
    task_campaign_path = validation["task_campaign_path"]
    task_campaign_sha256 = validation["task_campaign_sha256"]
    task_campaign = validation["task_campaign"]
    selected = validation["selected"]
    selected_attempt = selected["attempt"]
    relative_workspace = selected.get("workspace")
    if not isinstance(relative_workspace, str):
        raise ValueError("selected attempt has no workspace lineage")
    if Path(relative_workspace).is_absolute():
        raise ValueError("selected attempt workspace must be run-relative")
    attempt_directory = validation["attempt_root"] / f"attempt_{selected_attempt:02d}"
    expected_selected_workspace = get_task_workspace_path(
        attempt_directory, task_name, validation["timestamp"]
    )
    selected_workspace = run_directory / relative_workspace
    if selected_workspace != expected_selected_workspace:
        raise ValueError("selected attempt workspace path is not canonical")
    _require_regular_directory_chain(
        selected_workspace, run_directory, "selected attempt workspace"
    )

    selected_manifest = _regular_tree_manifest(selected_workspace)
    selected_manifest_sha256 = hashlib.sha256(
        json.dumps(
            selected_manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if selected.get("workspace_manifest_sha256") != selected_manifest_sha256:
        raise ValueError("selected attempt workspace digest changed")
    selected_result_path = selected_workspace / "task_result.yaml"
    expected_report_relative = str(selected_result_path.relative_to(run_directory))
    selected_result_sha256 = _sha256_file(selected_result_path)
    if (
        selected.get("central_evaluator_report") != expected_report_relative
        or selected.get("central_evaluator_report_sha256")
        != selected_result_sha256
        or selected_manifest.get("task_result.yaml") != selected_result_sha256
    ):
        raise ValueError("selected evaluator result lineage changed")

    evidence_hashes = {
        name: selected_manifest.get(name)
        for name in ("baseline_perf.yaml", "optimized_perf.yaml")
    }
    if any(value is None for value in evidence_hashes.values()):
        raise ValueError("selected attempt lacks performance evidence lineage")
    expected_campaign_evidence = {
        "schema": "aka.matched-task-attempts/v1",
        "campaign_manifest_sha256": formal["campaign_manifest_sha256"],
        "comparison_contract_sha256": formal["comparison_contract_sha256"],
        "task_campaign_sha256": task_campaign_sha256,
        "attempt_count": task_campaign.get("policy", {}).get("attempts"),
        "selected_attempt": selected_attempt,
        "selection_policy": task_campaign.get("policy", {}).get("selection_policy"),
        "selected_measured_rate_per_ms": selected.get("measured_rate_per_ms"),
        "attempt_manifest": str(task_campaign_path.relative_to(run_directory)),
        "measurement_contract": task_campaign.get("measurement_contract"),
        "is_apex_canonical_300_sample_grade": False,
        "selected_central_evaluator_report_sha256": selected_result_sha256,
        "selected_performance_evidence_sha256": evidence_hashes,
        "selected_workspace_manifest_sha256": selected_manifest_sha256,
    }
    if campaign_evidence != expected_campaign_evidence:
        raise ValueError("canonical task result campaign lineage is inconsistent")
    selected_result = yaml.safe_load(
        selected_result_path.read_text(encoding="utf-8")
    ) or {}
    if not isinstance(selected_result, dict):
        raise ValueError("selected evaluator result is unreadable")
    expected_result = dict(selected_result)
    expected_result["campaign_evidence"] = expected_campaign_evidence
    if result != expected_result:
        raise ValueError("canonical task result differs from selected evaluator result")
    for name, expected_sha256 in evidence_hashes.items():
        canonical_evidence = canonical / name
        if (
            not _safe_read_only_file(canonical_evidence)
            or _sha256_file(canonical_evidence) != expected_sha256
        ):
            raise ValueError(f"canonical performance evidence differs: {name}")

    canonical_manifest = _regular_tree_manifest(canonical)
    expected_canonical_manifest = dict(selected_manifest)
    expected_canonical_manifest["task_result.yaml"] = _sha256_file(result_path)
    if canonical_manifest != expected_canonical_manifest:
        raise ValueError("canonical workspace tree differs from selected workspace")
    canonical_manifest_sha256 = hashlib.sha256(
        json.dumps(
            canonical_manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return result, {
        "canonical_workspace_manifest_sha256": canonical_manifest_sha256,
        "selected_workspace_manifest_sha256": selected_manifest_sha256,
    }


def _validated_failure_binding(
    run_directory: Path,
    task_name: str,
    formal: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate a single terminal failure marker against sealed campaign evidence."""
    task_entry = formal["task_entries"][task_name]
    expected_index = task_entry["task_index"]
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", task_name).strip("_")
    expected_suffix = f"{expected_index:06d}_{safe_name or 'task'}.yaml"
    expected_workspace = get_task_workspace_path(
        run_directory,
        task_name,
        re.match(r"^run_(\d{8}_\d{6})(?:_|$)", run_directory.name).group(1),
    )

    marker_payloads: List[tuple[Path, Dict[str, Any]]] = []
    marker_candidates: List[Path] = []
    failed_dir = run_directory / ".parallel" / "failed"
    if failed_dir.exists():
        try:
            failed_metadata = failed_dir.lstat()
        except OSError:
            failed_metadata = None
        if (
            failed_metadata is None
            or stat.S_ISLNK(failed_metadata.st_mode)
            or not stat.S_ISDIR(failed_metadata.st_mode)
        ):
            return {
                "reason_codes": ["failed_marker_directory_unsafe"],
                "campaign_evidence_path": None,
                "campaign_evidence_sha256": None,
                "terminal_binding_verified": False,
            }
        marker_name = re.compile(
            rf"worker_[A-Za-z0-9._-]+__{re.escape(expected_suffix)}"
        )
        marker_candidates = sorted(
            path
            for path in failed_dir.iterdir()
            if marker_name.fullmatch(path.name)
        )
        for descriptor in marker_candidates:
            if not _safe_read_only_file(descriptor):
                marker_payloads.append((descriptor, {}))
                continue
            try:
                payload = yaml.safe_load(
                    _read_immutable_file_no_follow(descriptor)
                ) or {}
            except (OSError, UnicodeError, yaml.YAMLError):
                marker_payloads.append((descriptor, {}))
                continue
            marker_payloads.append(
                (descriptor, payload if isinstance(payload, dict) else {})
            )

    marker_errors: List[str] = []
    descriptor_binding: Dict[str, Any] | None = None
    if len(marker_candidates) != 1:
        marker_errors.append(
            "duplicate_failed_markers" if len(marker_candidates) > 1
            else "failed_marker_missing"
        )
    else:
        descriptor, payload = marker_payloads[0]
        descriptor_match = re.fullmatch(
            rf"worker_([A-Za-z0-9._-]+)__{re.escape(expected_suffix)}",
            descriptor.name,
        )
        if (
            descriptor_match is None
            or payload.get("task_name") != task_name
            or payload.get("index") != expected_index
            or payload.get("total_tasks") != len(formal["task_names"])
            or payload.get("status") != "failed"
            or payload.get("workspace_path") != str(expected_workspace)
            or payload.get("worker_id") != descriptor_match.group(1)
        ):
            marker_errors.append("failed_marker_identity_invalid")
        raw_binding = payload.get("failure")
        if (
            not isinstance(raw_binding, dict)
            or raw_binding.get("schema") != "aka.formal-task-failure/v1"
            or raw_binding.get("task_name") != task_name
        ):
            marker_errors.append("failed_marker_binding_invalid")
        else:
            descriptor_binding = raw_binding

    evidence_path = (
        run_directory
        / ".campaign_attempts"
        / campaign_task_path_component(task_name)
        / "task_campaign.yaml"
    )
    evidence_relative: str | None = None
    evidence_sha256: str | None = None
    evidence_reasons: List[str] = []
    evidence_errors: List[str] = []
    evidence: Dict[str, Any] | None = None
    if _safe_read_only_file(evidence_path):
        try:
            loaded = yaml.safe_load(
                _read_immutable_file_no_follow(evidence_path)
            ) or {}
        except (OSError, UnicodeError, yaml.YAMLError):
            loaded = None
        if isinstance(loaded, dict):
            evidence = loaded
            recomputed_reasons = _campaign_failure_reasons(evidence)
            raw_reasons = evidence.get("failure_reasons")
            try:
                _revalidate_task_campaign(
                    run_directory=run_directory,
                    task_name=task_name,
                    formal=formal,
                    require_complete=False,
                )
            except (OSError, UnicodeError, ValueError, yaml.YAMLError):
                posthoc_valid = False
            else:
                posthoc_valid = True
            if (
                not posthoc_valid
                or raw_reasons != recomputed_reasons
                or not recomputed_reasons
            ):
                evidence_errors.append("task_campaign_evidence_contract_invalid")
            else:
                evidence_reasons = recomputed_reasons
                evidence_relative = str(evidence_path.relative_to(run_directory))
                evidence_sha256 = _sha256_file(evidence_path)
        else:
            evidence_errors.append("task_campaign_evidence_unreadable")
    else:
        evidence_errors.append("immutable_task_campaign_evidence_missing")

    primary_reason: str | None = None
    if descriptor_binding is not None and not evidence_errors and not marker_errors:
        declared_primary = descriptor_binding.get("primary_reason")
        expected_declared_reasons = sorted(
            set(
                evidence_reasons
                + (
                    [declared_primary]
                    if declared_primary in _FORMAL_PRIMARY_FAILURE_REASONS
                    else []
                )
            )
        )
        if (
            declared_primary not in _FORMAL_PRIMARY_FAILURE_REASONS
            or descriptor_binding.get("campaign_manifest_sha256")
            != formal["campaign_manifest_sha256"]
            or descriptor_binding.get("comparison_contract_sha256")
            != formal["comparison_contract_sha256"]
            or descriptor_binding.get("campaign_evidence_path") != evidence_relative
            or descriptor_binding.get("campaign_evidence_sha256") != evidence_sha256
            or descriptor_binding.get("reason_codes") != expected_declared_reasons
        ):
            marker_errors.append("failed_marker_full_binding_invalid")
        else:
            primary_reason = declared_primary

    terminal_verified = not marker_errors and not evidence_errors
    reason_codes = sorted(
        set(
            evidence_reasons
            + marker_errors
            + evidence_errors
            + ([primary_reason] if primary_reason else [])
        )
    )
    if not reason_codes:
        reason_codes.append("formal_task_not_canonical")
    return {
        "reason_codes": reason_codes,
        "campaign_evidence_path": evidence_relative,
        "campaign_evidence_sha256": evidence_sha256,
        "terminal_binding_verified": terminal_verified,
    }


def _read_immutable_file_no_follow(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"unsafe immutable evidence file: {path}") from error
    try:
        opened = os.fstat(descriptor)
        lexical = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_mode & 0o222
            or opened.st_dev != lexical.st_dev
            or opened.st_ino != lexical.st_ino
        ):
            raise ValueError(f"unsafe immutable evidence file: {path}")
        chunks: List[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _prepare_reports_directory(run_directory: Path) -> Path:
    run_root = run_directory.resolve(strict=True)
    run_metadata = run_directory.lstat()
    if stat.S_ISLNK(run_metadata.st_mode) or not stat.S_ISDIR(run_metadata.st_mode):
        raise ValueError(f"formal report run directory is unsafe: {run_directory}")
    reports_directory = run_directory / "reports"
    if not reports_directory.exists():
        try:
            os.mkdir(reports_directory, 0o755)
        except FileExistsError:
            pass
    try:
        reports_metadata = reports_directory.lstat()
    except OSError as error:
        raise ValueError("reports directory cannot be inspected") from error
    if (
        stat.S_ISLNK(reports_metadata.st_mode)
        or not stat.S_ISDIR(reports_metadata.st_mode)
    ):
        raise ValueError(f"reports directory is unsafe: {reports_directory}")
    resolved_reports = reports_directory.resolve(strict=True)
    try:
        resolved_reports.relative_to(run_root)
    except ValueError as error:
        raise ValueError("reports directory escapes the run directory") from error
    if resolved_reports.parent != run_root:
        raise ValueError("reports directory is not the run-local reports directory")
    return reports_directory


def _publish_report(path: Path, payload: str, *, immutable: bool) -> None:
    """Publish a report without following target/temp symlinks or clobbering formal data."""
    parent_metadata = path.parent.lstat()
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise ValueError(f"report parent directory is unsafe: {path.parent}")
    encoded = payload.encode("utf-8")
    try:
        path.lstat()
    except FileNotFoundError:
        exists = False
    else:
        exists = True
    if immutable and exists:
        if _read_immutable_file_no_follow(path) != encoded:
            raise ValueError(f"sealed formal report differs from recomputed report: {path}")
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        temporary_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(temporary_metadata.st_mode) or temporary_metadata.st_nlink != 1:
            raise ValueError(f"unsafe formal report temporary file: {temporary}")
        written = 0
        while written < len(encoded):
            written += os.write(descriptor, encoded[written:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444 if immutable else 0o644)
        os.close(descriptor)
        descriptor = -1
        if immutable:
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError as error:
                raise ValueError(f"formal report path already exists: {path}") from error
        else:
            os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _formal_queue_has_no_unfinished_work(run_directory: Path) -> bool:
    queue_root = run_directory / ".parallel"
    if not queue_root.exists():
        return True
    try:
        metadata = queue_root.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        return False
    for state in ("pending", "running"):
        state_dir = queue_root / state
        if not state_dir.exists():
            continue
        try:
            state_metadata = state_dir.lstat()
        except OSError:
            return False
        if (
            stat.S_ISLNK(state_metadata.st_mode)
            or not stat.S_ISDIR(state_metadata.st_mode)
            or any(state_dir.iterdir())
        ):
            return False
    return True


def general_post_processing(
    workspace_paths: Union[str, List[str]],
    logger: Optional[logging.Logger],
    *,
    run_directory: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Process all task results and generate a comprehensive report.

    Args:
        workspace_paths: List of workspace directory paths
        logger: Logger instance to use for output
    Returns:
        dict: Report containing statistics and task details
            - total_tasks: Total number of tasks
            - total_score: Sum of all scores
            - compilation_pass_count: Number of tasks that passed compilation
            - compilation_pass_rate: Percentage of tasks that passed compilation
            - correctness_pass_count: Number of tasks that passed correctness tests
            - correctness_pass_rate: Percentage of tasks that passed correctness tests
            - speedup_gt_1_count: Number of tasks with speedup > 1.0
            - speedup_gt_1_rate: Percentage of tasks with speedup > 1.0
            - average_speedup: Average speedup ratio (only valid speedups)
            - task_details: List of detailed information for each task
    """
    logger = _ensure_logger(logger)
    normalized_workspace_paths = _normalize_workspace_paths(workspace_paths)

    resolved_run_directory = Path(run_directory).resolve() if run_directory else None
    if resolved_run_directory is None and normalized_workspace_paths:
        try:
            resolved_run_directory = _get_run_directory(normalized_workspace_paths)
        except Exception as e:
            logger.warning(f"Could not determine run directory: {e}")

    formal_context = (
        _load_formal_cohort(resolved_run_directory)
        if resolved_run_directory is not None
        else None
    )
    formal_campaign = formal_context is not None
    formal_cohort = formal_context["task_names"] if formal_context else None
    task_sources: List[tuple[str, Optional[Path], Dict[str, Any] | None]] = []
    if formal_cohort is not None and resolved_run_directory is not None:
        workspace_map = _formal_workspace_map(
            resolved_run_directory, formal_cohort
        )
        for task_name in formal_cohort:
            workspace = workspace_map.get(task_name)
            failure = (
                _validated_failure_binding(
                    resolved_run_directory, task_name, formal_context
                )
                if workspace is None
                else None
            )
            task_sources.append((task_name, workspace, failure))
        normalized_workspace_paths = [
            str(workspace)
            for _, workspace, _ in task_sources
            if workspace is not None
        ]
        logger.info(
            "Loaded formal cohort from campaign manifest: tasks=%d canonical=%d failed=%d",
            len(formal_cohort),
            len(normalized_workspace_paths),
            len(formal_cohort) - len(normalized_workspace_paths),
        )
    else:
        if resolved_run_directory is not None:
            all_task_paths = _collect_all_tasks_from_run(resolved_run_directory)
            if all_task_paths:
                logger.info(
                    "Collected %d total tasks from run directory (including previously completed)",
                    len(all_task_paths),
                )
                normalized_workspace_paths = all_task_paths
        task_sources = [
            (Path(workspace_path).name, Path(workspace_path), None)
            for workspace_path in normalized_workspace_paths
        ]

    total_tasks = len(task_sources)
    total_score = 0.0
    compilation_pass_count = 0
    correctness_pass_count = 0
    speedup_gt_1_count = 0
    speedup_values = []

    task_details = []

    for task_name, workspace, failure in task_sources:
        task_info = {
            'task_name': task_name,
            'workspace_path': str(workspace) if workspace is not None else None,
            'score': 0.0,
            'pass_compilation': False,
            'pass_correctness': False,
            'speedup_ratio': 0.0,
            'error': None,
            'optimization_summary': '',
            'campaign_status': (
                'failed' if workspace is None else ('canonical' if formal_campaign else 'ordinary')
            ),
            'failure_reasons': [],
            'campaign_evidence_path': None,
            'campaign_evidence_sha256': None,
            'canonical_workspace_manifest_sha256': None,
            'selected_workspace_manifest_sha256': None,
            'terminal_failure_binding_verified': False,
        }

        if workspace is None:
            failure = failure or {
                'reason_codes': ['formal_task_not_canonical'],
                'campaign_evidence_path': None,
                'campaign_evidence_sha256': None,
                'terminal_binding_verified': False,
            }
            task_info['failure_reasons'] = failure['reason_codes']
            task_info['campaign_evidence_path'] = failure['campaign_evidence_path']
            task_info['campaign_evidence_sha256'] = failure['campaign_evidence_sha256']
            task_info['terminal_failure_binding_verified'] = failure[
                'terminal_binding_verified'
            ]
            task_info['error'] = '; '.join(failure['reason_codes'])
            task_details.append(task_info)
            continue

        try:
            result_file = workspace / "task_result.yaml"
            if formal_campaign:
                result_data, lineage = _validate_canonical_lineage(
                    run_directory=resolved_run_directory,
                    task_name=task_name,
                    canonical=workspace,
                    formal=formal_context,
                )
                task_info.update(lineage)
            else:
                with open(result_file, 'r') as f:
                    result_data = yaml.safe_load(f) or {}
            if not isinstance(result_data, dict):
                raise ValueError("task_result.yaml must contain a mapping")

            task_info['task_name'] = result_data.get('task_name', task_name)
            if formal_campaign and task_info['task_name'] != task_name:
                raise ValueError(
                    f"canonical task result name differs from manifest: {task_info['task_name']!r}"
                )
            pass_compilation = result_data.get('pass_compilation', False)
            pass_correctness = result_data.get('pass_correctness', False)
            optimization_summary = result_data.get('optimization_summary', '') or ''
            if formal_campaign and (
                type(pass_compilation) is not bool
                or type(pass_correctness) is not bool
                or pass_compilation is not True
                or pass_correctness is not True
                or not isinstance(optimization_summary, str)
            ):
                raise ValueError("canonical task result has invalid success fields")

            base_execution_time = result_data.get('base_execution_time', 0.0)
            best_optimized_execution_time = result_data.get('best_optimized_execution_time', 0.0)
            speedup_ratio = resolve_speedup_ratio(
                speedup_ratio=result_data.get('speedup_ratio', 0.0),
                base_execution_time=base_execution_time,
                best_optimized_execution_time=best_optimized_execution_time,
            )
            if formal_campaign and (
                not math.isfinite(speedup_ratio) or speedup_ratio <= 0
            ):
                raise ValueError("canonical task result has invalid speedup")
            calculated_score = task_result_scoring(str(workspace))

            # Commit the task and aggregate counters only after every field used
            # by reports/CSV has passed validation. A malformed canonical result
            # therefore remains a zero-score cohort failure.
            task_info['score'] = calculated_score
            task_info['pass_compilation'] = pass_compilation
            task_info['pass_correctness'] = pass_correctness
            task_info['optimization_summary'] = optimization_summary
            task_info['speedup_ratio'] = speedup_ratio
            total_score += calculated_score

            if pass_compilation:
                compilation_pass_count += 1

            if pass_correctness:
                correctness_pass_count += 1

            if pass_compilation and pass_correctness:
                if speedup_ratio > 1.0:
                    speedup_gt_1_count += 1
                if speedup_ratio > 0:
                    speedup_values.append(speedup_ratio)

        except FileNotFoundError as e:
            task_info['error'] = f"task_result.yaml not found: {e}"
            task_info['score'] = 0.0

        except (KeyError, ValueError, TypeError) as e:
            task_info['error'] = f"Invalid or missing data in task_result.yaml: {e}"
            task_info['score'] = 0.0

        except Exception as e:
            task_info['error'] = f"Unexpected error: {e}"
            task_info['score'] = 0.0

        if formal_campaign and task_info['error']:
            task_info['task_name'] = task_name
            task_info['campaign_status'] = 'failed'
            task_info['failure_reasons'] = ['canonical_result_invalid']

        task_details.append(task_info)

    # Calculate rates
    compilation_pass_rate = (compilation_pass_count / total_tasks * 100) if total_tasks > 0 else 0.0
    correctness_pass_rate = (correctness_pass_count / total_tasks * 100) if total_tasks > 0 else 0.0
    speedup_gt_1_rate = (speedup_gt_1_count / total_tasks * 100) if total_tasks > 0 else 0.0

    # Calculate speedup statistics using shared helper
    speed_stats = _compute_speedup_stats(speedup_values)

    canonical_success_count = sum(
        task.get('campaign_status') == 'canonical' for task in task_details
    )
    failed_task_count = (
        total_tasks - canonical_success_count if formal_campaign else 0
    )
    failure_reason_counts: Dict[str, int] = defaultdict(int)
    for task in task_details:
        for reason in task.get('failure_reasons', []):
            failure_reason_counts[reason] += 1

    terminal_task_count = sum(
        task.get('campaign_status') == 'canonical'
        or task.get('terminal_failure_binding_verified') is True
        for task in task_details
    )
    formal_completion_verified = bool(
        formal_campaign
        and terminal_task_count == total_tasks
        and resolved_run_directory is not None
        and _formal_queue_has_no_unfinished_work(resolved_run_directory)
    )
    canonical_workspace_manifests = {
        task['task_name']: task['canonical_workspace_manifest_sha256']
        for task in task_details
        if task.get('campaign_status') == 'canonical'
        and isinstance(task.get('canonical_workspace_manifest_sha256'), str)
    }

    aggregate_result = {
        'total_tasks': total_tasks,
        'total_score': total_score,
        'average_score': total_score / total_tasks if total_tasks > 0 else 0.0,

        'compilation_pass_count': compilation_pass_count,
        'compilation_pass_rate': compilation_pass_rate,

        'correctness_pass_count': correctness_pass_count,
        'correctness_pass_rate': correctness_pass_rate,

        'speedup_gt_1_count': speedup_gt_1_count,
        'speedup_gt_1_rate': speedup_gt_1_rate,

        **speed_stats,
        'valid_speedup_count': len(speedup_values),
        'speedup_population': (
            'canonical_compilation_and_correctness_successes_only'
            if formal_campaign
            else 'compilation_and_correctness_successes_only'
        ),

        'task_details': task_details,
        'formal_campaign': formal_campaign,
        'canonical_success_count': canonical_success_count,
        'failed_task_count': failed_task_count,
        'failure_reason_counts': dict(sorted(failure_reason_counts.items())),
        'terminal_task_count': terminal_task_count,
        'formal_completion_verified': formal_completion_verified,
        'canonical_workspace_manifests': canonical_workspace_manifests,
        'campaign_manifest_sha256': (
            formal_context['campaign_manifest_sha256'] if formal_context else None
        ),
        'comparison_contract_sha256': (
            formal_context['comparison_contract_sha256'] if formal_context else None
        ),
        'ordered_cohort_sha256': (
            formal_context['ordered_cohort_sha256'] if formal_context else None
        ),
    }

    # Aggregate statistics by task type
    task_type_breakdown = _aggregate_by_task_type(task_details)

    # Formal reports are immutable evidence. Publishing a partial projection
    # would permanently poison a long-running campaign because later terminal
    # results necessarily produce different bytes and sealed reports cannot be
    # overwritten. Fail before creating the reports directory or any report.
    if formal_campaign and not formal_completion_verified:
        raise ValueError(
            "formal campaign is not terminal; refusing to publish immutable reports"
        )

    if resolved_run_directory is not None:
        run_metadata = _extract_run_metadata(resolved_run_directory)
        reports_directory = _prepare_reports_directory(resolved_run_directory)
        report_lines = _build_general_report_lines(
            aggregate_result, run_metadata, task_type_breakdown
        )
        report_path = reports_directory / "overall_report.txt"
        _publish_report(
            report_path,
            "\n".join(report_lines) + "\n",
            immutable=formal_campaign,
        )
        logger.info(f"Report written to: {report_path}")

        json_data = {
            'run_timestamp': run_metadata.get('timestamp', 'unknown'),
            'agent': run_metadata.get('agent', 'unknown'),
            'target_gpu': run_metadata.get('target_gpu', 'unknown'),
            'overall': {
                'total_tasks': aggregate_result['total_tasks'],
                'total_score': aggregate_result['total_score'],
                'average_score': aggregate_result['average_score'],
                'compilation_pass_count': aggregate_result['compilation_pass_count'],
                'compilation_pass_rate': aggregate_result['compilation_pass_rate'],
                'correctness_pass_count': aggregate_result['correctness_pass_count'],
                'correctness_pass_rate': aggregate_result['correctness_pass_rate'],
                'speedup_gt_1_count': aggregate_result['speedup_gt_1_count'],
                'speedup_gt_1_rate': aggregate_result['speedup_gt_1_rate'],
                'average_speedup': aggregate_result['average_speedup'],
                'median_speedup': aggregate_result.get('median_speedup', 0.0),
                'std_dev_speedup': aggregate_result.get('std_dev_speedup', 0.0),
                'p25_speedup': aggregate_result.get('p25_speedup', 0.0),
                'p75_speedup': aggregate_result.get('p75_speedup', 0.0),
                'p90_speedup': aggregate_result.get('p90_speedup', 0.0),
                'valid_speedup_count': aggregate_result['valid_speedup_count'],
                'speedup_population': aggregate_result['speedup_population'],
                'formal_campaign': formal_campaign,
                'canonical_success_count': canonical_success_count,
                'failed_task_count': failed_task_count,
                'failure_reason_counts': aggregate_result['failure_reason_counts'],
                'campaign_manifest_sha256': aggregate_result['campaign_manifest_sha256'],
                'comparison_contract_sha256': aggregate_result['comparison_contract_sha256'],
                'ordered_cohort_sha256': aggregate_result['ordered_cohort_sha256'],
                'terminal_task_count': terminal_task_count,
                'formal_completion_verified': formal_completion_verified,
            },
            'task_types': task_type_breakdown,
            'formal_evidence': {
                'schema': 'aka.formal-report-evidence/v1' if formal_campaign else None,
                'campaign_manifest_sha256': aggregate_result['campaign_manifest_sha256'],
                'comparison_contract_sha256': aggregate_result['comparison_contract_sha256'],
                'ordered_cohort_sha256': aggregate_result['ordered_cohort_sha256'],
                'completion_verified': formal_completion_verified,
                'terminal_task_count': terminal_task_count,
                'canonical_workspace_manifests': canonical_workspace_manifests,
            },
            'failed_tasks': [
                {
                    'task_name': task['task_name'],
                    'reason_codes': task['failure_reasons'],
                    'campaign_evidence_path': task['campaign_evidence_path'],
                    'campaign_evidence_sha256': task['campaign_evidence_sha256'],
                }
                for task in task_details
                if task.get('campaign_status') == 'failed'
            ],
        }
        json_path = reports_directory / "task_type_breakdown.json"
        _publish_report(
            json_path,
            json.dumps(json_data, indent=2, sort_keys=True) + "\n",
            immutable=formal_campaign,
        )
        logger.info(f"Task type breakdown JSON written to: {json_path}")
    else:
        run_metadata = None
        reports_directory = None

    general_log_report(aggregate_result, logger, run_metadata, task_type_breakdown)
    export_task_results_csv(
        task_details,
        normalized_workspace_paths,
        logger,
        reports_directory,
        immutable=formal_campaign,
    )
    return aggregate_result


def export_task_results_csv(
    task_details: List[Dict[str, Any]],
    workspace_paths: List[str],
    logger: logging.Logger,
    reports_directory: Optional[Path] = None,
    *,
    immutable: bool = False,
) -> None:
    """
    Export per-task summary as CSV under the reports directory.

    CSV columns:
      - Task Name
      - Task Type
      - Score
      - Speedup
      - Optimization_summary
    """
    if not workspace_paths and reports_directory is None:
        logger.warning("CSV export skipped: empty workspace_paths")
        return

    # Use reports directory if provided, otherwise fall back to run directory
    if reports_directory:
        csv_path = reports_directory / "overall_summary.csv"
    else:
        # Fallback: use run directory (parent of first workspace)
        run_directory = Path(workspace_paths[0]).resolve().parent
        csv_path = run_directory / "task_results_summary.csv"

    rows: List[Dict[str, Any]] = []
    for task in task_details:
        task_name = task.get("task_name", "")
        task_type = (
            task_name.split("/", 1)[0]
            if isinstance(task_name, str) and "/" in task_name
            else ""
        )
        score = task.get("score", 0.0) if isinstance(task.get("score", 0.0), (int, float)) else 0.0
        speedup = task.get("speedup_ratio", 0.0) if isinstance(task.get("speedup_ratio", 0.0), (int, float)) else 0.0
        optimization_summary = task.get("optimization_summary", "") or ""

        rows.append({
            "Task Name": task_name,
            "Task Type": task_type,
            "Score": f"{float(score):.4f}",
            "Speedup": f"{float(speedup):.4f}",
            "Optimization_summary": optimization_summary.strip(),
            "Campaign Status": task.get("campaign_status", "ordinary"),
            "Failure Reasons": ";".join(task.get("failure_reasons", [])),
            "Campaign Evidence": task.get("campaign_evidence_path") or "",
        })

    output = io.StringIO(newline="")
    fieldnames = [
        "Task Name",
        "Task Type",
        "Score",
        "Speedup",
        "Optimization_summary",
        "Campaign Status",
        "Failure Reasons",
        "Campaign Evidence",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    _publish_report(csv_path, output.getvalue(), immutable=immutable)

    logger.info(f"CSV report generated: {csv_path}")


if __name__ == "__main__":
    
    # manually generate report
    workspace_path = "workspace_MI300_claude_code"
    general_post_processing(workspace_path, logger = None)
