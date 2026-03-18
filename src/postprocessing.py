# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import yaml
import logging
import csv
import json
import sys
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
try:
    from src.score import resolve_speedup_ratio, task_result_scoring
except ModuleNotFoundError:
    # Allow direct execution: `python src/postprocessing.py`
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.score import resolve_speedup_ratio, task_result_scoring


def _build_general_report_lines(
    aggregate_result: Dict[str, Any], 
    run_metadata: Optional[Dict[str, str]] = None,
    task_type_breakdown: Optional[Dict[str, Dict[str, Any]]] = None
) -> List[str]:
    """Build report lines shared by logger output and fallback txt output."""
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
        f"  Average Speedup:       {aggregate_result['average_speedup']:.2f}x",
        f"  Median Speedup:        {aggregate_result.get('median_speedup', 0.0):.2f}x",
        f"  Std Dev Speedup:       {aggregate_result.get('std_dev_speedup', 0.0):.2f}x",
        f"  P25/P75/P90 Speedup:   {aggregate_result.get('p25_speedup', 0.0):.2f}x / {aggregate_result.get('p75_speedup', 0.0):.2f}x / {aggregate_result.get('p90_speedup', 0.0):.2f}x",
        f"  Valid Speedup Count:   {aggregate_result['valid_speedup_count']}",
    ])
    
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
            lines.append(f"    Average Speedup:     {stats['average_speedup']:.2f}x")
            lines.append(f"    Median Speedup:      {stats.get('median_speedup', 0.0):.2f}x")
            lines.append(f"    Std Dev Speedup:     {stats.get('std_dev_speedup', 0.0):.2f}x")
            lines.append(f"    P25/P75/P90 Speedup: {stats.get('p25_speedup', 0.0):.2f}x / {stats.get('p75_speedup', 0.0):.2f}x / {stats.get('p90_speedup', 0.0):.2f}x")
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
        f"  Overall Average Speedup:  {aggregate_result['average_speedup']:.2f}x",
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
        status = "PASS" if task["pass_correctness"] else ("PARTIAL" if task["pass_compilation"] else "FAIL")
        lines.append(
            f"{status:<8} {task['task_name']:<40} Score: {task['score']:>6.1f}  Speedup: {task['speedup_ratio']:.2f}x"
        )
        if task["error"]:
            lines.append(f"         Error: {task['error']}")

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
    Compute speedup statistics: average, median, std dev, and percentiles (P25, P75, P90).

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

    if len(speedup_values) == 1:
        p25 = p75 = p90 = speedup_values[0]
    else:
        try:
            # statistics.quantiles with n=100 gives 99 cut points; index 24 = P25, etc.
            quantile_cuts = statistics.quantiles(speedup_values, n=100)
            p25 = quantile_cuts[24]
            p75 = quantile_cuts[74]
            p90 = quantile_cuts[89]
        except (statistics.StatisticsError, IndexError, ValueError):
            p25 = p75 = p90 = 0.0

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


def general_post_processing(
    workspace_paths: Union[str, List[str]], logger: Optional[logging.Logger]
) -> None:
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
    
    # If we have workspace paths, try to detect the run directory and collect ALL tasks
    # This ensures that when resuming, we include previously completed tasks in the report
    if normalized_workspace_paths:
        try:
            run_directory = _get_run_directory(normalized_workspace_paths)
            # Collect all tasks from the run directory (including previously completed ones)
            all_task_paths = _collect_all_tasks_from_run(run_directory)
            if all_task_paths:
                logger.info(f"Collected {len(all_task_paths)} total tasks from run directory (including previously completed)")
                normalized_workspace_paths = all_task_paths
        except Exception as e:
            logger.warning(f"Could not collect all tasks from run directory: {e}. Using provided workspace paths only.")
    total_tasks = len(normalized_workspace_paths)
    total_score = 0.0
    compilation_pass_count = 0
    correctness_pass_count = 0
    speedup_gt_1_count = 0
    speedup_values = []

    task_details = []

    for workspace_path in normalized_workspace_paths:
        workspace = Path(workspace_path)
        task_name = workspace.name  # Get task folder name

        task_info = {
            'task_name': task_name,
            'workspace_path': str(workspace_path),
            'score': 0.0,
            'pass_compilation': False,
            'pass_correctness': False,
            'speedup_ratio': 0.0,
            'error': None
        }

        try:
            # Try to calculate score (this will read task_result.yaml and update it with score)
            calculated_score = task_result_scoring(str(workspace_path))
            task_info['score'] = calculated_score

            # Read the task_result.yaml to get detailed information
            result_file = workspace / "task_result.yaml"
            with open(result_file, 'r') as f:
                result_data = yaml.safe_load(f)

            # Extract information
            task_info['task_name'] = result_data.get('task_name', task_name)
            task_info['pass_compilation'] = result_data.get('pass_compilation', False)
            task_info['pass_correctness'] = result_data.get('pass_correctness', False)

            base_execution_time = result_data.get('base_execution_time', 0.0)
            best_optimized_execution_time = result_data.get('best_optimized_execution_time', 0.0)
            task_info['speedup_ratio'] = resolve_speedup_ratio(
                speedup_ratio=result_data.get('speedup_ratio', 0.0),
                base_execution_time=base_execution_time,
                best_optimized_execution_time=best_optimized_execution_time,
            )

            # Update counters
            total_score += calculated_score

            if task_info['pass_compilation']:
                compilation_pass_count += 1

            if task_info['pass_correctness']:
                correctness_pass_count += 1

            # Check speedup (only if both compilation and correctness passed)
            if task_info['pass_compilation'] and task_info['pass_correctness']:
                speedup = task_info['speedup_ratio']
                if speedup > 1.0:
                    speedup_gt_1_count += 1
                if speedup > 0:  # Only include valid speedups
                    speedup_values.append(speedup)

        except FileNotFoundError as e:
            task_info['error'] = f"task_result.yaml not found: {e}"
            task_info['score'] = 0.0

        except (KeyError, ValueError, TypeError) as e:
            task_info['error'] = f"Invalid or missing data in task_result.yaml: {e}"
            task_info['score'] = 0.0

        except Exception as e:
            task_info['error'] = f"Unexpected error: {e}"
            task_info['score'] = 0.0

        task_details.append(task_info)

    # Calculate rates
    compilation_pass_rate = (compilation_pass_count / total_tasks * 100) if total_tasks > 0 else 0.0
    correctness_pass_rate = (correctness_pass_count / total_tasks * 100) if total_tasks > 0 else 0.0
    speedup_gt_1_rate = (speedup_gt_1_count / total_tasks * 100) if total_tasks > 0 else 0.0

    # Calculate speedup statistics using shared helper
    speed_stats = _compute_speedup_stats(speedup_values)

    # Generate aggregate_result
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

        'task_details': task_details
    }

    # Aggregate statistics by task type
    task_type_breakdown = _aggregate_by_task_type(task_details)

    # Determine run directory and create reports subdirectory
    try:
        run_directory = _get_run_directory(normalized_workspace_paths)
        run_metadata = _extract_run_metadata(run_directory)
        
        # Create reports subdirectory
        reports_directory = run_directory / "reports"
        reports_directory.mkdir(parents=True, exist_ok=True)
        
        # Write overall_report.txt to reports directory
        report_lines = _build_general_report_lines(aggregate_result, run_metadata, task_type_breakdown)
        report_path = reports_directory / "overall_report.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines) + "\n")
        logger.info(f"Report written to: {report_path}")
        
        # Write task_type_breakdown.json
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
                'valid_speedup_count': aggregate_result['valid_speedup_count']
            },
            'task_types': task_type_breakdown
        }
        json_path = reports_directory / "task_type_breakdown.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"Task type breakdown JSON written to: {json_path}")
        
    except Exception as e:
        logger.warning(f"Could not determine run directory or create reports: {e}")
        run_metadata = None
        reports_directory = None
        task_type_breakdown = None
    
    # Log report
    general_log_report(aggregate_result, logger, run_metadata, task_type_breakdown)
    
    # Export CSV to reports directory
    export_task_results_csv(task_details, normalized_workspace_paths, logger, reports_directory)


def export_task_results_csv(
    task_details: List[Dict[str, Any]],
    workspace_paths: List[str],
    logger: logging.Logger,
    reports_directory: Optional[Path] = None
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
    if not workspace_paths:
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
        workspace = Path(task.get("workspace_path", ""))
        result_file = workspace / "task_result.yaml"

        task_name = task.get("task_name", workspace.name)
        task_type = (
            task_name.split("/", 1)[0]
            if isinstance(task_name, str) and "/" in task_name
            else ""
        )
        score = task.get("score", 0.0) if isinstance(task.get("score", 0.0), (int, float)) else 0.0
        speedup = task.get("speedup_ratio", 0.0) if isinstance(task.get("speedup_ratio", 0.0), (int, float)) else 0.0
        optimization_summary = ""

        # If task_result.yaml is missing or invalid, force score/speedup to 0.
        if result_file.exists():
            try:
                with open(result_file, "r") as f:
                    result_data = yaml.safe_load(f) or {}
                optimization_summary = result_data.get("optimization_summary", "") or ""
                task_name = result_data.get("task_name", task_name)
                if "/" in task_name:
                    task_type = task_name.split("/", 1)[0]
            except Exception:
                score = 0.0
                speedup = 0.0
                optimization_summary = ""
        else:
            score = 0.0
            speedup = 0.0

        rows.append({
            "Task Name": task_name,
            "Task Type": task_type,
            "Score": f"{float(score):.4f}",
            "Speedup": f"{float(speedup):.4f}",
            "Optimization_summary": optimization_summary.strip(),
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Task Name", "Task Type", "Score", "Speedup", "Optimization_summary"]
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"CSV report generated: {csv_path}")


if __name__ == "__main__":
    
    # manually generate report
    workspace_path = "workspace_MI300_claude_code"
    general_post_processing(workspace_path, logger = None)
