# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import yaml
import logging
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
try:
    from src.score import resolve_speedup_ratio, task_result_scoring
except ModuleNotFoundError:
    # Allow direct execution: `python src/postprocessing.py`
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.score import resolve_speedup_ratio, task_result_scoring


def _build_general_report_lines(aggregate_result: Dict[str, Any]) -> List[str]:
    """Build report lines shared by logger output and fallback txt output."""
    lines = [
        "=" * 80,
        "AgentKernelArena Task Results Report",
        "=" * 80,
        "Overall Statistics:",
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
        f"  Valid Speedup Count:   {aggregate_result['valid_speedup_count']}",
        "Task Details:",
        "-" * 80,
    ]

    for task in aggregate_result["task_details"]:
        status = "PASS" if task["pass_correctness"] else ("PARTIAL" if task["pass_compilation"] else "FAIL")
        lines.append(
            f"{status:<8} {task['task_name']:<40} Score: {task['score']:>6.1f}  Speedup: {task['speedup_ratio']:.2f}x"
        )
        if task["error"]:
            lines.append(f"         Error: {task['error']}")

    lines.append("=" * 80)
    return lines


def _write_report_txt_if_log_empty(
    report_lines: List[str], workspace_paths: List[str], logger: logging.Logger
) -> None:
    """Write fallback txt report when logger file output is empty."""
    if not workspace_paths:
        return

    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    log_is_empty = not file_handlers

    for handler in file_handlers:
        base_filename = getattr(handler, "baseFilename", None)
        if not base_filename:
            continue
        log_path = Path(base_filename)
        if log_path.exists() and log_path.stat().st_size > 0:
            log_is_empty = False
            break

    if not log_is_empty:
        return

    workspace_root = Path(workspace_paths[0]).resolve().parent
    txt_path = workspace_root / "task_results_report.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    logger.info(f"Log was empty; wrote fallback text report: {txt_path}")


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



def general_log_report(aggregate_result: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Log a formatted report using the provided logger.

    Args:
        aggregate_result: Report dictionary from post_processing()
        logger: Logger instance to use for output
    """
    for line in _build_general_report_lines(aggregate_result):
        logger.info(line)


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

    # Calculate average speedup (only for valid speedups)
    average_speedup = (sum(speedup_values) / len(speedup_values)) if speedup_values else 0.0

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

        'average_speedup': average_speedup,
        'valid_speedup_count': len(speedup_values),

        'task_details': task_details
    }

    general_log_report(aggregate_result, logger)
    export_task_results_csv(task_details, normalized_workspace_paths, logger)
    _write_report_txt_if_log_empty(_build_general_report_lines(aggregate_result), normalized_workspace_paths, logger)


def export_task_results_csv(
    task_details: List[Dict[str, Any]],
    workspace_paths: List[str],
    logger: logging.Logger
) -> None:
    """
    Export per-task summary as CSV under the workspace root directory.

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

    # All task workspaces are expected to be siblings under one workspace root.
    workspace_root = Path(workspace_paths[0]).resolve().parent
    csv_path = workspace_root / "task_results_summary.csv"

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
