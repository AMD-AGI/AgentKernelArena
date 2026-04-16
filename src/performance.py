# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Performance measurement and parsing for evaluator.
"""
import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from .testcases import TestCaseResult, parse_test_cases_from_json, parse_test_cases_from_stdout
from .evaluator_utils import run_command

# Default for performance_command subprocess (CMake benchmarks can be slow)
_DEFAULT_PERFORMANCE_COMMAND_TIMEOUT_S = 3600


def find_performance_report_files(workspace: Path, task_type: Optional[str] = None) -> List[Path]:
    """
    Find potential performance report files based on task type and common patterns.
    
    Args:
        workspace: Workspace directory
        task_type: Task type (e.g., 'triton2triton', 'hip2hip')
        
    Returns:
        List of potential report file paths to check
    """
    report_files = []
    
    # Common report file locations
    common_paths = [
        workspace / "build" / "performance_report.json",
        workspace / "performance_report.json",
        workspace / "build" / "perf_report.json",
        workspace / "perf_report.json",
        workspace / "perf" / "benchmark_results.json",
    ]
    
    
    # Add common paths
    report_files.extend(common_paths)
    
    # Filter to only existing files
    return [f for f in report_files if f.exists()]


def parse_execution_time_from_json(
    report_file: Path, 
    logger: Optional[logging.Logger] = None,
    is_baseline: bool = False,
    task_type: Optional[str] = None
) -> float:
    """
    Parse execution time from JSON report file.
    
    Uses the same extraction logic as parse_test_cases_from_json but for single values.
    
    Args:
        report_file: Path to JSON report file
        logger: Optional logger
        is_baseline: If True, use ori_time for torch2hip; if False, use opt_time
        task_type: Task type (e.g., 'torch2hip')
        
    Returns:
        Execution time in milliseconds, or 0.0 if not found
    """
    from .testcases import _extract_time_from_dict
    
    log = logger or logging.getLogger(__name__)
    
    try:
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        # If it's an array, try first element
        if isinstance(report, list) and len(report) > 0:
            report = report[0]
        
        time_ms, _ = _extract_time_from_dict(report, is_baseline, task_type)
        
        if time_ms > 0:
            return time_ms
        
        log.warning(f"No recognized time key found in {report_file}. Available keys: {list(report.keys())}")
        return 0.0
        
    except json.JSONDecodeError as e:
        log.warning(f"Failed to parse JSON from {report_file}: {e}")
        return 0.0
    except Exception as e:
        log.warning(f"Error reading report file {report_file}: {e}")
        return 0.0


def parse_execution_time_from_stdout(output: str, logger: Optional[logging.Logger] = None) -> float:
    """
    Parse execution time from command stdout/stderr text.
    
    Args:
        output: Command output text
        logger: Optional logger
        
    Returns:
        Execution time in milliseconds, or 0.0 if not found
    """
    log = logger or logging.getLogger(__name__)
    
    # Patterns to match (in order of specificity)
    patterns = [
        # Specific patterns with "Performance:" prefix
        (r'Performance:\s*([0-9.]+)\s*ms', 1.0),  # "Performance: 123.45 ms"
        (r'Performance:\s*([0-9.]+)\s*s(?:econds?)?', 1000.0),  # "Performance: 1.23 s"
        
        # Generic time patterns
        (r'execution[_\s]time[:\s]+([0-9.]+)\s*ms', 1.0),  # "execution time: 123.45 ms"
        (r'execution[_\s]time[:\s]+([0-9.]+)\s*s(?:econds?)?', 1000.0),  # "execution time: 1.23 s"
        
        (r'elapsed[_\s]time[:\s]+([0-9.]+)\s*ms', 1.0),  # "elapsed time: 123.45 ms"
        (r'elapsed[_\s]time[:\s]+([0-9.]+)\s*s(?:econds?)?', 1000.0),  # "elapsed time: 1.23 s"
        
        (r'avg[_\s]time[:\s]+([0-9.]+)\s*ms', 1.0),  # "avg time: 123.45 ms"
        (r'avg[_\s]time[:\s]+([0-9.]+)\s*s(?:econds?)?', 1000.0),  # "avg time: 1.23 s"
        
        # Generic number + unit patterns (less specific, try last)
        (r'([0-9.]+)\s*ms\b', 1.0),  # "123.45 ms"
        (r'([0-9.]+)\s*s(?:econds?)?\b', 1000.0),  # "1.23 s" or "1.23 seconds"
    ]
    
    for pattern, multiplier in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            time_val = float(match.group(1))
            result = time_val * multiplier
            log.debug(f"Parsed execution time from stdout: {result:.4f} ms (pattern: {pattern})")
            return result
    
    log.debug("Could not parse execution time from stdout")
    return 0.0


def parse_execution_time(
    output: str,
    workspace: Path,
    task_type: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    is_baseline: bool = False
) -> float:
    """
    Parse execution time from command output or report files.
    
    Tries multiple strategies:
    1. Look for JSON report files in common locations
    2. Parse from stdout/stderr text
    3. Default to 0.0 if not found
    
    Args:
        output: Command stdout/stderr
        workspace: Workspace directory to search for report files
        task_type: Task type (e.g., 'triton2triton', 'hip2hip')
        logger: Optional logger
        
    Returns:
        Execution time in milliseconds (converted from seconds if needed)
    """
    log = logger or logging.getLogger(__name__)
    
    # Strategy 1: Check for JSON report files
    report_files = find_performance_report_files(workspace, task_type)
    for report_file in report_files:
        if report_file.suffix == '.json':
            time_val = parse_execution_time_from_json(report_file, logger, is_baseline, task_type)
            if time_val > 0:
                log.info(f"Found execution time in {report_file}: {time_val:.4f} ms")
                return time_val
        # Note: .pt files (PyTorch tensors) would need special handling if needed
    
    # Strategy 2: Parse from output text
    # time_val = parse_execution_time_from_stdout(output, logger)
    # if time_val > 0:
    #     log.info(f"Parsed execution time from stdout: {time_val:.4f} ms")
    #     return time_val
    
    log.warning("Could not parse execution time from any source")
    return 0.0


def parse_all_test_cases(
    output: str,
    workspace: Path,
    task_type: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    is_baseline: bool = False
) -> List[TestCaseResult]:
    """
    Parse all test case results from command output and report files.
    
    Args:
        output: Command stdout/stderr
        workspace: Workspace directory
        task_type: Task type
        logger: Optional logger
        is_baseline: If True, use ori_time for torch2hip; if False, use opt_time
        
    Returns:
        List of TestCaseResult objects
    """
    log = logger or logging.getLogger(__name__)
    all_test_cases = []
    
    # Strategy 1: Check JSON report files for multiple test cases
    report_files = find_performance_report_files(workspace, task_type)
    for report_file in report_files:
        if report_file.suffix == '.json':
            test_cases = parse_test_cases_from_json(report_file, logger, is_baseline, task_type)
            if test_cases:
                all_test_cases.extend(test_cases)
                break  # Use first file that has test cases
    
    # Strategy 2: Parse from stdout if no test cases found in files
    if not all_test_cases:
        test_cases = parse_test_cases_from_stdout(output, logger)
        if test_cases:
            all_test_cases.extend(test_cases)
    
    # If still no test cases, try to parse as single result
    if not all_test_cases:
        single_time = parse_execution_time(output, workspace, task_type, logger, is_baseline)
        if single_time > 0:
            all_test_cases.append(TestCaseResult(
                test_case_id="test_case_0",
                execution_time_ms=single_time
            ))
    
    return all_test_cases


def measure_performance(
    workspace: Path,
    task_config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    is_baseline: bool = False
) -> List[TestCaseResult]:
    """
    Measure kernel execution time for all test cases.
    
    Args:
        workspace: Workspace directory
        task_config: Task configuration dict
        logger: Optional logger
        is_baseline: If True, use ori_time for torch2hip; if False, use opt_time
        
    Returns:
        List of TestCaseResult objects (empty list if measurement failed)
    """
    log = logger or logging.getLogger(__name__)
    performance_commands = task_config.get('performance_command', [])
    task_type = task_config.get('task_type')
    
    if not performance_commands:
        log.warning("No performance_command found in task config")
        return []

    perf_timeout = int(
        task_config.get("performance_timeout", _DEFAULT_PERFORMANCE_COMMAND_TIMEOUT_S)
    )

    for cmd in performance_commands:
        if is_baseline and task_type == 'torch2hip':
            cmd = cmd + " --baseline_only"
        success, stdout, stderr = run_command(cmd, workspace, timeout=perf_timeout, logger=log)
        
        # Combine stdout and stderr for parsing
        combined_output = stdout + stderr
        
        if success:
            # Try to parse all test cases (will check report files and stdout)
            test_cases = parse_all_test_cases(combined_output, workspace, task_type, logger, is_baseline)
            if test_cases:
                log.info(f"Measured {len(test_cases)} test case(s)")
                return test_cases
            else:
                log.warning("Could not parse test cases from performance command output")
                log.debug(f"Command output (first 500 chars): {combined_output[:500]}")
        else:
            log.warning(f"Performance command failed: {stderr}")
    
    return []


def measure_baseline(
    workspace: Path,
    task_config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> List[TestCaseResult]:
    """
    Measure baseline execution time for all test cases before optimization.
    
    This should be called BEFORE the agent modifies the kernel.
    The original kernel should still be in place.
    Results are saved to baseline_perf.yaml.
    
    Args:
        workspace: Workspace directory
        task_config: Task configuration dict
        logger: Optional logger
        
    Returns:
        List of TestCaseResult objects (empty list if measurement failed)
    """
    from .testcases import save_performance_results
    
    log = logger or logging.getLogger(__name__)
    log.info("Measuring baseline performance...")
    
    baseline_cases = measure_performance(workspace, task_config, logger, is_baseline=True)
    
    if baseline_cases:
        # Save baseline results
        save_performance_results(baseline_cases, workspace, "baseline_perf.yaml", logger)
        total_time = sum(c.execution_time_ms for c in baseline_cases)
        avg_time = total_time / len(baseline_cases)
        log.info(f"Baseline: {len(baseline_cases)} test case(s), average time: {avg_time:.4f} ms")
    else:
        log.warning("Failed to measure baseline execution time")
    
    return baseline_cases

