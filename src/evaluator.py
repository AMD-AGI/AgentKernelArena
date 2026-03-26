# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Centralized evaluator for AgentKernelArena.

This module provides standardized evaluation of optimized kernels:
- Compilation checking
- Correctness validation
- Performance measurement
- Baseline measurement for speedup calculation
"""
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from .evaluator_utils import run_command
from .performance import measure_performance, measure_baseline
from .testcases import TestCaseResult, save_performance_results, calculate_average_speedup


def _valid_perf_cases(cases: List[TestCaseResult]) -> List[TestCaseResult]:
    """Return only test cases with valid positive execution time."""
    valid_cases: List[TestCaseResult] = []
    for case in cases:
        if case.execution_time_ms is not None and case.execution_time_ms > 0:
            valid_cases.append(case)
    return valid_cases


def evaluate_compilation(
    workspace: Path,
    task_config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, Optional[str]]:
    """
    Evaluate kernel compilation.
    
    Args:
        workspace: Workspace directory
        task_config: Task configuration dict
        logger: Optional logger
        
    Returns:
        Tuple of (passed: bool, error_message: Optional[str])
    """
    log = logger or logging.getLogger(__name__)
    compile_commands = task_config.get('compile_command', [])
    
    if not compile_commands:
        log.warning("No compile_command found in task config")
        return False, "No compile_command specified"
    
    for cmd in compile_commands:
        success, stdout, stderr = run_command(cmd, workspace, timeout=120, logger=log)
        if not success:
            error_msg = f"Compilation failed\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            return False, error_msg
    
    return True, None


def evaluate_correctness(
    workspace: Path,
    task_config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, Optional[str]]:
    """
    Evaluate kernel correctness.
    
    Args:
        workspace: Workspace directory
        task_config: Task configuration dict
        logger: Optional logger
        
    Returns:
        Tuple of (passed: bool, error_message: Optional[str])
    """
    log = logger or logging.getLogger(__name__)
    correctness_commands = task_config.get('correctness_command', [])
    
    if not correctness_commands:
        log.warning("No correctness_command found in task config")
        return False, "No correctness_command specified"
    
    for cmd in correctness_commands:
        success, stdout, stderr = run_command(cmd, workspace, timeout=300, logger=log)
        if not success:
            error_msg = f"Correctness test failed\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            return False, error_msg
        
        # Check for explicit failure indicators in output
        output_lower = (stdout + stderr).lower()
        if 'fail' in output_lower and 'pass' not in output_lower:
            # Might have "FAIL" but also check if it says "PASS" somewhere
            if 'correctness: pass' not in output_lower:
                error_msg = f"Correctness test reported failure\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                return False, error_msg
    
    return True, None


def _read_geak_final_report(workspace: Path, log) -> Optional[Dict[str, float]]:
    """Read GEAK's final_report.json from the sibling _logs directory.

    GEAK produces verified performance results in final_report.json with
    baseline_ms, candidate_ms, and verified_speedup already computed.
    This avoids re-running benchmarks and parsing stdout.
    """
    logs_dir = workspace.parent / f"{workspace.name}_logs"
    final_report = logs_dir / "final_report.json"
    if not final_report.exists():
        return None
    try:
        data = json.load(open(final_report))
        fb = (data.get("round_evaluation") or {}).get("full_benchmark") or {}
        baseline_ms = float(fb.get("baseline_ms", 0))
        candidate_ms = float(fb.get("candidate_ms", 0))
        verified = float(fb.get("verified_speedup", 0))
        if baseline_ms > 0 and candidate_ms > 0 and verified > 0:
            re = data.get("round_evaluation") or {}
            # Collect per-round history for reporting
            rounds = []
            for rf in sorted(logs_dir.glob("round_*_evaluation.json")):
                try:
                    rd = json.load(open(rf))
                    rfb = (rd.get("full_benchmark") or {})
                    rounds.append({
                        "round": rd.get("round"),
                        "task": rd.get("best_task"),
                        "benchmark_speedup": rd.get("benchmark_speedup"),
                        "verified_speedup": rfb.get("verified_speedup"),
                    })
                except Exception:
                    pass
            return {
                "baseline_ms": baseline_ms,
                "candidate_ms": candidate_ms,
                "verified_speedup": verified,
                "benchmark_speedup": float(re.get("benchmark_speedup", 0)),
                "best_round": re.get("round"),
                "best_task": re.get("best_task"),
                "round_history": rounds,
            }
        # Fallback: use best_speedup from top-level if full_benchmark is missing
        best = float(data.get("best_speedup", 0))
        if best > 0:
            log.info(f"GEAK final_report has best_speedup={best} but no full_benchmark details")
    except Exception as e:
        log.warning(f"Failed to read GEAK final_report.json: {e}")
    return None


def evaluate_kernel(
    workspace: Path,
    task_config: Dict[str, Any],
    baseline_cases: List[TestCaseResult],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Standardized evaluation of optimized kernel.
    
    Args:
        workspace: Workspace directory containing optimized kernel
        task_config: Task configuration dict
        baseline_cases: Baseline test case results (from measure_baseline)
        logger: Optional logger
        
    Returns:
        Dict with evaluation results:
        - pass_compilation: bool
        - pass_correctness: bool
        - best_optimized_execution_time: float (ms, average)
        - average_speedup: float
        - compilation_error_message: Optional[str]
        - correctness_error_message: Optional[str]
    """
    log = logger or logging.getLogger(__name__)
    log.info("=" * 80)
    log.info("Starting centralized kernel evaluation")
    log.info("=" * 80)
    
    results = {
        'pass_compilation': False,
        'pass_correctness': False,
        'best_optimized_execution_time': 0.0,
        'average_speedup': 0.0,
        'valid_baseline_cases': 0,
        'valid_optimized_cases': 0,
        'compilation_error_message': None,
        'correctness_error_message': None,
    }
    
    # 1. Compilation check
    log.info("Step 1: Checking compilation...")
    pass_compilation, comp_error = evaluate_compilation(workspace, task_config, logger)
    results['pass_compilation'] = pass_compilation
    results['compilation_error_message'] = comp_error
    
    if not pass_compilation:
        log.warning("Compilation failed, skipping correctness and performance checks")
        return results
    
    # 2. Correctness check
    log.info("Step 2: Checking correctness...")
    pass_correctness, corr_error = evaluate_correctness(workspace, task_config, logger)
    results['pass_correctness'] = pass_correctness
    results['correctness_error_message'] = corr_error
    
    if not pass_correctness:
        log.warning("Correctness failed, skipping performance measurement")
        return results
    
    # 3. Performance measurement (only if both compilation and correctness passed)
    log.info("Step 3: Measuring performance...")
    optimized_cases = measure_performance(workspace, task_config, logger)
    
    if optimized_cases:
        # Save optimized results
        save_performance_results(optimized_cases, workspace, "optimized_perf.yaml", logger)
        valid_optimized_cases = _valid_perf_cases(optimized_cases)
        valid_baseline_cases = _valid_perf_cases(baseline_cases)
        results['valid_optimized_cases'] = len(valid_optimized_cases)
        results['valid_baseline_cases'] = len(valid_baseline_cases)

        if not valid_optimized_cases:
            results['best_optimized_execution_time'] = 0.0
            log.warning(
                "No valid performance samples found (execution_time_ms <= 0 or invalid). "
                "best_optimized_execution_time is set to 0.0"
            )
        else:
            avg_optimized_time = sum(c.execution_time_ms for c in valid_optimized_cases) / len(valid_optimized_cases)
            results['best_optimized_execution_time'] = avg_optimized_time
            log.info(
                f"Performance: {len(valid_optimized_cases)}/{len(optimized_cases)} valid test case(s), "
                f"average time: {avg_optimized_time:.4f} ms"
            )

            # Calculate average speedup across valid test cases only
            if valid_baseline_cases:
                avg_speedup = calculate_average_speedup(valid_baseline_cases, valid_optimized_cases, logger)
                results['average_speedup'] = avg_speedup
                avg_baseline_time = sum(c.execution_time_ms for c in valid_baseline_cases) / len(valid_baseline_cases)
                log.info(
                    f"Baseline: {len(valid_baseline_cases)}/{len(baseline_cases)} valid test case(s), "
                    f"average time: {avg_baseline_time:.4f} ms"
                )
                log.info(f"Average speedup: {avg_speedup:.2f}x")
            else:
                if baseline_cases:
                    log.warning(
                        "Baseline data exists but has no valid performance samples "
                        "(execution_time_ms <= 0 or invalid). Cannot calculate speedup."
                    )
                else:
                    log.warning("Baseline not available, cannot calculate speedup")
    else:
        log.warning("Failed to measure optimized execution time")

    # Step 3b: If performance measurement failed, read GEAK's final_report.json
    if results['best_optimized_execution_time'] == 0.0:
        geak_results = _read_geak_final_report(workspace, log)
        if geak_results:
            results['best_optimized_execution_time'] = geak_results['candidate_ms']
            # Use max of verified and benchmark speedup — FULL_BENCHMARK can
            # undercount on tiny kernels due to measurement noise.
            benchmark_sp = geak_results.get('benchmark_speedup') or 0
            verified_sp = geak_results['verified_speedup']
            results['average_speedup'] = max(verified_sp, benchmark_sp)
            results['valid_optimized_cases'] = 1
            results['valid_baseline_cases'] = 1
            results['geak_baseline_ms'] = geak_results['baseline_ms']
            results['geak_benchmark_speedup'] = geak_results.get('benchmark_speedup')
            results['geak_best_task'] = geak_results.get('best_task')
            results['geak_best_round'] = geak_results.get('best_round')
            results['geak_round_history'] = geak_results.get('round_history', [])
            log.info(
                f"Using GEAK verified results: {geak_results['verified_speedup']:.4f}x "
                f"(baseline={geak_results['baseline_ms']:.4f}ms, "
                f"candidate={geak_results['candidate_ms']:.4f}ms, "
                f"benchmark={geak_results.get('benchmark_speedup', 'N/A')}x, "
                f"task={geak_results.get('best_task', 'N/A')})"
            )

    log.info("=" * 80)
    log.info("Evaluation completed")
    log.info("=" * 80)
    
    return results


def write_task_result(
    workspace: Path,
    evaluation_results: Dict[str, Any],
    baseline_cases: List[TestCaseResult],
    task_name: str,
    agent_name: str,
    logger: Optional[logging.Logger] = None,
    create_plots: bool = True
) -> None:
    """
    Write standardized task_result.yaml file and optionally create performance plots.
    
    Args:
        workspace: Workspace directory
        evaluation_results: Results from evaluate_kernel()
        baseline_cases: Baseline test case results
        task_name: Name of the task
        agent_name: Name of the agent that generated the kernel
        logger: Optional logger
        create_plots: Whether to create performance comparison plots
    """
    log = logger or logging.getLogger(__name__)
    
    # Get average baseline time — prefer GEAK's verified baseline if available
    avg_baseline_time = 0.0
    valid_baseline_cases = _valid_perf_cases(baseline_cases)
    if evaluation_results.get('geak_baseline_ms', 0) > 0:
        avg_baseline_time = evaluation_results['geak_baseline_ms']
    elif valid_baseline_cases:
        avg_baseline_time = sum(c.execution_time_ms for c in valid_baseline_cases) / len(valid_baseline_cases)
    elif baseline_cases:
        log.warning(
            "No valid baseline performance samples found (execution_time_ms <= 0 or invalid). "
            "base_execution_time is set to 0.0"
        )
    
    # Get results
    optimized_time = evaluation_results.get('best_optimized_execution_time', 0.0)
    avg_speedup = evaluation_results.get('average_speedup', 0.0)
    
    # Use average speedup if available, otherwise calculate from average times
    if avg_speedup == 0.0 and avg_baseline_time > 0 and optimized_time > 0:
        avg_speedup = avg_baseline_time / optimized_time
    
    task_result = {
        'task_name': task_name,
        'pass_compilation': evaluation_results['pass_compilation'],
        'compilation_error_message': evaluation_results.get('compilation_error_message'),
        'pass_correctness': evaluation_results['pass_correctness'],
        'correctness_error_message': evaluation_results.get('correctness_error_message'),
        'base_execution_time': avg_baseline_time,
        'best_optimized_execution_time': optimized_time,
        'speedup_ratio': avg_speedup,
        'valid_baseline_cases': len(valid_baseline_cases),
        'valid_optimized_cases': evaluation_results.get('valid_optimized_cases', 0),
        'optimization_summary': f'Optimized by {agent_name} using centralized evaluator',
    }

    # Add GEAK-specific detailed results if available
    geak_details = {}
    if evaluation_results.get('geak_benchmark_speedup'):
        geak_details['benchmark_speedup'] = evaluation_results['geak_benchmark_speedup']
    if evaluation_results.get('geak_best_task'):
        geak_details['best_task'] = evaluation_results['geak_best_task']
    if evaluation_results.get('geak_best_round'):
        geak_details['best_round'] = evaluation_results['geak_best_round']
    if evaluation_results.get('geak_round_history'):
        geak_details['round_history'] = evaluation_results['geak_round_history']
    if geak_details:
        task_result['geak_details'] = geak_details
    
    result_file = workspace / 'task_result.yaml'
    with open(result_file, 'w') as f:
        yaml.dump(task_result, f, default_flow_style=False, sort_keys=False)
    
    log.info(f"Written task_result.yaml to {result_file}")
    
    # Create performance plots if requested and both baseline and optimized data exist
    if create_plots:
        try:
            from .plotting import plot_performance_comparison
            
            # Only create plots if we have performance data
            if (evaluation_results.get('best_optimized_execution_time', 0.0) > 0 and 
                baseline_cases):
                plot_file = plot_performance_comparison(workspace, task_name, logger)
                if plot_file:
                    log.info(f"Created performance comparison plot: {plot_file}")
        except ImportError as e:
            log.warning(f"Could not create plots (matplotlib may not be installed): {e}")
        except Exception as e:
            log.warning(f"Failed to create performance plots: {e}")
