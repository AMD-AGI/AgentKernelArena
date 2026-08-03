# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Centralized evaluator for AgentKernelArena.

This module provides standardized evaluation of optimized kernels:
- Compilation checking
- Correctness validation
- Performance measurement
- Baseline measurement for speedup calculation
"""
import logging
import math
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from .evaluator_utils import inspect_target_definitions, run_command
from .jit_rebuild import force_jit_rebuild
from .performance import measure_performance, measure_baseline
from .testcases import (
    TestCaseResult,
    analyze_benchmark_method_consistency,
    calculate_average_speedup,
    collect_benchmark_methods,
    save_performance_results,
    select_method_matched_baselines,
)

# Default timeouts for run_command (seconds). Repository CMake builds can exceed a few minutes.
_DEFAULT_COMPILE_TIMEOUT_S = 3600
_DEFAULT_CORRECTNESS_TIMEOUT_S = 3600


def _valid_perf_cases(cases: List[TestCaseResult]) -> List[TestCaseResult]:
    """Return only test cases with valid positive execution time."""
    valid_cases: List[TestCaseResult] = []
    for case in cases:
        if (
            case.execution_time_ms is not None
            and math.isfinite(case.execution_time_ms)
            and case.execution_time_ms > 0
        ):
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
    rebuild_env = force_jit_rebuild(task_config, log, workspace)
    compile_commands = task_config.get('compile_command', [])
    
    if not compile_commands:
        log.warning("No compile_command found in task config")
        return False, "No compile_command specified"

    compile_timeout = int(task_config.get("compile_timeout", _DEFAULT_COMPILE_TIMEOUT_S))
    
    for cmd in compile_commands:
        success, stdout, stderr = run_command(cmd, workspace, timeout=compile_timeout, logger=log, extra_env=rebuild_env)
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
    rebuild_env = force_jit_rebuild(task_config, log, workspace)
    correctness_commands = task_config.get('correctness_command', [])
    
    if not correctness_commands:
        log.warning("No correctness_command found in task config")
        return False, "No correctness_command specified"

    correctness_timeout = int(
        task_config.get("correctness_timeout", _DEFAULT_CORRECTNESS_TIMEOUT_S)
    )
    
    for cmd in correctness_commands:
        success, stdout, stderr = run_command(
            cmd, workspace, timeout=correctness_timeout, logger=log, extra_env=rebuild_env
        )
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
        'speedup_calculation_error_message': None,
        'benchmark_method_consistent': False,
        'benchmark_method_mismatches': [],
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
    # The as-shipped torch2flydsl starter is allowed during baseline and task
    # package validation, both of which bypass this optimized-kernel pipeline.
    # Once an optimization agent has run, however, its declared targets must no
    # longer be unconditional NotImplementedError stubs; otherwise a harness
    # could silently time and validate its reference fallback.
    if task_config.get("task_type") == "torch2flydsl":
        missing_names, stub_names = inspect_target_definitions(workspace, task_config)
        target_errors = []
        if missing_names:
            target_errors.append(
                "missing declared top-level target definition(s): "
                + ", ".join(missing_names)
            )
        if stub_names:
            target_errors.append(
                "unimplemented target stub(s): " + ", ".join(stub_names)
            )
        if target_errors:
            corr_error = "Invalid torch2flydsl optimization submission: " + "; ".join(
                target_errors
            )
            results['correctness_error_message'] = corr_error
            log.warning(corr_error)
            return results

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
        # Record the timing method(s) used for the optimized measurement so the final
        # task_result can flag mixed-method (baseline vs optimized) comparisons.
        results['optimized_benchmark_methods'] = collect_benchmark_methods(optimized_cases)
        comparison_baseline_cases = select_method_matched_baselines(
            baseline_cases, optimized_cases, log
        )
        results['_comparison_baseline_cases'] = comparison_baseline_cases
        save_performance_results(
            comparison_baseline_cases,
            workspace,
            "comparison_baseline_perf.yaml",
            logger,
        )
        valid_optimized_cases = _valid_perf_cases(optimized_cases)
        valid_baseline_cases = _valid_perf_cases(comparison_baseline_cases)
        results['valid_optimized_cases'] = len(valid_optimized_cases)
        results['valid_baseline_cases'] = len(valid_baseline_cases)
        method_consistent, method_mismatches = analyze_benchmark_method_consistency(
            valid_baseline_cases,
            valid_optimized_cases,
            logger,
            require_complete_match=True,
        )
        results['benchmark_method_consistent'] = method_consistent
        results['benchmark_method_mismatches'] = method_mismatches

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
                avg_baseline_time = sum(c.execution_time_ms for c in valid_baseline_cases) / len(valid_baseline_cases)
                log.info(
                    f"Baseline: {len(valid_baseline_cases)}/{len(comparison_baseline_cases)} valid test case(s), "
                    f"average time: {avg_baseline_time:.4f} ms"
                )

                if method_mismatches:
                    mismatch_parts = []
                    for item in method_mismatches:
                        if item.get('reason') == 'ambiguous_mixed_aggregate':
                            mismatch_parts.append(
                                f"{item['test_case_id']}: ambiguous aggregate "
                                f"{item['baseline_benchmark_method']!r}"
                            )
                        elif item.get('reason') == 'missing_or_unknown_benchmark_method':
                            mismatch_parts.append(
                                f"{item['test_case_id']}: missing or unknown method "
                                f"(baseline={item['baseline_benchmark_method']!r}, "
                                f"optimized={item['optimized_benchmark_method']!r})"
                            )
                        else:
                            mismatch_parts.append(
                                f"{item['test_case_id']}: "
                                f"{item['baseline_benchmark_method']!r} != "
                                f"{item['optimized_benchmark_method']!r}"
                            )
                    mismatch_summary = "; ".join(mismatch_parts)
                    error_msg = (
                        "Cannot calculate speedup because matched test cases used "
                        f"different benchmark methods: {mismatch_summary}"
                    )
                    results['speedup_calculation_error_message'] = error_msg
                    log.warning(error_msg)
                elif (
                    len(valid_baseline_cases) != len(comparison_baseline_cases)
                    or len(valid_optimized_cases) != len(optimized_cases)
                ):
                    error_msg = (
                        "Cannot calculate speedup because performance results contain invalid "
                        "test case timings: "
                        f"baseline_valid={len(valid_baseline_cases)}/{len(comparison_baseline_cases)}, "
                        f"optimized_valid={len(valid_optimized_cases)}/{len(optimized_cases)}"
                    )
                    results['speedup_calculation_error_message'] = error_msg
                    log.warning(error_msg)
                else:
                    avg_speedup = calculate_average_speedup(
                        valid_baseline_cases,
                        valid_optimized_cases,
                        logger,
                        require_complete_match=True,
                    )
                    if avg_speedup > 0:
                        results['average_speedup'] = avg_speedup
                        log.info(f"Average speedup: {avg_speedup:.2f}x")
                    else:
                        error_msg = (
                            "Cannot calculate speedup because baseline and optimized "
                            "test cases did not match completely"
                        )
                        results['speedup_calculation_error_message'] = error_msg
                        log.warning(error_msg)
            else:
                if comparison_baseline_cases:
                    error_msg = (
                        "Baseline data exists but has no valid performance samples "
                        "(execution_time_ms <= 0 or invalid). Cannot calculate speedup."
                    )
                    results['speedup_calculation_error_message'] = error_msg
                    log.warning(error_msg)
                else:
                    error_msg = "Baseline not available, cannot calculate speedup"
                    results['speedup_calculation_error_message'] = error_msg
                    log.warning(error_msg)
    else:
        log.warning("Failed to measure optimized execution time")
    
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
    
    # Get average baseline time
    avg_baseline_time = 0.0
    comparison_baseline_cases = evaluation_results.get(
        '_comparison_baseline_cases', baseline_cases
    )
    valid_baseline_cases = _valid_perf_cases(comparison_baseline_cases)
    if valid_baseline_cases:
        avg_baseline_time = sum(c.execution_time_ms for c in valid_baseline_cases) / len(valid_baseline_cases)
    elif baseline_cases:
        log.warning(
            "No valid baseline performance samples found (execution_time_ms <= 0 or invalid). "
            "base_execution_time is set to 0.0"
        )
    
    # Get results
    optimized_time = evaluation_results.get('best_optimized_execution_time', 0.0)
    avg_speedup = evaluation_results.get('average_speedup', 0.0)
    speedup_error = evaluation_results.get('speedup_calculation_error_message')
    benchmark_method_consistent = bool(
        evaluation_results.get('benchmark_method_consistent', False)
    )
    
    # Use average speedup if available, otherwise calculate from average times
    if (
        avg_speedup == 0.0
        and not speedup_error
        and benchmark_method_consistent
        and avg_baseline_time > 0
        and optimized_time > 0
    ):
        avg_speedup = avg_baseline_time / optimized_time

    # Surface task-wide method sets for diagnostics, but enforce consistency on
    # matched cases.  One shape may use graph while another falls back to events;
    # that is fair as long as each baseline/optimized pair uses the same method.
    baseline_methods = collect_benchmark_methods(comparison_baseline_cases)
    optimized_methods = evaluation_results.get('optimized_benchmark_methods', [])
    benchmark_method_mismatches = evaluation_results.get(
        'benchmark_method_mismatches', []
    )
    if benchmark_method_mismatches:
        log.warning(
            "Benchmark method mismatch on matched case(s); speedup_ratio is disabled: %s",
            benchmark_method_mismatches,
        )

    task_result = {
        'task_name': task_name,
        'pass_compilation': evaluation_results['pass_compilation'],
        'compilation_error_message': evaluation_results.get('compilation_error_message'),
        'pass_correctness': evaluation_results['pass_correctness'],
        'correctness_error_message': evaluation_results.get('correctness_error_message'),
        'base_execution_time': avg_baseline_time,  # Average baseline time
        'best_optimized_execution_time': optimized_time,  # Average optimized time
        'speedup_ratio': avg_speedup,  # Average speedup across test cases
        'baseline_benchmark_methods': baseline_methods,
        'optimized_benchmark_methods': optimized_methods,
        'benchmark_method_consistent': benchmark_method_consistent,
        'benchmark_method_mismatches': benchmark_method_mismatches,
        'valid_baseline_cases': len(valid_baseline_cases),
        'valid_optimized_cases': evaluation_results.get('valid_optimized_cases', 0),
        'speedup_calculation_error_message': speedup_error,
        'optimization_summary': f'Optimized by {agent_name} using centralized evaluator'
    }
    
    result_file = workspace / 'task_result.yaml'
    with open(result_file, 'w') as f:
        yaml.dump(task_result, f, default_flow_style=False, sort_keys=False)
    
    log.info(f"Written task_result.yaml to {result_file}")
    
    # Create performance plots if requested and both baseline and optimized data exist
    if create_plots:
        try:
            from .plotting import plot_performance_comparison
            
            # Only create plots if we have performance data
            if (
                evaluation_results.get('best_optimized_execution_time', 0.0) > 0
                and baseline_cases
                and benchmark_method_consistent
                and not speedup_error
            ):
                plot_file = plot_performance_comparison(workspace, task_name, logger)
                if plot_file:
                    log.info(f"Created performance comparison plot: {plot_file}")
            elif not benchmark_method_consistent:
                log.warning(
                    "Skipping performance plots because benchmark methods are "
                    "not consistently matched"
                )
        except ImportError as e:
            log.warning(f"Could not create plots (matplotlib may not be installed): {e}")
        except Exception as e:
            log.warning(f"Failed to create performance plots: {e}")
