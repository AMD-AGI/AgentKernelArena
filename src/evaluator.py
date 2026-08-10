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
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from .evaluator_utils import (
    inspect_formal_source_anti_tamper,
    inspect_target_definitions,
    run_command,
)
from .jit_rebuild import force_jit_rebuild
from .performance import measure_performance, measure_baseline
from .testcases import TestCaseResult, save_performance_results, calculate_average_speedup, collect_benchmark_methods

# Default timeouts for run_command (seconds). Repository CMake builds can exceed a few minutes.
_DEFAULT_COMPILE_TIMEOUT_S = 3600
_DEFAULT_CORRECTNESS_TIMEOUT_S = 3600


def _deadline_timeout(
    configured_seconds: float,
    deadline_monotonic: float | None,
    clock,
) -> float:
    if deadline_monotonic is None:
        return configured_seconds
    return min(configured_seconds, max(0.0, deadline_monotonic - clock()))


def _valid_perf_cases(cases: List[TestCaseResult]) -> List[TestCaseResult]:
    """Return only test cases with valid positive execution time."""
    valid_cases: List[TestCaseResult] = []
    for case in cases:
        if case.execution_time_ms is not None and case.execution_time_ms > 0:
            valid_cases.append(case)
    return valid_cases


def _source_anti_tamper_error(report: Dict[str, Any], stage: str) -> str:
    details = []
    for violation in report.get("violations", []):
        if isinstance(violation, dict):
            details.append(str(violation.get("rule") or "unknown_violation"))
    for file_report in report.get("files", []):
        if not isinstance(file_report, dict):
            continue
        for violation in file_report.get("violations", []):
            if isinstance(violation, dict):
                details.append(
                    f"{file_report.get('path')}:{violation.get('line', 0)}:"
                    f"{violation.get('rule') or 'unknown_violation'}"
                )
    summary = ", ".join(details[:8]) or "invalid formal source evidence"
    return f"Formal source anti-tamper guard failed before/after {stage}: {summary}"


def _verify_formal_source_anchor(
    workspace: Path,
    task_config: Dict[str, Any],
    expected_source_manifest_sha256: str,
    stage: str,
) -> Optional[str]:
    report = inspect_formal_source_anti_tamper(
        workspace,
        task_config,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )
    if report["verdict"] != "PASS":
        return _source_anti_tamper_error(report, stage)
    return None


def _refresh_formal_source_evidence(
    results: Dict[str, Any],
    workspace: Path,
    task_config: Dict[str, Any],
    expected_source_manifest_sha256: str,
    stage: str,
) -> Optional[str]:
    report = inspect_formal_source_anti_tamper(
        workspace,
        task_config,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )
    results["source_anti_tamper"] = report
    if report["verdict"] != "PASS":
        return _source_anti_tamper_error(report, stage)
    return None


def evaluate_compilation(
    workspace: Path,
    task_config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    deadline_monotonic: float | None = None,
    clock=time.monotonic,
    source_anti_tamper_required: bool = False,
    expected_source_manifest_sha256: Optional[str] = None,
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
    if source_anti_tamper_required:
        if not expected_source_manifest_sha256:
            return False, "Formal compilation is missing its source-manifest anchor"
        guard_error = _verify_formal_source_anchor(
            workspace,
            task_config,
            expected_source_manifest_sha256,
            "compilation",
        )
        if guard_error is not None:
            return False, guard_error
    rebuild_env = force_jit_rebuild(task_config, log, workspace)
    compile_commands = task_config.get('compile_command', [])
    
    if not compile_commands:
        log.warning("No compile_command found in task config")
        return False, "No compile_command specified"

    compile_timeout = float(task_config.get("compile_timeout", _DEFAULT_COMPILE_TIMEOUT_S))
    
    for cmd in compile_commands:
        command_timeout = _deadline_timeout(compile_timeout, deadline_monotonic, clock)
        if command_timeout <= 0:
            return False, "Compilation skipped because the hard task deadline expired"
        success, stdout, stderr = run_command(cmd, workspace, timeout=command_timeout, logger=log, extra_env=rebuild_env)
        if not success:
            error_msg = f"Compilation failed\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            return False, error_msg

    if source_anti_tamper_required:
        guard_error = _verify_formal_source_anchor(
            workspace,
            task_config,
            expected_source_manifest_sha256,
            "compilation",
        )
        if guard_error is not None:
            return False, guard_error
    
    return True, None


def evaluate_correctness(
    workspace: Path,
    task_config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    deadline_monotonic: float | None = None,
    clock=time.monotonic,
    source_anti_tamper_required: bool = False,
    expected_source_manifest_sha256: Optional[str] = None,
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
    if source_anti_tamper_required:
        if not expected_source_manifest_sha256:
            return False, "Formal correctness is missing its source-manifest anchor"
        guard_error = _verify_formal_source_anchor(
            workspace,
            task_config,
            expected_source_manifest_sha256,
            "correctness",
        )
        if guard_error is not None:
            return False, guard_error
    rebuild_env = force_jit_rebuild(task_config, log, workspace)
    correctness_commands = task_config.get('correctness_command', [])
    
    if not correctness_commands:
        log.warning("No correctness_command found in task config")
        return False, "No correctness_command specified"

    correctness_timeout = float(
        task_config.get("correctness_timeout", _DEFAULT_CORRECTNESS_TIMEOUT_S)
    )
    
    for cmd in correctness_commands:
        command_timeout = _deadline_timeout(correctness_timeout, deadline_monotonic, clock)
        if command_timeout <= 0:
            return False, "Correctness skipped because the hard task deadline expired"
        success, stdout, stderr = run_command(
            cmd, workspace, timeout=command_timeout, logger=log, extra_env=rebuild_env
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

    if source_anti_tamper_required:
        guard_error = _verify_formal_source_anchor(
            workspace,
            task_config,
            expected_source_manifest_sha256,
            "correctness",
        )
        if guard_error is not None:
            return False, guard_error
    
    return True, None


def evaluate_kernel(
    workspace: Path,
    task_config: Dict[str, Any],
    baseline_cases: List[TestCaseResult],
    logger: Optional[logging.Logger] = None,
    deadline_monotonic: float | None = None,
    clock=time.monotonic,
    source_anti_tamper_required: bool = False,
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
    }

    source_manifest_sha256 = None
    if source_anti_tamper_required:
        initial_source_report = inspect_formal_source_anti_tamper(workspace, task_config)
        source_manifest_sha256 = initial_source_report["source_manifest_sha256"]
        source_report = inspect_formal_source_anti_tamper(
            workspace,
            task_config,
            expected_source_manifest_sha256=source_manifest_sha256,
        )
        results["source_anti_tamper"] = source_report
        if source_report["verdict"] != "PASS":
            results["compilation_error_message"] = _source_anti_tamper_error(
                source_report, "compilation"
            )
            log.warning(results["compilation_error_message"])
            return results
    
    # 1. Compilation check
    log.info("Step 1: Checking compilation...")
    pass_compilation, comp_error = evaluate_compilation(
        workspace,
        task_config,
        logger,
        deadline_monotonic,
        clock,
        source_anti_tamper_required,
        source_manifest_sha256,
    )
    if source_anti_tamper_required:
        guard_error = _refresh_formal_source_evidence(
            results,
            workspace,
            task_config,
            source_manifest_sha256,
            "compilation",
        )
        if guard_error is not None:
            pass_compilation, comp_error = False, guard_error
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
            if source_anti_tamper_required:
                guard_error = _refresh_formal_source_evidence(
                    results,
                    workspace,
                    task_config,
                    source_manifest_sha256,
                    "correctness",
                )
                if guard_error is not None:
                    results['correctness_error_message'] = guard_error
            log.warning(corr_error)
            return results

    pass_correctness, corr_error = evaluate_correctness(
        workspace,
        task_config,
        logger,
        deadline_monotonic,
        clock,
        source_anti_tamper_required,
        source_manifest_sha256,
    )
    if source_anti_tamper_required:
        guard_error = _refresh_formal_source_evidence(
            results,
            workspace,
            task_config,
            source_manifest_sha256,
            "correctness",
        )
        if guard_error is not None:
            pass_correctness, corr_error = False, guard_error
    results['pass_correctness'] = pass_correctness
    results['correctness_error_message'] = corr_error
    
    if not pass_correctness:
        log.warning("Correctness failed, skipping performance measurement")
        return results
    
    # 3. Performance measurement (only if both compilation and correctness passed)
    log.info("Step 3: Measuring performance...")
    if source_anti_tamper_required:
        guard_error = _refresh_formal_source_evidence(
            results,
            workspace,
            task_config,
            source_manifest_sha256,
            "performance",
        )
        if guard_error is not None:
            results['pass_correctness'] = False
            results['correctness_error_message'] = guard_error
            log.warning(guard_error)
            return results
    optimized_cases = measure_performance(
        workspace,
        task_config,
        logger,
        deadline_monotonic=deadline_monotonic,
        clock=clock,
    )
    if source_anti_tamper_required:
        guard_error = _refresh_formal_source_evidence(
            results,
            workspace,
            task_config,
            source_manifest_sha256,
            "performance",
        )
        if guard_error is not None:
            results['pass_correctness'] = False
            results['correctness_error_message'] = guard_error
            log.warning(guard_error)
            return results
    
    if optimized_cases:
        # Save optimized results
        save_performance_results(optimized_cases, workspace, "optimized_perf.yaml", logger)
        # Record the timing method(s) used for the optimized measurement so the final
        # task_result can flag mixed-method (baseline vs optimized) comparisons.
        results['optimized_benchmark_methods'] = collect_benchmark_methods(optimized_cases)
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
                avg_baseline_time = sum(c.execution_time_ms for c in valid_baseline_cases) / len(valid_baseline_cases)
                log.info(
                    f"Baseline: {len(valid_baseline_cases)}/{len(baseline_cases)} valid test case(s), "
                    f"average time: {avg_baseline_time:.4f} ms"
                )

                if (
                    len(valid_baseline_cases) != len(baseline_cases)
                    or len(valid_optimized_cases) != len(optimized_cases)
                ):
                    error_msg = (
                        "Cannot calculate speedup because performance results contain invalid "
                        "test case timings: "
                        f"baseline_valid={len(valid_baseline_cases)}/{len(baseline_cases)}, "
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
                if baseline_cases:
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
    valid_baseline_cases = _valid_perf_cases(baseline_cases)
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
    
    # Use average speedup if available, otherwise calculate from average times
    if avg_speedup == 0.0 and not speedup_error and avg_baseline_time > 0 and optimized_time > 0:
        avg_speedup = avg_baseline_time / optimized_time

    # Surface the timing method(s) used on each side. If baseline and optimized were
    # measured with different methods (e.g. cuda_graph vs cuda_event_fallback), the
    # reported speedup_ratio may reflect the measurement-method delta rather than kernel
    # quality — make that visible so such comparisons can be spotted/discounted.
    baseline_methods = collect_benchmark_methods(baseline_cases)
    optimized_methods = evaluation_results.get('optimized_benchmark_methods', [])
    benchmark_method_consistent = (
        bool(baseline_methods)
        and bool(optimized_methods)
        and len(set(baseline_methods) | set(optimized_methods)) == 1
    )
    if baseline_methods and optimized_methods and not benchmark_method_consistent:
        log.warning(
            f"Benchmark method mismatch — baseline={baseline_methods} optimized={optimized_methods}. "
            "speedup_ratio may reflect the measurement-method delta (e.g. cuda_graph vs "
            "cuda_event_fallback overhead), not kernel quality."
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
        'valid_baseline_cases': len(valid_baseline_cases),
        'valid_optimized_cases': evaluation_results.get('valid_optimized_cases', 0),
        'speedup_calculation_error_message': speedup_error,
        'optimization_summary': f'Optimized by {agent_name} using centralized evaluator'
    }
    if "source_anti_tamper" in evaluation_results:
        task_result["source_anti_tamper"] = evaluation_results["source_anti_tamper"]
    
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
