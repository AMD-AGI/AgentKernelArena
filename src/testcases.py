# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Test case results for the evaluator: data structures, matching, consistency
checks, speedup calculation and persistence.

Results come from validated task actions (``src.task_protocol.performance_cases``)
and always carry the stable case IDs of the session's case manifest.
"""
import math
import logging
import yaml
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class TestCaseResult:
    """Represents a single test case performance result."""
    test_case_id: str  # Stable case identifier from the task's case manifest
    shape: Optional[List[Any]] = None  # Shape/size parameters (e.g., [256, 256, 256])
    execution_time_ms: float = 0.0
    metadata: Optional[Dict[str, Any]] = None  # Additional info (params, dtype, benchmark_*)


def match_test_cases(
    baseline_cases: List[TestCaseResult],
    optimized_cases: List[TestCaseResult],
    logger: Optional[logging.Logger] = None,
) -> List[Tuple[TestCaseResult, TestCaseResult]]:
    """
    Pair baseline and optimized results by their stable case IDs.

    An ID pairs results only when it identifies exactly one case on each side;
    duplicated or missing IDs stay unmatched, so callers that require a complete
    match refuse to score them. Workload identity is checked separately.

    Returns:
        List of (baseline_case, optimized_case) tuples in baseline order
    """
    log = logger or logging.getLogger(__name__)
    baseline_ids = [case.test_case_id for case in baseline_cases]
    optimized_by_id: Dict[str, List[TestCaseResult]] = {}
    for case in optimized_cases:
        optimized_by_id.setdefault(case.test_case_id, []).append(case)
    matched = [
        (case, optimized_by_id[case.test_case_id][0])
        for case in baseline_cases
        if case.test_case_id and baseline_ids.count(case.test_case_id) == 1
        and len(optimized_by_id.get(case.test_case_id, [])) == 1
    ]
    log.info(f"Matched {len(matched)} test case(s) between baseline and optimized")
    return matched


def calculate_average_speedup(
    baseline_cases: List[TestCaseResult],
    optimized_cases: List[TestCaseResult],
    logger: Optional[logging.Logger] = None,
    require_complete_match: bool = True,
) -> float:
    """
    Calculate average speedup across all matched test cases.
    
    Args:
        baseline_cases: Baseline test case results
        optimized_cases: Optimized test case results
        logger: Optional logger
        require_complete_match: Require every baseline and optimized case to match
        
    Returns:
        Average speedup ratio (baseline_time / optimized_time), or 0.0 if no valid matches
    """
    log = logger or logging.getLogger(__name__)
    matched = match_test_cases(baseline_cases, optimized_cases, logger)

    if not matched:
        log.warning("No test cases matched, cannot calculate speedup")
        return 0.0

    if require_complete_match and (
        len(matched) != len(baseline_cases) or len(matched) != len(optimized_cases)
    ):
        log.warning(
            "Incomplete test case match, refusing to calculate speedup: "
            f"matched={len(matched)}, baseline={len(baseline_cases)}, optimized={len(optimized_cases)}"
        )
        return 0.0

    workload_mismatches = _workload_mismatches(matched)
    if workload_mismatches:
        log.warning(
            "Workload changed between baseline and optimized cases: %s",
            workload_mismatches,
        )
        return 0.0
    
    method_mismatches = _benchmark_method_mismatches(matched)
    if method_mismatches:
        for mismatch in method_mismatches:
            log.warning(
                "Benchmark method mismatch for test case %s: baseline=%r, optimized=%r",
                mismatch['test_case_id'],
                mismatch['baseline_benchmark_method'],
                mismatch['optimized_benchmark_method'],
            )
        log.warning(
            "Refusing to calculate speedup across different benchmark methods"
        )
        return 0.0

    speedups = []
    for base_case, opt_case in matched:
        if (
            math.isfinite(base_case.execution_time_ms)
            and math.isfinite(opt_case.execution_time_ms)
            and base_case.execution_time_ms > 0
            and opt_case.execution_time_ms > 0
        ):
            speedup = base_case.execution_time_ms / opt_case.execution_time_ms
            speedups.append(speedup)
            log.debug(f"Test case {base_case.test_case_id}: {base_case.execution_time_ms:.4f} ms -> {opt_case.execution_time_ms:.4f} ms (speedup: {speedup:.2f}x)")
        else:
            log.warning(f"Invalid execution times for test case {base_case.test_case_id}: baseline={base_case.execution_time_ms}, optimized={opt_case.execution_time_ms}")
            if require_complete_match:
                log.warning("Invalid case time encountered, refusing to calculate speedup")
                return 0.0
    
    if not speedups:
        log.warning("No valid speedups calculated")
        return 0.0
    
    # Geometric mean: each case contributes its speedup ratio equally, so a
    # large shape cannot outweigh a small one and a 2x win cancels a 2x loss.
    avg_speedup = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
    log.info(f"Geometric-mean speedup across {len(speedups)} test case(s): {avg_speedup:.2f}x")
    return avg_speedup


def _benchmark_method_mismatches(
    matched_cases: List[Tuple[TestCaseResult, TestCaseResult]],
) -> List[Dict[str, Any]]:
    """Describe exact per-pair benchmark method mismatches.

    A ``mixed:...`` value is an aggregate set, not a stable case-to-method
    mapping.  Equal mixed strings therefore cannot prove that the same shapes
    used the same timer in the baseline and optimized runs.  Refuse those
    aggregate comparisons instead of silently scoring a potentially swapped
    graph/event assignment.
    """

    mismatches: List[Dict[str, Any]] = []
    from .measurement import DEVICE_METHODS, SERVING_METHOD, cold_cache_identity
    comparable_methods = DEVICE_METHODS | {SERVING_METHOD}
    for base_case, opt_case in matched_cases:
        base_metadata = base_case.metadata or {}
        opt_metadata = opt_case.metadata or {}
        base_method = base_metadata.get('benchmark_method')
        opt_method = opt_metadata.get('benchmark_method')
        task_reported_mismatch = (
            base_metadata.get('benchmark_method_consistent') is False
            or opt_metadata.get('benchmark_method_consistent') is False
        )
        mixed_aggregate = (
            isinstance(base_method, str) and base_method.startswith('mixed:')
        ) or (
            isinstance(opt_method, str) and opt_method.startswith('mixed:')
        )
        missing_or_unknown = (
            base_method not in comparable_methods
            or opt_method not in comparable_methods
        )
        cache_policy_mismatch = False
        if base_method == opt_method == "cuda_graph_cold_cache":
            try:
                cache_policy_mismatch = cold_cache_identity(base_metadata) != cold_cache_identity(opt_metadata)
            except ValueError:
                cache_policy_mismatch = True
        if (
            base_method == opt_method
            and not mixed_aggregate
            and not missing_or_unknown
            and not task_reported_mismatch
            and not cache_policy_mismatch
        ):
            continue
        mismatch = {
            'test_case_id': base_case.test_case_id,
            'optimized_test_case_id': opt_case.test_case_id,
            'baseline_benchmark_method': base_method,
            'optimized_benchmark_method': opt_method,
        }
        if task_reported_mismatch:
            mismatch['reason'] = 'task_reported_method_mismatch'
        elif cache_policy_mismatch:
            mismatch['reason'] = 'cold_cache_policy_mismatch'
        elif mixed_aggregate:
            mismatch['reason'] = 'ambiguous_mixed_aggregate'
        elif missing_or_unknown:
            mismatch['reason'] = 'missing_or_unknown_benchmark_method'
        mismatches.append(mismatch)
    return mismatches


def _workload_mismatches(
    matched_cases: List[Tuple[TestCaseResult, TestCaseResult]],
) -> List[Dict[str, Any]]:
    """An ID pairs results; it does not prove their workloads are identical.

    Compare every workload field reported by either side, including removal of
    baseline metadata. Timing/profiling metadata is deliberately outside this
    comparison.
    """
    def workload(case: TestCaseResult) -> Dict[str, Any]:
        metadata = case.metadata or {}
        fields = {
            key: metadata[key] for key in ('params', 'dtype') if key in metadata
        }
        if case.shape is not None:
            fields['shape'] = case.shape
        return fields

    mismatches = []
    for base_case, opt_case in matched_cases:
        base = workload(base_case)
        opt = workload(opt_case)
        changed = sorted(
            key for key in base.keys() | opt.keys()
            if key not in base or key not in opt or base[key] != opt[key]
        )
        if changed:
            mismatches.append({
                'test_case_id': base_case.test_case_id,
                'optimized_test_case_id': opt_case.test_case_id,
                'reason': 'workload_changed',
                'fields': changed,
                'baseline_workload': base,
                'optimized_workload': opt,
            })
    return mismatches


def analyze_workload_consistency(
    baseline_cases: List[TestCaseResult],
    optimized_cases: List[TestCaseResult],
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Require complete case pairing and equal declared workloads."""
    matched = match_test_cases(baseline_cases, optimized_cases, logger)
    mismatches = _workload_mismatches(matched)
    complete = (
        bool(matched) and len(matched) == len(baseline_cases) == len(optimized_cases)
    )
    return complete and not mismatches, mismatches


def analyze_benchmark_method_consistency(
    baseline_cases: List[TestCaseResult],
    optimized_cases: List[TestCaseResult],
    logger: Optional[logging.Logger] = None,
    require_complete_match: bool = True,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Check benchmark methods on matched cases, allowing per-shape fallback.

    Different cases may legitimately use different methods.  A comparison is
    consistent when every matched baseline/optimized pair uses the same exact
    non-aggregate method.  ``mixed:...`` summaries are rejected because they do
    not preserve which shape used which method.
    """
    matched = match_test_cases(baseline_cases, optimized_cases, logger)
    if not matched:
        return False, []
    if require_complete_match and (
        len(matched) != len(baseline_cases) or len(matched) != len(optimized_cases)
    ):
        return False, []
    mismatches = _benchmark_method_mismatches(matched)
    return not mismatches, mismatches


def save_performance_results(
    test_cases: List[TestCaseResult],
    workspace: Path,
    filename: str,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Save test case results to YAML file.
    
    Saves identifying fields, timing, params, dtype, and canonical ``benchmark_*``
    metadata. Other task-specific metadata is excluded to keep the file compact.
    
    Args:
        test_cases: List of test case results
        workspace: Workspace directory
        filename: Filename (e.g., 'baseline_perf.yaml')
        logger: Optional logger
    """
    log = logger or logging.getLogger(__name__)
    
    results = {
        'test_cases': []
    }
    
    for case in test_cases:
        case_dict = {
            'test_case_id': case.test_case_id,
            'execution_time_ms': case.execution_time_ms
        }
        if case.shape is not None:
            case_dict['shape'] = case.shape
        # Preserve workload identity as well as the timing diagnostics below.
        if case.metadata:
            for key in ('params', 'dtype'):
                if key in case.metadata:
                    case_dict[key] = case.metadata[key]
        # Persist all canonical benchmark metadata so baseline and optimized
        # measurements can be audited and compared after workspace reload.
        if case.metadata:
            for key, value in case.metadata.items():
                if key.startswith('benchmark_') and value is not None:
                    case_dict[key] = value
        results['test_cases'].append(case_dict)
    
    output_file = workspace / filename
    with open(output_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    
    log.info(f"Saved {len(test_cases)} test case(s) to {output_file}")


def load_performance_results(
    workspace: Path,
    filename: str,
    logger: Optional[logging.Logger] = None
) -> List[TestCaseResult]:
    """
    Load test case results from YAML file.
    
    Args:
        workspace: Workspace directory
        filename: Filename (e.g., 'baseline_perf.yaml')
        logger: Optional logger
        
    Returns:
        List of TestCaseResult objects
    """
    log = logger or logging.getLogger(__name__)
    input_file = workspace / filename
    
    if not input_file.exists():
        log.warning(f"Performance results file not found: {input_file}")
        return []
    
    try:
        with open(input_file, 'r') as f:
            data = yaml.safe_load(f)
        
        test_cases = []
        for case_dict in data.get('test_cases', []):
            test_cases.append(TestCaseResult(
                test_case_id=case_dict.get('test_case_id', 'unknown'),
                shape=case_dict.get('shape'),
                execution_time_ms=float(case_dict.get('execution_time_ms', 0.0)),
                metadata={k: v for k, v in case_dict.items() if k not in ['test_case_id', 'shape', 'execution_time_ms']}
            ))
        
        log.info(f"Loaded {len(test_cases)} test case(s) from {input_file}")
        return test_cases

    except Exception as e:
        log.error(f"Failed to load performance results from {input_file}: {e}")
        return []


def collect_benchmark_methods(test_cases: List[TestCaseResult]) -> List[str]:
    """Return the sorted set of distinct `benchmark_method` values across test cases.

    The method is read from each case's metadata (populated from the benchmark JSON
    and persisted into baseline_perf.yaml / optimized_perf.yaml). An empty list
    means no method was recorded; such cases are retained but are not comparable
    for speedup.
    """
    methods = set()
    for case in test_cases:
        if case.metadata:
            method = case.metadata.get('benchmark_method')
            if method:
                methods.add(str(method))
    return sorted(methods)
