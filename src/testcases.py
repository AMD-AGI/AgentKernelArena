# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Test case handling for evaluator: data structures, parsing, matching, and speedup calculation.
"""
import json
import math
import re
import logging
import yaml
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

_SYNTHETIC_TEST_ID_METADATA_KEY = "_synthetic_test_case_id"
_TIMING_SOURCE_METADATA_KEY = "_timing_source"
_ALTERNATE_EVENT_PREFIX = "benchmark_alternate_event_"

_DEVICE_TIME_KEYS = [
    "device_time_ms",
    "gpu_time_ms",
    "kernel_time_ms",
    "cuda_event_time_ms",
    "cuda_graph_time_ms",
    "execution_time_ms",
]
_AMBIGUOUS_TIME_KEYS = ["execution_time", "time_ms", "time"]
_HOST_TIME_KEYS = [
    "host_time_ms",
    "wall_time_ms",
    "elapsed_time_ms",
    "total_time_ms",
    "end_to_end_time_ms",
]


@dataclass
class TestCaseResult:
    """Represents a single test case performance result."""
    test_case_id: str  # Unique identifier (e.g., "shape_0", "test_1")
    shape: Optional[List[Any]] = None  # Shape/size parameters (e.g., [256, 256, 256])
    execution_time_ms: float = 0.0
    metadata: Optional[Dict[str, Any]] = None  # Additional info (dtype, etc.)


def _safe_float(value: Any) -> Optional[float]:
    """Safely convert input to float; return None for invalid values."""
    try:
        if value is None:
            return None
        converted = float(value)
        return converted if math.isfinite(converted) else None
    except (TypeError, ValueError):
        return None


def _extract_time_from_dict(
    data: Dict[str, Any],
    is_baseline: bool = False,
    task_type: Optional[str] = None
) -> Tuple[float, Optional[str]]:
    """
    Extract execution time from a dictionary, handling various formats.
    
    Returns:
        Tuple of (time_ms, matched_key) or (0.0, None) if not found
    """
    # Special handling for torch2hip tasks
    if task_type == 'torch2hip':
        if is_baseline and 'ori_time' in data:
            time_val = _safe_float(data.get('ori_time'))
            if time_val is not None:
                return time_val, 'ori_time'  # Task runners already write milliseconds
        elif not is_baseline and 'opt_time' in data:
            time_val = _safe_float(data.get('opt_time'))
            if time_val is not None:
                return time_val, 'opt_time'  # Task runners already write milliseconds
    
    # Prefer device-side timings.  Host/wall timings include Python launch,
    # synchronization, subprocess, and harness overhead and can be gamed by
    # editing the test harness; they are not valid kernel timings.
    for key in _DEVICE_TIME_KEYS:
        if key in data:
            time_val = _safe_float(data.get(key))
            if time_val is None:
                continue
            return time_val, key

    # Ambiguous legacy fields are retained for compatibility.  Prefer adding a
    # device-specific field above in new task runners.
    for key in _AMBIGUOUS_TIME_KEYS:
        if key in data:
            time_val = _safe_float(data.get(key))
            if time_val is None:
                continue
            if key == 'time_ms':
                return time_val, key
            if time_val < 1000.0:  # Likely already in ms
                return time_val, key
            return time_val * 1000.0, key

    if any(key in data for key in _HOST_TIME_KEYS):
        return 0.0, None

    # Pytest benchmark format: nested timing_ms structure
    if 'timing_ms' in data:
        timing = data['timing_ms']
        if isinstance(timing, dict):
            # Prefer mean, fallback to median, then min
            if 'mean' in timing:
                time_val = _safe_float(timing.get('mean'))
                if time_val is not None:
                    return time_val, 'timing_ms.mean'
            elif 'median' in timing:
                time_val = _safe_float(timing.get('median'))
                if time_val is not None:
                    return time_val, 'timing_ms.median'
            elif 'min' in timing:
                time_val = _safe_float(timing.get('min'))
                if time_val is not None:
                    return time_val, 'timing_ms.min'

    # Repository throughput (e.g. rocPRIM benchmarks): GB/s — higher is better.
    # calculate_average_speedup uses baseline_time / optimized_time with "lower is better".
    # Store cost = 1/throughput so speedup = (1/bps_base)/(1/bps_opt) = bps_opt/bps_base.
    if 'bytes_per_second_gs' in data:
        bps = _safe_float(data.get('bytes_per_second_gs'))
        if bps is not None and bps > 0:
            return 1.0 / bps, 'bytes_per_second_gs (inverse for speedup)'

    return 0.0, None


def _build_metadata_from_case(
    case: Dict[str, Any],
    exclude_keys: List[str]
) -> Dict[str, Any]:
    """Build metadata dict excluding specified keys."""
    metadata = {k: v for k, v in case.items() if k not in exclude_keys}
    
    # Always include params if present
    if 'params' in case:
        metadata['params'] = case['params']
    
    return metadata


def _parse_single_case_from_dict(
    case: Dict[str, Any],
    default_test_id: str,
    is_baseline: bool = False,
    task_type: Optional[str] = None
) -> Optional[TestCaseResult]:
    """Parse a single test case from a dictionary."""
    has_explicit_test_id = 'test_case_id' in case
    test_id = case.get('test_case_id', default_test_id)
    shape = case.get('shape') or case.get('shapes')
    
    time_ms, matched_key = _extract_time_from_dict(case, is_baseline, task_type)
    
    # Allow negative values (e.g., -1.0) as valid error indicators from task runners
    # Reject only if time is missing (0.0) and no key was matched
    if time_ms == 0.0 and matched_key is None:
        return None
    
    # Build metadata
    exclude_keys = ['test_case_id', 'shape', 'shapes', 'timing_ms', 'params',
                   'ori_time', 'opt_time', 'bytes_per_second_gs'] + _DEVICE_TIME_KEYS + _AMBIGUOUS_TIME_KEYS + _HOST_TIME_KEYS
    metadata = _build_metadata_from_case(case, exclude_keys)
    if matched_key:
        metadata[_TIMING_SOURCE_METADATA_KEY] = matched_key
    
    # For torch2hip, include both ori_time and opt_time in metadata for reference
    if task_type == 'torch2hip':
        if 'ori_time' in case:
            metadata['ori_time'] = case['ori_time']
        if 'opt_time' in case:
            metadata['opt_time'] = case['opt_time']
        if 'speedup' in case:
            metadata['speedup'] = case['speedup']
    if 'bytes_per_second_gs' in case:
        metadata['bytes_per_second_gs'] = case['bytes_per_second_gs']
    metadata[_SYNTHETIC_TEST_ID_METADATA_KEY] = not has_explicit_test_id

    return TestCaseResult(
        test_case_id=test_id,
        shape=shape,
        execution_time_ms=time_ms,
        metadata=metadata
    )


def parse_test_cases_from_json(
    report_file: Path,
    logger: Optional[logging.Logger] = None,
    is_baseline: bool = False,
    task_type: Optional[str] = None
) -> List[TestCaseResult]:
    """
    Parse multiple test case results from JSON report file.
    
    Handles:
    - Array of test cases (hip2hip, pytest benchmark format)
    - Single object with standard keys (triton2triton/vllm)
    - Single object with torch2hip keys (ori_time/opt_time)
    - Single object with custom _ms keys
    
    Args:
        report_file: Path to JSON report file
        logger: Optional logger
        is_baseline: If True, use ori_time for torch2hip; if False, use opt_time
        task_type: Task type (e.g., 'torch2hip', 'hip2hip', 'triton2triton')
        
    Returns:
        List of TestCaseResult objects
    """
    log = logger or logging.getLogger(__name__)
    test_cases = []
    
    try:
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        # Format 1: Array of test cases (hip2hip, pytest benchmark)
        if isinstance(report, list):
            for idx, case in enumerate(report):
                try:
                    if not isinstance(case, dict):
                        log.warning(f"Skipping non-dict test case at index {idx} in {report_file}")
                        continue
                    test_case = _parse_single_case_from_dict(
                        case, f"test_case_{idx}", is_baseline, task_type
                    )
                    if test_case:
                        test_cases.append(test_case)
                except Exception as e:
                    log.warning(f"Skipping invalid test case at index {idx} in {report_file}: {e}")
        
        # Format 2: Object with 'test_cases' key
        elif 'test_cases' in report:
            for idx, case in enumerate(report['test_cases']):
                try:
                    if not isinstance(case, dict):
                        log.warning(f"Skipping non-dict test case at index {idx} in {report_file}")
                        continue
                    test_case = _parse_single_case_from_dict(
                        case, f"test_case_{idx}", is_baseline, task_type
                    )
                    if test_case:
                        test_cases.append(test_case)
                except Exception as e:
                    log.warning(f"Skipping invalid test case at index {idx} in {report_file}: {e}")
        
        # Format 3: Single object
        else:
            # Try standard parsing first
            test_case = _parse_single_case_from_dict(
                report, "test_case_0", is_baseline, task_type
            )
            
            if test_case:
                test_cases.append(test_case)
            else:
                # Fallback: Look for custom device-side *_ms timings.  Exclude
                # explicit host/wall timings; they include harness overhead and
                # should not be scored as kernel execution time.
                ms_keys = [
                    k for k in report.keys()
                    if k.endswith('_ms') and k not in _HOST_TIME_KEYS
                ]
                if ms_keys:
                    for idx, ms_key in enumerate(sorted(ms_keys)):
                        time_val = _safe_float(report.get(ms_key))
                        if time_val is None:
                            log.warning(f"Skipping invalid timing value {ms_key}={report.get(ms_key)!r} in {report_file}")
                            continue
                        # Build metadata excluding timing
                        exclude_keys = ['shape', 'shapes'] + [k for k in report.keys() if k.endswith('_ms')]
                        metadata = _build_metadata_from_case(report, exclude_keys)
                        
                        # Include other _ms keys in metadata for reference
                        other_timings = {k: report[k] for k in ms_keys if k != ms_key}
                        if other_timings:
                            metadata['other_timings'] = other_timings
                        
                        # Create descriptive test_case_id
                        key_name = ms_key.replace('_ms', '')
                        test_case_id = f"{key_name}_{idx}" if len(ms_keys) > 1 else key_name
                        
                        test_cases.append(TestCaseResult(
                            test_case_id=test_case_id,
                            shape=report.get('shape') or report.get('shapes'),
                            execution_time_ms=time_val,
                            metadata={
                                **metadata,
                                _SYNTHETIC_TEST_ID_METADATA_KEY: True,
                            }
                        ))
        
        log.info(f"Parsed {len(test_cases)} test case(s) from {report_file}")
        
    except Exception as e:
        log.warning(f"Failed to parse test cases from {report_file}: {e}")
    
    return test_cases


def parse_test_cases_from_stdout(
    output: str,
    logger: Optional[logging.Logger] = None
) -> List[TestCaseResult]:
    """
    Parse multiple test case results from stdout.
    
    Looks for patterns like:
    - "Test case 0: 123.45 ms"
    - "Shape [256, 256, 256]: 123.45 ms"
    - Multiple "Performance: X ms" lines
    
    Args:
        output: Command output text
        logger: Optional logger
        
    Returns:
        List of TestCaseResult objects
    """
    log = logger or logging.getLogger(__name__)
    test_cases = []

    benchmark_metadata: Dict[str, Any] = {}
    method_match = re.search(
        r'^\s*GEAK_BENCHMARK_METHOD=(.*)$', output, re.MULTILINE
    )
    if method_match:
        benchmark_metadata['benchmark_method'] = method_match.group(1).strip()
    fallback_match = re.search(
        r'^\s*GEAK_BENCHMARK_FALLBACK_REASON=(.*)$', output, re.MULTILINE
    )
    if fallback_match:
        benchmark_metadata['benchmark_fallback_reason'] = fallback_match.group(1).strip()

    # GEAK harnesses use a machine-readable token rather than a JSON report.
    geak_matches = re.findall(r'GEAK_RESULT_LATENCY_MS=([0-9.]+)', output)
    for idx, time_str in enumerate(geak_matches):
        test_cases.append(TestCaseResult(
            test_case_id=f"geak_{idx}",
            execution_time_ms=float(time_str),
            metadata={
                **benchmark_metadata,
                _SYNTHETIC_TEST_ID_METADATA_KEY: True,
                _TIMING_SOURCE_METADATA_KEY: "GEAK_RESULT_LATENCY_MS",
            },
        ))
    
    # Pattern 1: "Test case N: X ms" or "TestCase N: X ms"
    pattern1 = r'(?:Test\s+case|TestCase)\s+(\d+)[:\s]+([0-9.]+)\s*ms'
    if not test_cases:
        matches1 = re.findall(pattern1, output, re.IGNORECASE)
        for match in matches1:
            test_id, time_str = match
            test_cases.append(TestCaseResult(
                test_case_id=f"test_case_{test_id}",
                execution_time_ms=float(time_str),
                metadata={
                    **benchmark_metadata,
                    _SYNTHETIC_TEST_ID_METADATA_KEY: True,
                },
            ))
    
    # Pattern 2: "Shape [X, Y, Z]: X ms" or "shape: [X, Y, Z], time: X ms"
    pattern2 = r'(?:Shape|shape)[:\s]+\[([0-9,\s]+)\][:\s]+([0-9.]+)\s*ms'
    if not test_cases:
        matches2 = re.findall(pattern2, output, re.IGNORECASE)
        for idx, match in enumerate(matches2):
            shape_str, time_str = match
            shape = [int(x.strip()) for x in shape_str.split(',')]
            test_cases.append(TestCaseResult(
                test_case_id=f"shape_{idx}",
                shape=shape,
                execution_time_ms=float(time_str),
                metadata={
                    **benchmark_metadata,
                    _SYNTHETIC_TEST_ID_METADATA_KEY: True,
                },
            ))
    
    # Pattern 3: Multiple "Performance: X ms" lines (if no other pattern matched)
    if not test_cases:
        pattern3 = r'Performance:\s*([0-9.]+)\s*ms'
        matches3 = re.findall(pattern3, output, re.IGNORECASE)
        for idx, time_str in enumerate(matches3):
            test_cases.append(TestCaseResult(
                test_case_id=f"perf_{idx}",
                execution_time_ms=float(time_str),
                metadata={
                    **benchmark_metadata,
                    _SYNTHETIC_TEST_ID_METADATA_KEY: True,
                },
            ))
    
    log.info(f"Parsed {len(test_cases)} test case(s) from stdout")
    return test_cases


def match_test_cases(
    baseline_cases: List[TestCaseResult],
    optimized_cases: List[TestCaseResult],
    logger: Optional[logging.Logger] = None,
    allow_index_fallback: bool = True,
) -> List[Tuple[TestCaseResult, TestCaseResult]]:
    """
    Match test cases between baseline and optimized results.
    
    Matching strategy:
    1. Match by unique explicit test_case_id
    2. Match by unique params when present
    3. Match by unique shape when present
    4. Match by index for any remaining cases when allowed

    A key is only usable when it identifies exactly one remaining case on each
    side.  Ambiguous duplicate semantic keys are deferred to a later strategy
    instead of being paired greedily by input order.
    
    Args:
        baseline_cases: Baseline test case results
        optimized_cases: Optimized test case results
        logger: Optional logger
        allow_index_fallback: Whether to pair unmatched cases by remaining order
        
    Returns:
        List of (baseline_case, optimized_case) tuples
    """
    log = logger or logging.getLogger(__name__)
    matched = []
    used_baseline = set()
    used_optimized = set()

    def freeze_value(value):
        """Convert nested metadata values into comparable tuples."""
        if isinstance(value, dict):
            return tuple(sorted((k, freeze_value(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(freeze_value(v) for v in value)
        return value

    def get_params_key(case: TestCaseResult):
        params = case.metadata.get('params') if case.metadata else None
        if not params or not isinstance(params, dict):
            return None
        return freeze_value(params)

    def get_shape_key(case: TestCaseResult):
        if not case.shape:
            return None
        return freeze_value(case.shape)

    def get_explicit_id_key(case: TestCaseResult):
        if not case.test_case_id:
            return None
        metadata = case.metadata or {}
        if metadata.get(_SYNTHETIC_TEST_ID_METADATA_KEY, False):
            return None
        return case.test_case_id

    def remaining_cases(cases, used_ids):
        return [case for case in cases if id(case) not in used_ids]

    def match_by_unique_key(key_fn, strategy_name: str) -> None:
        nonlocal matched
        remaining_baseline = remaining_cases(baseline_cases, used_baseline)
        remaining_optimized = remaining_cases(optimized_cases, used_optimized)

        baseline_by_key = {}
        optimized_by_key = {}
        for base_case in remaining_baseline:
            base_key = key_fn(base_case)
            if base_key is not None:
                baseline_by_key.setdefault(base_key, []).append(base_case)
        for opt_case in remaining_optimized:
            opt_key = key_fn(opt_case)
            if opt_key is not None:
                optimized_by_key.setdefault(opt_key, []).append(opt_case)

        for base_case in remaining_baseline:
            base_key = key_fn(base_case)
            if base_key is None:
                continue
            baseline_candidates = baseline_by_key.get(base_key, [])
            optimized_candidates = optimized_by_key.get(base_key, [])
            if len(baseline_candidates) != 1 or len(optimized_candidates) != 1:
                continue

            opt_case = optimized_candidates[0]
            matched.append((base_case, opt_case))
            used_baseline.add(id(base_case))
            used_optimized.add(id(opt_case))

        if matched:
            log.debug(f"Matched {len(matched)} cumulative test case(s) after {strategy_name}")

    # Stable producer-supplied IDs are the strongest identity signal. Generated
    # parser IDs are excluded above because they merely encode result order.
    match_by_unique_key(get_explicit_id_key, "explicit test_case_id matching")
    match_by_unique_key(get_params_key, "params matching")
    match_by_unique_key(get_shape_key, "shape matching")

    if allow_index_fallback:
        remaining_baseline = remaining_cases(baseline_cases, used_baseline)
        remaining_optimized = remaining_cases(optimized_cases, used_optimized)

        if len(remaining_baseline) != len(remaining_optimized):
            log.warning(
                "Index fallback has different remaining case counts: "
                f"baseline={len(remaining_baseline)}, optimized={len(remaining_optimized)}"
            )

        min_len = min(len(remaining_baseline), len(remaining_optimized))
        for i in range(min_len):
            matched.append((remaining_baseline[i], remaining_optimized[i]))
            used_baseline.add(id(remaining_baseline[i]))
            used_optimized.add(id(remaining_optimized[i]))

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
    
    allow_index_fallback = (
        not require_complete_match
        or (len(baseline_cases) == 1 and len(optimized_cases) == 1)
    )
    matched = match_test_cases(
        baseline_cases,
        optimized_cases,
        logger,
        allow_index_fallback=allow_index_fallback,
    )
    
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
    
    avg_speedup = sum(speedups) / len(speedups)
    log.info(f"Average speedup across {len(speedups)} test case(s): {avg_speedup:.2f}x")
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
    comparable_methods = {'cuda_graph', 'cuda_event_fallback'}
    for base_case, opt_case in matched_cases:
        base_method = (base_case.metadata or {}).get('benchmark_method')
        opt_method = (opt_case.metadata or {}).get('benchmark_method')
        mixed_aggregate = (
            isinstance(base_method, str) and base_method.startswith('mixed:')
        ) or (
            isinstance(opt_method, str) and opt_method.startswith('mixed:')
        )
        missing_or_unknown = (
            base_method not in comparable_methods
            or opt_method not in comparable_methods
        )
        if (
            base_method == opt_method
            and not mixed_aggregate
            and not missing_or_unknown
        ):
            continue
        mismatch = {
            'test_case_id': base_case.test_case_id,
            'optimized_test_case_id': opt_case.test_case_id,
            'baseline_benchmark_method': base_method,
            'optimized_benchmark_method': opt_method,
        }
        if mixed_aggregate:
            mismatch['reason'] = 'ambiguous_mixed_aggregate'
        elif missing_or_unknown:
            mismatch['reason'] = 'missing_or_unknown_benchmark_method'
        mismatches.append(mismatch)
    return mismatches


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

    allow_index_fallback = (
        not require_complete_match
        or (len(baseline_cases) == 1 and len(optimized_cases) == 1)
    )
    matched = match_test_cases(
        baseline_cases,
        optimized_cases,
        logger,
        allow_index_fallback=allow_index_fallback,
    )
    if not matched:
        return False, []
    if require_complete_match and (
        len(matched) != len(baseline_cases) or len(matched) != len(optimized_cases)
    ):
        return False, []
    mismatches = _benchmark_method_mismatches(matched)
    return not mismatches, mismatches


def attach_event_fallback_baselines(
    baseline_cases: List[TestCaseResult],
    event_cases: List[TestCaseResult],
    logger: Optional[logging.Logger] = None,
) -> List[TestCaseResult]:
    """Attach forced-Event timing as an auditable alternate baseline.

    The normal baseline remains the graph-first result.  When an optimized
    callable genuinely cannot be captured, evaluation can select the paired
    Event timing for that exact case instead of comparing unlike methods.
    """

    log = logger or logging.getLogger(__name__)
    matched = match_test_cases(
        baseline_cases,
        event_cases,
        log,
        # Index matching is only safe for the unavoidable singleton legacy
        # format. Equal-length multi-case lists can be unrelated or reordered.
        allow_index_fallback=len(baseline_cases) == len(event_cases) == 1,
    )
    if len(matched) != len(baseline_cases) or len(matched) != len(event_cases):
        log.warning(
            "Could not attach complete Event baseline set: matched=%d, "
            "baseline=%d, event=%d",
            len(matched),
            len(baseline_cases),
            len(event_cases),
        )

    for baseline_case, event_case in matched:
        event_metadata = event_case.metadata or {}
        if event_metadata.get("benchmark_method") != "cuda_event_fallback":
            log.warning(
                "Ignoring non-Event alternate baseline for %s: %r",
                baseline_case.test_case_id,
                event_metadata.get("benchmark_method"),
            )
            continue
        if baseline_case.metadata is None:
            baseline_case.metadata = {}
        baseline_case.metadata[f"{_ALTERNATE_EVENT_PREFIX}time_ms"] = (
            event_case.execution_time_ms
        )
        for key, value in event_metadata.items():
            if key.startswith("benchmark_") and value is not None:
                suffix = key[len("benchmark_") :]
                baseline_case.metadata[f"{_ALTERNATE_EVENT_PREFIX}{suffix}"] = value
    return baseline_cases


def select_method_matched_baselines(
    baseline_cases: List[TestCaseResult],
    optimized_cases: List[TestCaseResult],
    logger: Optional[logging.Logger] = None,
) -> List[TestCaseResult]:
    """Select each baseline timing variant matching the optimized method."""

    log = logger or logging.getLogger(__name__)
    allow_index_fallback = len(baseline_cases) == len(optimized_cases) == 1
    matched = match_test_cases(
        baseline_cases,
        optimized_cases,
        log,
        allow_index_fallback=allow_index_fallback,
    )
    if len(matched) != len(baseline_cases) or len(matched) != len(optimized_cases):
        return baseline_cases

    selected: List[TestCaseResult] = []
    for baseline_case, optimized_case in matched:
        metadata = dict(baseline_case.metadata or {})
        execution_time_ms = baseline_case.execution_time_ms
        optimized_method = (optimized_case.metadata or {}).get("benchmark_method")
        baseline_method = metadata.get("benchmark_method")
        alternate_method = metadata.get(f"{_ALTERNATE_EVENT_PREFIX}method")
        alternate_time = _safe_float(
            metadata.get(f"{_ALTERNATE_EVENT_PREFIX}time_ms")
        )

        if (
            optimized_method == "cuda_event_fallback"
            and baseline_method != optimized_method
            and alternate_method == optimized_method
            and alternate_time is not None
            and alternate_time > 0.0
        ):
            execution_time_ms = alternate_time
            for key, value in list(metadata.items()):
                if not key.startswith(_ALTERNATE_EVENT_PREFIX):
                    continue
                suffix = key[len(_ALTERNATE_EVENT_PREFIX) :]
                if suffix != "time_ms":
                    metadata[f"benchmark_{suffix}"] = value
            metadata["benchmark_baseline_variant"] = "forced_event"

        selected.append(TestCaseResult(
            test_case_id=baseline_case.test_case_id,
            shape=baseline_case.shape,
            execution_time_ms=execution_time_ms,
            metadata=metadata,
        ))
    return selected


def save_performance_results(
    test_cases: List[TestCaseResult],
    workspace: Path,
    filename: str,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Save test case results to YAML file.
    
    Saves identifying fields, timing, params, and canonical ``benchmark_*``
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
        if case.shape:
            case_dict['shape'] = case.shape
        # Only include params from metadata, exclude everything else
        if case.metadata and 'params' in case.metadata:
            case_dict['params'] = case.metadata['params']
        # Persist all canonical benchmark metadata so baseline and optimized
        # measurements can be audited and compared after workspace reload.
        if case.metadata:
            for key, value in case.metadata.items():
                if key.startswith('benchmark_') and value is not None:
                    case_dict[key] = value
            # Preserve whether an ID was generated from list order. Without
            # this private provenance bit, a YAML reload would treat a
            # synthetic ID as an explicit stable key and could mispair cases.
            if case.metadata.get(_SYNTHETIC_TEST_ID_METADATA_KEY):
                case_dict[_SYNTHETIC_TEST_ID_METADATA_KEY] = True
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
